// =========================================
// =========================================
// crates/motionloom/src/mesh_authoring/tests.rs

use super::*;
use crate::mesh_reference::{
    MESH_REFERENCE_SCHEMA_VERSION, MeshAssetProposal, MeshObjectiveComponents,
    MeshReferenceDiagnostic, MeshReferenceEvaluation, MeshReferenceViewEvaluation,
    MeshVertexChange,
};

fn fusiform_recipe() -> GeometryRecipe {
    GeometryRecipe {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        id: "fusiform".into(),
        subdivision: 0,
        operations: vec![
            GeometryOperation::CreateLoop {
                spec: LoopSpec {
                    id: "nose".into(),
                    center: [-1.0, 0.0, 0.0],
                    radius_a: 0.05,
                    radius_b: 0.05,
                    segments: 8,
                    normal_axis: Axis::X,
                    rotation_degrees: 0.0,
                },
            },
            GeometryOperation::CreateLoop {
                spec: LoopSpec {
                    id: "body".into(),
                    center: [0.0, 0.0, 0.0],
                    radius_a: 0.5,
                    radius_b: 0.35,
                    segments: 8,
                    normal_axis: Axis::X,
                    rotation_degrees: 0.0,
                },
            },
            GeometryOperation::CreateLoop {
                spec: LoopSpec {
                    id: "tail".into(),
                    center: [1.0, 0.0, 0.0],
                    radius_a: 0.08,
                    radius_b: 0.08,
                    segments: 8,
                    normal_axis: Axis::X,
                    rotation_degrees: 0.0,
                },
            },
            GeometryOperation::LoftLoops {
                id: "body-volume".into(),
                loops: vec!["nose".into(), "body".into(), "tail".into()],
                cap_start: true,
                cap_end: true,
            },
            GeometryOperation::GenerateUv {
                id: "body-uv".into(),
                region: Some("body-volume".into()),
                projection: UvProjection::Cylindrical {
                    axis: Axis::X,
                    scale: [1.0, 1.0],
                    offset: [0.0, 0.0],
                },
            },
        ],
    }
}

fn graph_source(cage: &crate::ControlCageNode) -> String {
    format!(
        r##"<Graph fps={{24}} duration="1s" size={{[64,64]}}>
<Assets>
<MaterialAsset id="clay" baseColor="#888888" />
{}
</Assets>
<Background color="#111111" />
<Present from="scene" />
</Graph>"##,
        mesh_asset_element("shape", "clay", cage)
    )
}

fn evaluation(source: &str, cage: &crate::ControlCageNode) -> MeshReferenceEvaluation {
    let source_fingerprint = crate::mesh_reference::mesh_source_fingerprint(source);
    let topology_signature = crate::mesh_reference::mesh_topology_signature(cage);
    let camera_fingerprint = "camera".to_string();
    let reference_set_fingerprint = "references".to_string();
    let metric_profile_fingerprint = "metrics".to_string();
    let evaluation_fingerprint = crate::mesh_reference::evaluation_context_fingerprint(
        &source_fingerprint,
        &topology_signature,
        &camera_fingerprint,
        &reference_set_fingerprint,
        &metric_profile_fingerprint,
    );
    MeshReferenceEvaluation {
        schema_version: crate::mesh_reference::MESH_REFERENCE_SCHEMA_VERSION.into(),
        measurement: "test".into(),
        source_fingerprint,
        topology_signature,
        camera_fingerprint,
        reference_set_fingerprint,
        metric_profile_fingerprint,
        evaluation_fingerprint,
        target_asset_id: "shape".into(),
        target_model_id: "shape-model".into(),
        objective: 1.0,
        passes_quality_gates: false,
        views: Vec::<MeshReferenceViewEvaluation>::new(),
        diagnostics: Vec::<MeshReferenceDiagnostic>::new(),
    }
}

#[test]
fn recipe_builds_a_deterministic_valid_control_cage() {
    let first = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let second = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    assert!(first.topology.valid);
    assert_eq!(first.cage, second.cage);
    assert_eq!(first.topology_signature, second.topology_signature);
    assert_eq!(first.cage.positions.len(), 26);
    assert_eq!(first.cage.faces.len(), 32);
    assert!(first.cage.uvs.iter().any(|uv| *uv != [0.0, 0.0]));
}

#[test]
fn emitted_mesh_asset_round_trips_through_the_dsl() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let source = graph_source(&result.cage);
    let parsed = crate::mesh_reference::mesh_asset(&source, "shape").unwrap();
    assert_eq!(parsed, result.cage);
}

#[test]
fn topology_proposal_checks_all_fingerprints_and_rewrites_the_asset() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let source = graph_source(&result.cage);
    let expected = evaluation(&source, &result.cage);
    let mut proposal = MeshTopologyProposal {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        source_fingerprint: expected.source_fingerprint.clone(),
        topology_signature: expected.topology_signature.clone(),
        camera_fingerprint: expected.camera_fingerprint.clone(),
        reference_set_fingerprint: expected.reference_set_fingerprint.clone(),
        metric_profile_fingerprint: expected.metric_profile_fingerprint.clone(),
        evaluation_fingerprint: expected.evaluation_fingerprint.clone(),
        target_asset_id: "shape".into(),
        reason: "generate stable cylindrical UVs".into(),
        operations: vec![GeometryOperation::GenerateUv {
            id: "all-uv".into(),
            region: None,
            projection: UvProjection::Cylindrical {
                axis: Axis::X,
                scale: [1.0, 1.0],
                offset: [0.1, 0.2],
            },
        }],
        regions: vec![],
        validation: Default::default(),
        evidence_views: vec!["left".into(), "top".into()],
        confidence: 0.8,
    };
    let applied = apply_mesh_topology_proposal(&source, &[], &proposal, &expected).unwrap();
    assert_ne!(applied.source_fingerprint, expected.source_fingerprint);
    assert_eq!(applied.topology_signature, expected.topology_signature);
    assert!(applied.topology.valid);

    proposal.camera_fingerprint = "stale-camera".into();
    assert!(apply_mesh_topology_proposal(&source, &[], &proposal, &expected).is_err());
}

#[test]
fn candidate_decision_preserves_the_last_accepted_revision_on_regression() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let source = graph_source(&result.cage);
    let mut session = create_mesh_authoring_session(
        "test",
        ReconstructionClass::PartialMultiviewFit,
        source.clone(),
        "shape",
        "shape-model",
        vec!["right view is inferred".into()],
    )
    .unwrap();
    session.revisions[0].evaluation = Some(evaluation(&source, &result.cage));
    let mut candidate_evaluation = evaluation(&source, &result.cage);
    candidate_evaluation.objective = 1.1;
    session.revisions.push(MeshRevision {
        id: "revision-1".into(),
        parent: Some("revision-0".into()),
        status: RevisionStatus::Candidate,
        source,
        source_fingerprint: candidate_evaluation.source_fingerprint.clone(),
        topology_signature: candidate_evaluation.topology_signature.clone(),
        evaluation: Some(candidate_evaluation),
        correspondence: Default::default(),
        reason: "test regression".into(),
    });
    let decision = decide_candidate_revision(&mut session, "revision-1", 0.0, 0.001).unwrap();
    assert!(!decision.accepted);
    assert_eq!(session.accepted_revision, "revision-0");
    assert_eq!(session.revisions[1].status, RevisionStatus::Rejected);
}

#[test]
fn schema_exposes_the_complete_llm_workflow() {
    let schema = mesh_authoring_schema_json();
    assert!(schema.contains("executeGeometryRecipe"));
    assert!(schema.contains("applyMeshTopologyProposal"));
    assert!(schema.contains("acceptOrRejectRevision"));
    let _ = MeshObjectiveComponents::default();
}

#[test]
fn surface_thickening_creates_a_closed_valid_volume() {
    let recipe = GeometryRecipe {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        id: "panel".into(),
        subdivision: 0,
        operations: vec![
            GeometryOperation::CreateSurface {
                id: "surface".into(),
                rows: vec![
                    vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0]],
                    vec![[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                ],
                close_columns: false,
            },
            GeometryOperation::ThickenSurface {
                id: "shell".into(),
                region: "surface".into(),
                thickness: 0.2,
            },
        ],
    };
    let result = execute_geometry_recipe(&recipe).unwrap();
    assert!(result.topology.valid);
    assert_eq!(result.topology.open_edges, 0);
    assert_eq!(result.cage.faces.len(), 6);
}

#[test]
fn revision_scoped_handles_reject_a_stale_topology() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let source = graph_source(&result.cage);
    let session = create_mesh_authoring_session(
        "handles",
        ReconstructionClass::PartialMultiviewFit,
        source,
        "shape",
        "shape-model",
        vec![],
    )
    .unwrap();
    let mut handle = revision_handles(&session, "revision-0").unwrap()[0].clone();
    validate_handle(&session, &handle).unwrap();
    handle.topology_signature = "stale".into();
    assert!(validate_handle(&session, &handle).is_err());
}

#[test]
fn face_extrusion_returns_old_to_new_face_correspondence() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let authored = super::geometry::execute_operations_on_cage(
        result.cage,
        vec![SemanticRegion {
            id: "selected-face".into(),
            vertices: vec![],
            faces: vec![0],
        }],
        &[GeometryOperation::ExtrudeRegion {
            id: "extruded-face".into(),
            region: "selected-face".into(),
            vector: [-0.05, 0.0, 0.0],
            segments: 1,
        }],
        Default::default(),
    )
    .unwrap();
    assert!(authored.topology.valid);
    assert_eq!(authored.correspondence.old_to_new_faces.len(), 32);
    assert_eq!(
        authored.correspondence.invalidated_regions,
        ["selected-face"]
    );
    assert!(authored.regions.contains_key("extruded-face"));
}

#[test]
fn session_position_proposal_creates_an_immutable_candidate() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let source = graph_source(&result.cage);
    let baseline = evaluation(&source, &result.cage);
    let mut session = create_mesh_authoring_session(
        "position",
        ReconstructionClass::PartialMultiviewFit,
        source,
        "shape",
        "shape-model",
        vec![],
    )
    .unwrap();
    session.revisions[0].evaluation = Some(baseline.clone());
    let proposal = MeshAssetProposal {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        source_fingerprint: baseline.source_fingerprint.clone(),
        topology_signature: baseline.topology_signature.clone(),
        camera_fingerprint: Some(baseline.camera_fingerprint.clone()),
        reference_set_fingerprint: Some(baseline.reference_set_fingerprint.clone()),
        metric_profile_fingerprint: Some(baseline.metric_profile_fingerprint.clone()),
        evaluation_fingerprint: Some(baseline.evaluation_fingerprint.clone()),
        target_asset_id: "shape".into(),
        reason: "small measured profile correction".into(),
        changes: vec![MeshVertexChange {
            vertex: 8,
            before: result.cage.positions[8],
            after: [0.0, 0.49, 0.0],
            confidence: 0.8,
            evidence_views: vec!["left".into(), "top".into()],
        }],
        validation: Default::default(),
    };
    let applied = apply_position_proposal_to_session(&mut session, &proposal).unwrap();
    assert_eq!(applied.revision_id, "revision-1");
    assert_eq!(session.accepted_revision, "revision-0");
    assert_eq!(session.revisions[1].status, RevisionStatus::Candidate);
    assert_ne!(session.revisions[0].source, session.revisions[1].source);
}

#[test]
fn stop_conditions_report_quality_gates_before_iteration_limits() {
    let result = execute_geometry_recipe(&fusiform_recipe()).unwrap();
    let source = graph_source(&result.cage);
    let mut session = create_mesh_authoring_session(
        "stop",
        ReconstructionClass::PartialMultiviewFit,
        source.clone(),
        "shape",
        "shape-model",
        vec![],
    )
    .unwrap();
    let mut accepted = evaluation(&source, &result.cage);
    accepted.passes_quality_gates = true;
    session.revisions[0].evaluation = Some(accepted);
    let reason = stop_reason(
        &session,
        &StopConditions {
            negligible_improvement_threshold: 0.001,
            negligible_iteration_limit: 3,
            recent_objectives: vec![1.0, 1.0, 1.0, 1.0],
            contradictory_views: false,
            movement_budget_exhausted: false,
            topology_insufficient: false,
        },
    );
    assert_eq!(reason, Some(StopReason::QualityGatesPassed));
}
