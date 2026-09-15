// =========================================
// =========================================
// crates/motionloom/src/mesh_authoring/mod.rs

//! Deterministic, filesystem-free MeshAsset construction and fitting sessions.

mod geometry;
mod schema;

pub use geometry::execute_geometry_recipe;
pub use schema::*;

use crate::ControlCageNode;
use crate::mesh_reference::{
    FeatureBinding, LandmarkBinding, MeshAssetProposal, MeshReferenceError,
    MeshReferenceEvaluation, MeshReferenceSet, analyze_image_reference, apply_mesh_asset_proposal,
    evaluate_mesh_asset_reference, mesh_source_fingerprint, mesh_topology_signature,
    validate_mesh_topology,
};
use std::collections::BTreeSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MeshAuthoringError {
    #[error("unsupported mesh-authoring schema version {0}")]
    SchemaVersion(String),
    #[error("duplicate operation id {0}")]
    DuplicateOperation(String),
    #[error("unknown semantic region {0}")]
    UnknownRegion(String),
    #[error("invalid geometry operation: {0}")]
    Operation(String),
    #[error("mesh exceeds authoring limit ({vertices} vertices, {faces} faces)")]
    LimitExceeded { vertices: usize, faces: usize },
    #[error("mesh topology validation failed")]
    Topology(crate::mesh_reference::MeshTopologyReport),
    #[error("session state conflict: {0}")]
    Session(String),
    #[error(transparent)]
    Reference(#[from] MeshReferenceError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub fn mesh_authoring_capabilities() -> MeshAuthoringCapabilities {
    MeshAuthoringCapabilities {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        operations: [
            "createLoop",
            "defineRegion",
            "loftLoops",
            "createSurface",
            "sweepProfile",
            "extrudeRegion",
            "thickenSurface",
            "transformRegion",
            "mirrorRegion",
            "weldVertices",
            "capBoundary",
            "generateUv",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        reference_functions: ["analyzeImageReference", "evaluateMeshAssetReference"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        proposal_functions: ["applyMeshAssetProposal", "applyMeshTopologyProposal"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        maximum_vertices: 30_000,
        maximum_faces: 30_000,
    }
}

pub fn mesh_authoring_schema_json() -> String {
    serde_json::json!({
        "schemaVersion": MESH_AUTHORING_SCHEMA_VERSION,
        "namespace": "motionloom::api::mesh_authoring",
        "capabilities": mesh_authoring_capabilities(),
        "functions": {
            "executeGeometryRecipe": {
                "argument": "GeometryRecipe",
                "result": "MeshAuthoringResult"
            },
            "applyMeshTopologyProposal": {
                "arguments": ["source", "ImageReferenceAnalysis[]", "MeshTopologyProposal", "MeshReferenceEvaluation"],
                "result": "ApplyTopologyProposalResult"
            },
            "analyzeReferences": {
                "arguments": ["MeshAuthoringSession", "ReferenceInput[]"],
                "result": "ImageReferenceAnalysis[]"
            },
            "evaluateSessionRevision": {
                "arguments": ["MeshAuthoringSession", "revisionId", "MeshReferenceSet"],
                "result": "MeshReferenceEvaluation",
                "async": true
            },
            "applyPositionProposalToSession": {
                "arguments": ["MeshAuthoringSession", "MeshAssetProposal"],
                "result": "SessionPositionProposalResult"
            },
            "applyTopologyProposalToSession": {
                "arguments": ["MeshAuthoringSession", "MeshTopologyProposal"],
                "result": "SessionTopologyProposalResult"
            },
            "decideCandidateRevision": {
                "arguments": ["MeshAuthoringSession", "revisionId", "primaryViewTolerance", "minimumImprovement"],
                "result": "RevisionDecision"
            }
        },
        "operationInputs": {
            "createLoop": ["spec.id", "center", "radiusA", "radiusB", "segments", "normalAxis"],
            "defineRegion": ["id", "vertices", "faces"],
            "loftLoops": ["id", "loops", "capStart", "capEnd"],
            "createSurface": ["id", "rows", "closeColumns"],
            "sweepProfile": ["id", "path", "radius", "segments"],
            "extrudeRegion": ["id", "region", "vector", "segments"],
            "thickenSurface": ["id", "region", "thickness"],
            "transformRegion": ["id", "region", "translate", "scale"],
            "mirrorRegion": ["id", "region", "axis", "weldCenter", "tolerance"],
            "weldVertices": ["id", "region", "tolerance"],
            "capBoundary": ["id", "loopId"],
            "generateUv": ["id", "region", "projection"]
        },
        "requiredProposalFingerprints": [
            "sourceFingerprint",
            "topologySignature",
            "cameraFingerprint",
            "referenceSetFingerprint",
            "metricProfileFingerprint",
            "evaluationFingerprint"
        ],
        "workflow": [
            "analyzeReferences",
            "executeGeometryRecipe",
            "validateTopology",
            "evaluateRevision",
            "applyPositionProposalOrTopologyProposal",
            "reevaluateCandidate",
            "acceptOrRejectRevision"
        ],
        "revisionPolicy": "immutable candidates with explicit accept or reject",
        "handlePolicy": "handles are scoped to a revisionId and topologySignature",
        "topologyPolicy": "topology proposals return correspondence and invalidate affected bindings",
        "stopReasons": [
            "qualityGatesPassed",
            "negligibleImprovement",
            "contradictoryViews",
            "movementBudgetExhausted",
            "topologyInsufficient"
        ]
    })
    .to_string()
}

pub fn execute_geometry_recipe_json(recipe_json: &str) -> Result<String, MeshAuthoringError> {
    let recipe = serde_json::from_str(recipe_json)?;
    Ok(serde_json::to_string(&execute_geometry_recipe(&recipe)?)?)
}

pub fn apply_mesh_topology_proposal_json(
    source: &str,
    analyses_json: &str,
    proposal_json: &str,
    evaluation_json: &str,
) -> Result<String, MeshAuthoringError> {
    let analyses: Vec<crate::mesh_reference::ImageReferenceAnalysis> =
        serde_json::from_str(analyses_json)?;
    let proposal = serde_json::from_str(proposal_json)?;
    let evaluation = serde_json::from_str(evaluation_json)?;
    Ok(serde_json::to_string(&apply_mesh_topology_proposal(
        source,
        &analyses,
        &proposal,
        &evaluation,
    )?)?)
}

pub fn mesh_asset_element(id: &str, material: &str, cage: &ControlCageNode) -> String {
    let mut source = format!(
        "<MeshAsset id=\"{id}\" material=\"{material}\" subdivision=\"{}\">",
        cage.subdivision
    );
    for (index, position) in cage.positions.iter().enumerate() {
        let uv = cage.uvs.get(index).copied().unwrap_or([0.0; 2]);
        let pinned = cage.pinned.get(index).copied().unwrap_or(false);
        source.push_str(&format!(
            "\n  <Vertex position={{[{},{},{}]}} uv={{[{},{}]}} pinned=\"{}\" />",
            position[0], position[1], position[2], uv[0], uv[1], pinned
        ));
    }
    for face in &cage.faces {
        let indices = face
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        source.push_str(&format!("\n  <Face indices={{[{indices}]}} />"));
    }
    source.push_str("\n</MeshAsset>");
    source
}

pub fn create_mesh_authoring_session(
    session_id: impl Into<String>,
    classification: ReconstructionClass,
    source: String,
    target_asset_id: impl Into<String>,
    target_model_id: impl Into<String>,
    assumptions: Vec<String>,
) -> Result<MeshAuthoringSession, MeshAuthoringError> {
    let target_asset_id = target_asset_id.into();
    let target_model_id = target_model_id.into();
    let cage = crate::mesh_reference::mesh_asset(&source, &target_asset_id)?;
    let topology = validate_mesh_topology(&cage, &Default::default());
    if !topology.valid {
        return Err(MeshAuthoringError::Topology(topology));
    }
    let revision = MeshRevision {
        id: "revision-0".into(),
        parent: None,
        status: RevisionStatus::Accepted,
        source_fingerprint: mesh_source_fingerprint(&source),
        topology_signature: mesh_topology_signature(&cage),
        source,
        evaluation: None,
        correspondence: TopologyCorrespondence::default(),
        reason: "initial accepted source".into(),
    };
    Ok(MeshAuthoringSession {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        session_id: session_id.into(),
        classification,
        target_asset_id,
        target_model_id,
        accepted_revision: revision.id.clone(),
        revisions: vec![revision],
        analyses: vec![],
        frozen_evaluation: None,
        assumptions,
    })
}

pub fn analyze_references(
    session: &mut MeshAuthoringSession,
    inputs: &[ReferenceInput],
) -> Result<Vec<crate::mesh_reference::ImageReferenceAnalysis>, MeshAuthoringError> {
    let mut image_ids = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    let mut analyses = Vec::with_capacity(inputs.len());
    for input in inputs {
        let analysis = analyze_image_reference(&input.image_bytes, &input.request)?;
        if !image_ids.insert(analysis.image_id.clone()) {
            return Err(MeshAuthoringError::Session(format!(
                "duplicate imageId {}",
                analysis.image_id
            )));
        }
        if !hashes.insert(analysis.content_hash.clone()) {
            return Err(MeshAuthoringError::Session(format!(
                "duplicate reference contentHash {}",
                analysis.content_hash
            )));
        }
        analyses.push(analysis);
    }
    session.analyses = analyses.clone();
    Ok(analyses)
}

pub async fn evaluate_session_revision(
    session: &mut MeshAuthoringSession,
    revision_id: &str,
    request: &MeshReferenceSet,
) -> Result<MeshReferenceEvaluation, MeshAuthoringError> {
    if request.target_asset_id != session.target_asset_id
        || request.target_model_id != session.target_model_id
    {
        return Err(MeshAuthoringError::Session(
            "evaluation target does not match the authoring session".into(),
        ));
    }
    if !session.analyses.is_empty() {
        let expected: BTreeSet<_> = session
            .analyses
            .iter()
            .map(|analysis| (&analysis.image_id, &analysis.content_hash))
            .collect();
        let supplied: BTreeSet<_> = request
            .references
            .iter()
            .map(|reference| {
                (
                    &reference.analysis.image_id,
                    &reference.analysis.content_hash,
                )
            })
            .collect();
        if expected != supplied {
            return Err(MeshAuthoringError::Session(
                "evaluation references do not match the analyzed reference set".into(),
            ));
        }
    }
    let source = revision(session, revision_id)?.source.clone();
    let evaluation = evaluate_mesh_asset_reference(&source, request).await?;
    if let Some(frozen) = &session.frozen_evaluation {
        if frozen.camera_fingerprint != evaluation.camera_fingerprint
            || frozen.reference_set_fingerprint != evaluation.reference_set_fingerprint
            || frozen.metric_profile_fingerprint != evaluation.metric_profile_fingerprint
        {
            return Err(MeshAuthoringError::Session(
                "camera, references, or metric profile changed after baseline".into(),
            ));
        }
    } else {
        session.frozen_evaluation = Some(FrozenEvaluationSetup {
            camera_fingerprint: evaluation.camera_fingerprint.clone(),
            reference_set_fingerprint: evaluation.reference_set_fingerprint.clone(),
            metric_profile_fingerprint: evaluation.metric_profile_fingerprint.clone(),
        });
    }
    revision_mut(session, revision_id)?.evaluation = Some(evaluation.clone());
    Ok(evaluation)
}

pub fn apply_position_proposal_to_session(
    session: &mut MeshAuthoringSession,
    proposal: &MeshAssetProposal,
) -> Result<SessionPositionProposalResult, MeshAuthoringError> {
    let accepted = accepted_revision(session)?.clone();
    require_accepted_evaluation(&accepted, proposal_fingerprints(proposal))?;
    let applied = apply_mesh_asset_proposal(&accepted.source, proposal)?;
    let revision_id = push_candidate(
        session,
        applied.source.clone(),
        applied.source_fingerprint.clone(),
        applied.topology_signature.clone(),
        TopologyCorrespondence::default(),
        proposal.reason.clone(),
    );
    Ok(SessionPositionProposalResult {
        revision_id,
        applied,
    })
}

pub fn apply_mesh_topology_proposal(
    source: &str,
    analyses: &[crate::mesh_reference::ImageReferenceAnalysis],
    proposal: &MeshTopologyProposal,
    expected: &MeshReferenceEvaluation,
) -> Result<ApplyTopologyProposalResult, MeshAuthoringError> {
    if proposal.schema_version != MESH_AUTHORING_SCHEMA_VERSION {
        return Err(MeshAuthoringError::SchemaVersion(
            proposal.schema_version.clone(),
        ));
    }
    if proposal.operations.is_empty() {
        return Err(MeshAuthoringError::Operation(
            "topology proposal contains no operations".into(),
        ));
    }
    if proposal.target_asset_id != expected.target_asset_id {
        return Err(MeshAuthoringError::Session(
            "topology proposal targets a different MeshAsset".into(),
        ));
    }
    if !(0.0..=1.0).contains(&proposal.confidence) || !proposal.confidence.is_finite() {
        return Err(MeshAuthoringError::Operation(
            "topology proposal confidence must be finite and within 0..=1".into(),
        ));
    }
    require_topology_fingerprints(proposal, expected)?;
    if mesh_source_fingerprint(source) != proposal.source_fingerprint {
        return Err(MeshAuthoringError::Session(
            "stale source fingerprint".into(),
        ));
    }
    let cage = crate::mesh_reference::mesh_asset(source, &proposal.target_asset_id)?;
    if mesh_topology_signature(&cage) != proposal.topology_signature {
        return Err(MeshAuthoringError::Session(
            "stale topology signature".into(),
        ));
    }
    let result = geometry::execute_operations_on_cage(
        cage,
        proposal.regions.clone(),
        &proposal.operations,
        proposal.validation.clone(),
    )?;
    let next_source = crate::mesh_reference::rewrite_mesh_asset_cage(
        source,
        &proposal.target_asset_id,
        &result.cage,
    )?;
    let reparsed = crate::mesh_reference::mesh_asset(&next_source, &proposal.target_asset_id)?;
    if reparsed != result.cage {
        return Err(MeshAuthoringError::Session(
            "rewritten MeshAsset does not reproduce the validated cage".into(),
        ));
    }
    Ok(ApplyTopologyProposalResult {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        source_fingerprint: mesh_source_fingerprint(&next_source),
        topology_signature: mesh_topology_signature(&reparsed),
        source: next_source,
        topology: result.topology,
        invalidated_feature_ids: invalidated_feature_ids(analyses),
        correspondence: result.correspondence,
    })
}

pub fn apply_topology_proposal_to_session(
    session: &mut MeshAuthoringSession,
    proposal: &MeshTopologyProposal,
) -> Result<SessionTopologyProposalResult, MeshAuthoringError> {
    let accepted = accepted_revision(session)?.clone();
    let evaluation = accepted
        .evaluation
        .as_ref()
        .ok_or_else(|| MeshAuthoringError::Session("accepted revision has no evaluation".into()))?;
    let applied =
        apply_mesh_topology_proposal(&accepted.source, &session.analyses, proposal, evaluation)?;
    let revision_id = push_candidate(
        session,
        applied.source.clone(),
        applied.source_fingerprint.clone(),
        applied.topology_signature.clone(),
        applied.correspondence.clone(),
        proposal.reason.clone(),
    );
    Ok(SessionTopologyProposalResult {
        revision_id,
        applied,
    })
}

pub fn reject_candidate_revision(
    session: &mut MeshAuthoringSession,
    revision_id: &str,
    reason: impl Into<String>,
) -> Result<(), MeshAuthoringError> {
    let candidate = revision_mut(session, revision_id)?;
    if candidate.status != RevisionStatus::Candidate {
        return Err(MeshAuthoringError::Session(format!(
            "revision {revision_id} is not a candidate"
        )));
    }
    candidate.status = RevisionStatus::Rejected;
    candidate.reason = reason.into();
    Ok(())
}

pub fn decide_candidate_revision(
    session: &mut MeshAuthoringSession,
    revision_id: &str,
    primary_view_tolerance: f32,
    minimum_improvement: f32,
) -> Result<RevisionDecision, MeshAuthoringError> {
    let baseline = accepted_revision(session)?.clone();
    let candidate = revision(session, revision_id)?.clone();
    if candidate.status != RevisionStatus::Candidate {
        return Err(MeshAuthoringError::Session(format!(
            "revision {revision_id} is not a candidate"
        )));
    }
    let before = baseline
        .evaluation
        .as_ref()
        .ok_or_else(|| MeshAuthoringError::Session("accepted revision has no evaluation".into()))?;
    let after = candidate.evaluation.as_ref().ok_or_else(|| {
        MeshAuthoringError::Session("candidate revision has no evaluation".into())
    })?;
    let before_views: BTreeSet<_> = before.views.iter().map(|view| &view.id).collect();
    let after_views: BTreeSet<_> = after.views.iter().map(|view| &view.id).collect();
    if after.source_fingerprint != candidate.source_fingerprint
        || after.topology_signature != candidate.topology_signature
        || before.camera_fingerprint != after.camera_fingerprint
        || before.reference_set_fingerprint != after.reference_set_fingerprint
        || before.metric_profile_fingerprint != after.metric_profile_fingerprint
        || before_views != after_views
    {
        return Err(MeshAuthoringError::Session(
            "candidate evaluation does not match its revision or frozen comparison setup".into(),
        ));
    }
    let maximum_regression = before
        .views
        .iter()
        .filter_map(|view| {
            after
                .views
                .iter()
                .find(|candidate_view| candidate_view.id == view.id)
                .map(|candidate_view| candidate_view.error - view.error)
        })
        .fold(0.0_f32, f32::max);
    let improved = before.objective - after.objective >= minimum_improvement.max(0.0);
    let accepted = improved && maximum_regression <= primary_view_tolerance.max(0.0);
    if accepted {
        revision_mut(session, &baseline.id)?.status = RevisionStatus::Superseded;
        revision_mut(session, revision_id)?.status = RevisionStatus::Accepted;
        session.accepted_revision = revision_id.into();
    } else {
        revision_mut(session, revision_id)?.status = RevisionStatus::Rejected;
    }
    Ok(RevisionDecision {
        revision_id: revision_id.into(),
        accepted,
        objective_before: before.objective,
        objective_after: after.objective,
        maximum_primary_view_regression: maximum_regression,
        reason: if accepted {
            "aggregate objective improved within the per-view tolerance".into()
        } else {
            "candidate did not improve enough or regressed a primary view".into()
        },
    })
}

pub fn artifact_manifest(root: &str) -> MeshAuthoringArtifactManifest {
    let root = root.trim_end_matches('/');
    MeshAuthoringArtifactManifest {
        session: format!("{root}/session.json"),
        recipe: format!("{root}/geometry-recipe.json"),
        accepted_source: format!("{root}/accepted/main.motionloom"),
        control_cage: format!("{root}/accepted/control-cage.json"),
        references: format!("{root}/references/references.json"),
        analyses: format!("{root}/analysis"),
        evaluations: format!("{root}/evaluations"),
        proposals: format!("{root}/proposals"),
        overlays: format!("{root}/overlays"),
        differences: format!("{root}/differences"),
        fit_summary: format!("{root}/fit-summary.json"),
    }
}

pub fn revision_handles(
    session: &MeshAuthoringSession,
    revision_id: &str,
) -> Result<Vec<MeshHandle>, MeshAuthoringError> {
    let revision = revision(session, revision_id)?;
    let cage = crate::mesh_reference::mesh_asset(&revision.source, &session.target_asset_id)?;
    let vertices = (0..cage.positions.len()).map(|index| MeshHandle {
        revision_id: revision_id.into(),
        topology_signature: revision.topology_signature.clone(),
        kind: MeshHandleKind::Vertex,
        index: index as u64,
    });
    let faces = (0..cage.faces.len()).map(|index| MeshHandle {
        revision_id: revision_id.into(),
        topology_signature: revision.topology_signature.clone(),
        kind: MeshHandleKind::Face,
        index: index as u64,
    });
    Ok(vertices.chain(faces).collect())
}

pub fn validate_handle(
    session: &MeshAuthoringSession,
    handle: &MeshHandle,
) -> Result<(), MeshAuthoringError> {
    let revision = revision(session, &handle.revision_id)?;
    if revision.topology_signature != handle.topology_signature {
        return Err(MeshAuthoringError::Session(
            "mesh handle has a stale topology signature".into(),
        ));
    }
    let cage = crate::mesh_reference::mesh_asset(&revision.source, &session.target_asset_id)?;
    let valid = match handle.kind {
        MeshHandleKind::Vertex => handle.index < cage.positions.len() as u64,
        MeshHandleKind::Face => handle.index < cage.faces.len() as u64,
    };
    if !valid {
        return Err(MeshAuthoringError::Session(
            "mesh handle index is out of range".into(),
        ));
    }
    Ok(())
}

pub fn stop_reason(
    session: &MeshAuthoringSession,
    conditions: &StopConditions,
) -> Option<StopReason> {
    let accepted = accepted_revision(session).ok()?;
    if accepted
        .evaluation
        .as_ref()
        .is_some_and(|evaluation| evaluation.passes_quality_gates)
    {
        return Some(StopReason::QualityGatesPassed);
    }
    if conditions.contradictory_views {
        return Some(StopReason::ContradictoryViews);
    }
    if conditions.movement_budget_exhausted {
        return Some(StopReason::MovementBudgetExhausted);
    }
    if conditions.topology_insufficient {
        return Some(StopReason::TopologyInsufficient);
    }
    let required = conditions.negligible_iteration_limit.max(1) + 1;
    let recent = conditions
        .recent_objectives
        .iter()
        .rev()
        .take(required)
        .copied()
        .collect::<Vec<_>>();
    if recent.len() == required
        && recent
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() <= conditions.negligible_improvement_threshold)
    {
        return Some(StopReason::NegligibleImprovement);
    }
    None
}

fn proposal_fingerprints(
    proposal: &MeshAssetProposal,
) -> (
    &str,
    &str,
    Option<&str>,
    Option<&str>,
    Option<&str>,
    Option<&str>,
) {
    (
        &proposal.source_fingerprint,
        &proposal.topology_signature,
        proposal.camera_fingerprint.as_deref(),
        proposal.reference_set_fingerprint.as_deref(),
        proposal.metric_profile_fingerprint.as_deref(),
        proposal.evaluation_fingerprint.as_deref(),
    )
}

fn require_accepted_evaluation(
    revision: &MeshRevision,
    fingerprints: (
        &str,
        &str,
        Option<&str>,
        Option<&str>,
        Option<&str>,
        Option<&str>,
    ),
) -> Result<(), MeshAuthoringError> {
    let evaluation = revision
        .evaluation
        .as_ref()
        .ok_or_else(|| MeshAuthoringError::Session("accepted revision has no evaluation".into()))?;
    let (source, topology, camera, references, metrics, evaluation_id) = fingerprints;
    if source != evaluation.source_fingerprint
        || topology != evaluation.topology_signature
        || camera != Some(evaluation.camera_fingerprint.as_str())
        || references != Some(evaluation.reference_set_fingerprint.as_str())
        || metrics != Some(evaluation.metric_profile_fingerprint.as_str())
        || evaluation_id != Some(evaluation.evaluation_fingerprint.as_str())
    {
        return Err(MeshAuthoringError::Session(
            "proposal does not match the accepted evaluation fingerprints".into(),
        ));
    }
    Ok(())
}

fn require_topology_fingerprints(
    proposal: &MeshTopologyProposal,
    evaluation: &MeshReferenceEvaluation,
) -> Result<(), MeshAuthoringError> {
    let coherent_evaluation_fingerprint = crate::mesh_reference::evaluation_context_fingerprint(
        &evaluation.source_fingerprint,
        &evaluation.topology_signature,
        &evaluation.camera_fingerprint,
        &evaluation.reference_set_fingerprint,
        &evaluation.metric_profile_fingerprint,
    );
    if proposal.source_fingerprint != evaluation.source_fingerprint
        || proposal.topology_signature != evaluation.topology_signature
        || proposal.camera_fingerprint != evaluation.camera_fingerprint
        || proposal.reference_set_fingerprint != evaluation.reference_set_fingerprint
        || proposal.metric_profile_fingerprint != evaluation.metric_profile_fingerprint
        || proposal.evaluation_fingerprint != evaluation.evaluation_fingerprint
        || evaluation.evaluation_fingerprint != coherent_evaluation_fingerprint
    {
        return Err(MeshAuthoringError::Session(
            "topology proposal has stale evaluation fingerprints".into(),
        ));
    }
    Ok(())
}

fn invalidated_feature_ids(
    analyses: &[crate::mesh_reference::ImageReferenceAnalysis],
) -> Vec<String> {
    let mut ids = BTreeSet::new();
    for analysis in analyses {
        for landmark in &analysis.landmarks {
            if !matches!(
                landmark.binding,
                None | Some(LandmarkBinding::NearestContour)
            ) {
                ids.insert(format!("{}:{}", analysis.image_id, landmark.id));
            }
        }
        for feature in &analysis.features {
            if !matches!(feature.binding, None | Some(FeatureBinding::NearestContour)) {
                ids.insert(format!("{}:{}", analysis.image_id, feature.id));
            }
        }
    }
    ids.into_iter().collect()
}

fn push_candidate(
    session: &mut MeshAuthoringSession,
    source: String,
    source_fingerprint: String,
    topology_signature: String,
    correspondence: TopologyCorrespondence,
    reason: String,
) -> String {
    let id = format!("revision-{}", session.revisions.len());
    session.revisions.push(MeshRevision {
        id: id.clone(),
        parent: Some(session.accepted_revision.clone()),
        status: RevisionStatus::Candidate,
        source,
        source_fingerprint,
        topology_signature,
        evaluation: None,
        correspondence,
        reason,
    });
    id
}

fn accepted_revision(session: &MeshAuthoringSession) -> Result<&MeshRevision, MeshAuthoringError> {
    revision(session, &session.accepted_revision)
}

fn revision<'a>(
    session: &'a MeshAuthoringSession,
    id: &str,
) -> Result<&'a MeshRevision, MeshAuthoringError> {
    session
        .revisions
        .iter()
        .find(|revision| revision.id == id)
        .ok_or_else(|| MeshAuthoringError::Session(format!("unknown revision {id}")))
}

fn revision_mut<'a>(
    session: &'a mut MeshAuthoringSession,
    id: &str,
) -> Result<&'a mut MeshRevision, MeshAuthoringError> {
    session
        .revisions
        .iter_mut()
        .find(|revision| revision.id == id)
        .ok_or_else(|| MeshAuthoringError::Session(format!("unknown revision {id}")))
}

#[cfg(test)]
mod tests;
