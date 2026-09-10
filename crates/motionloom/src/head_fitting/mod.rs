// =========================================
// =========================================
// crates/motionloom/src/head_fitting/mod.rs

//! Opt-in, filesystem-free head authoring. Images and annotations never enter DSL.
mod projection;
mod source;
mod validation;

use crate::dsl::{PrimitiveAssetNode, PrimitiveGeometry, parse_graph_script};
pub use projection::comparison_svg;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use validation::validate_head_reference_set;

pub const SCHEMA_VERSION: &str = "1.0";
pub const MEASUREMENT: &str = "mesh triangle coverage; 128x128 CPU mask over [-1,1]^2; distances/head height; pixels via aligned reference height; Y up; +Z face; +X anatomical left; front u=X, left u=-Z, right u=Z, back u=-X; no implicit image mirroring";

#[derive(Debug, thiserror::Error)]
pub enum HeadFitError {
    #[error("invalid reference set: {0}")]
    References(String),
    #[error("invalid candidate: {0}")]
    Candidate(String),
    #[error("source or proposal mismatch: {0}")]
    Source(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum View {
    Front,
    Left,
    Right,
    Back,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Alignment {
    pub top: [f64; 2],
    pub bottom: [f64; 2],
    pub centerline: [[f64; 2]; 2],
}

fn one() -> f64 {
    1.0
}
fn enabled() -> bool {
    true
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Landmark {
    pub id: String,
    pub pixel: [f64; 2],
    #[serde(default = "one")]
    pub confidence: f64,
    #[serde(default = "one")]
    pub weight: f64,
    #[serde(default = "enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Contour {
    pub id: String,
    pub closed: bool,
    pub points: Vec<[f64; 2]>,
    #[serde(default = "one")]
    pub confidence: f64,
    #[serde(default = "one")]
    pub weight: f64,
    #[serde(default = "enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HeadReference {
    pub id: String,
    pub image_id: String,
    pub view: View,
    pub projection: String,
    pub image_size: [u32; 2],
    pub alignment: Alignment,
    #[serde(default)]
    pub landmarks: Vec<Landmark>,
    #[serde(default)]
    pub contours: Vec<Contour>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct FitOptions {
    pub preserve_height: bool,
    pub symmetry: String,
    pub max_iterations: usize,
    pub allowed_parameters: Vec<String>,
    pub locked_parameters: Vec<String>,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self {
            preserve_height: true,
            symmetry: "x".into(),
            max_iterations: 80,
            allowed_parameters: PARAMETERS.iter().map(|p| p.0.into()).collect(),
            locked_parameters: vec![],
        }
    }
}
// Missing individual options retain the same defaults as the typed API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HeadReferenceSet {
    pub schema_version: String,
    pub target_asset_id: String,
    pub references: Vec<HeadReference>,
    #[serde(default)]
    pub options: FitOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    pub severity: String,
    pub code: String,
    pub message: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReferenceValidation {
    pub schema_version: String,
    pub measurement: String,
    pub valid: bool,
    pub diagnostics: Vec<Diagnostic>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandmarkFit {
    pub id: String,
    pub reference: [f64; 2],
    pub candidate: Option<[f64; 2]>,
    pub distance: Option<f64>,
    pub pixel_distance: Option<f64>,
    pub method: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CurveFit {
    pub id: String,
    pub reference: Vec<[f64; 2]>,
    pub candidate: Option<Vec<[f64; 2]>>,
    pub distance: Option<f64>,
    pub pixel_distance: Option<f64>,
    pub method: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewFit {
    pub id: String,
    pub view: View,
    pub contour_distance: Option<f64>,
    pub contour_pixel_distance: Option<f64>,
    pub mask_iou: Option<f64>,
    pub reference_width_over_height: Option<f64>,
    pub candidate_width_over_height: f64,
    pub reference_contour: Vec<[f64; 2]>,
    pub candidate_contour: Vec<[f64; 2]>,
    pub landmarks: Vec<LandmarkFit>,
    pub curves: Vec<CurveFit>,
    pub error: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitMetrics {
    pub schema_version: String,
    pub measurement: String,
    pub objective: f64,
    pub width_over_height: f64,
    pub depth_over_height: f64,
    pub mesh_height: f64,
    pub geometry_penalty: f64,
    pub views: Vec<ViewFit>,
    pub diagnostics: Vec<Diagnostic>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadFitChange {
    pub node_id: String,
    pub child_tag: String,
    pub attribute: String,
    pub before: serde_json::Value,
    pub after: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadFitProposal {
    pub schema_version: String,
    pub measurement: String,
    pub target_asset_id: String,
    pub status: String,
    pub stop_reason: String,
    pub iterations: usize,
    pub source_fingerprint: String,
    pub changes: Vec<HeadFitChange>,
    pub before_metrics: FitMetrics,
    pub after_metrics: FitMetrics,
    pub diagnostics: Vec<Diagnostic>,
    pub candidate_dsl: String,
}

// Stage order is stable; conservative bounds exclude collapsed and extreme heads.
const PARAMETERS: &[(&str, f32, f32)] = &[
    ("HeadShape.width", 0.45, 1.5),
    ("HeadShape.depth", 0.45, 1.5),
    ("HeadShape.forehead", 0.65, 1.5),
    ("HeadShape.cheekWidth", 0.65, 1.4),
    ("HeadShape.jawWidth", 0.45, 1.4),
    ("HeadShape.chinLength", 0.0, 0.25),
    ("HeadShape.chinRoundness", 0.0, 1.0),
    ("Eye.positionY", -0.2, 0.35),
    ("Eye.pairSpacing", 0.3, 1.1),
    ("Nose.projection", 0.0, 0.45),
    ("Nose.length", 0.08, 0.6),
    ("Mouth.positionY", -0.65, -0.2),
    ("Eye.width", 0.12, 0.65),
    ("Eye.opening", 0.04, 0.35),
    ("Eye.tilt", -0.25, 0.25),
    ("Iris.radius", 0.025, 0.18),
    ("Iris.pupilRadius", 0.0, 0.12),
    ("Mouth.width", 0.08, 0.65),
    ("Mouth.opening", 0.0, 0.18),
    ("Mouth.upperLip", 0.0, 0.12),
    ("Mouth.lowerLip", 0.0, 0.15),
    ("Eye.socketWidth", 0.18, 0.90),
    ("Eye.socketHeight", 0.12, 0.80),
    ("Eye.socketDepth", 0.0, 0.40),
    ("Nose.positionY", -0.65, 0.25),
];

fn fingerprint(s: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(s.as_bytes()))
}
fn diagnostic(severity: &str, code: &str, message: impl Into<String>) -> Diagnostic {
    Diagnostic {
        severity: severity.into(),
        code: code.into(),
        message: message.into(),
    }
}
fn validated(request: &HeadReferenceSet) -> Result<ReferenceValidation, HeadFitError> {
    let report = validate_head_reference_set(request);
    if !report.valid {
        return Err(HeadFitError::References(serde_json::to_string(&report)?));
    }
    Ok(report)
}
fn asset(source: &str, id: &str) -> Result<PrimitiveAssetNode, HeadFitError> {
    let graph = parse_graph_script(source).map_err(|e| HeadFitError::Candidate(e.to_string()))?;
    let targets: Vec<_> = graph.assets.iter().filter(|a| a.id == id).collect();
    if targets.len() != 1 {
        return Err(HeadFitError::Source(
            "target must exist exactly once".into(),
        ));
    }
    let a = targets[0]
        .primitive()
        .ok_or_else(|| HeadFitError::Candidate("target is not a head".into()))?;
    if let PrimitiveGeometry::HeadSurface {
        segments, rings, ..
    } = &a.geometry
    {
        if *segments > 256 || *rings > 256 {
            return Err(HeadFitError::Candidate(
                "analysis mesh limit is 256 segments/rings".into(),
            ));
        }
    } else {
        return Err(HeadFitError::Candidate("target is not HeadAsset".into()));
    }
    Ok(a.clone())
}
fn check_dsl(s: &str) -> Result<(), HeadFitError> {
    let r = crate::authoring::analyze_motionloom_script(s);
    if !r.parse_succeeded || !r.compile_succeeded || r.summary.errors != 0 {
        return Err(HeadFitError::Candidate(serde_json::to_string(
            &r.diagnostics,
        )?));
    }
    Ok(())
}

/// Measure actual generated geometry, without loading image IDs or running rendering.
pub fn evaluate_head_reference_fit(
    s: &str,
    r: &HeadReferenceSet,
) -> Result<FitMetrics, HeadFitError> {
    let validation = validated(r)?;
    let a = asset(s, &r.target_asset_id)?;
    projection::evaluate(&a, r, &projection::anchors(&a), validation.diagnostics)
}

/// Deterministic bounded coordinate descent. Cancellation is checked before each trial.
pub fn fit_head_asset_to_references(
    s: &str,
    r: &HeadReferenceSet,
    mut cancelled: impl FnMut() -> bool,
) -> Result<HeadFitProposal, HeadFitError> {
    let validation = validated(r)?;
    check_dsl(s)?;
    source::target_range(s, &r.target_asset_id)?;
    let original = asset(s, &r.target_asset_id)?;
    if matches!(
        &original.geometry,
        PrimitiveGeometry::HeadSurface { topology, .. } if topology == "explicit"
    ) {
        return Err(HeadFitError::Candidate(
            "topology=explicit can be compared but cannot be parameter-fitted; edit its Vertex data or use topology=facialCage".into(),
        ));
    }
    if r.options.symmetry == "x" {
        if !original.modifiers.is_empty() {
            return Err(HeadFitError::Candidate(
                "symmetry=x with modifiers is unsupported; evaluate first or use symmetry=none"
                    .into(),
            ));
        }

        if let PrimitiveGeometry::HeadSurface { features, .. } = &original.geometry {
            if features
                .iter()
                .any(|f| f.center[0].abs() > 1e-5 && !f.mirror_x)
            {
                return Err(HeadFitError::Candidate("symmetry=x requires mirrored off-axis features; author symmetric geometry first".into()));
            }
        }
    }
    let anchors = projection::anchors(&original);
    let before = projection::evaluate(&original, r, &anchors, validation.diagnostics.clone())?;
    let mut best = original.clone();
    let mut metrics = before.clone();
    let mut best_score = before.objective;
    let mut stop = "iteration_limit";
    let mut iterations = 0;
    let mut step = 0.06f32;
    let mut boundary = false;
    'search: for iteration in 0..r.options.max_iterations {
        iterations = iteration + 1;
        let mut improved = false;
        for (i, &(name, low, high)) in PARAMETERS.iter().enumerate() {
            if !r.options.allowed_parameters.iter().any(|p| p == name)
                || r.options.locked_parameters.iter().any(|p| p == name)
            {
                continue;
            }
            let Some(start) = parameter(&best, i) else {
                continue;
            };
            for sign in [-1.0, 1.0] {
                if cancelled() {
                    stop = "cancelled";
                    break 'search;
                }
                let value = (start + sign * step).clamp(low, high);
                boundary |= value == low || value == high;
                let mut trial = best.clone();
                set_parameter(&mut trial, i, value);
                // An optimizer trial must satisfy the same patch constraints as parsed DSL.
                if let PrimitiveGeometry::HeadSurface {
                    topology,
                    facial_cage: Some(settings),
                    head_profile,
                    head_dome,
                    face_layout: Some(layout),
                    ..
                } = &trial.geometry
                {
                    if topology == "facialcage"
                        && crate::world::primitive::validate_facial_layout(
                            settings,
                            head_profile,
                            head_dome.as_ref(),
                            layout,
                        )
                        .is_err()
                    {
                        continue;
                    }
                }
                let Ok(m) = projection::evaluate(&trial, r, &anchors, vec![]) else {
                    continue;
                };
                // Reject height drift instead of silently changing locked vertical scale.
                if r.options.preserve_height
                    && (m.mesh_height / before.mesh_height - 1.0).abs() > 0.005
                {
                    continue;
                }
                let regularization: f64 = (0..PARAMETERS.len())
                    .filter_map(|j| Some((parameter(&trial, j)? - parameter(&original, j)?) as f64))
                    .map(|v| v * v * 0.002)
                    .sum();
                let score = m.objective + regularization;
                if score + 1e-8 < best_score && m.objective <= before.objective {
                    best = trial;
                    metrics = m;
                    best_score = score;
                    improved = true;
                }
            }
        }
        if !improved {
            step *= 0.5;
            if step < 0.003 {
                stop = if boundary {
                    "parameter_boundary"
                } else if best_score < before.objective {
                    "converged"
                } else {
                    "no_improvement"
                };
                break;
            }
        }
    }
    let changes = source::changes(&original, &best, &r.target_asset_id);
    let candidate_dsl = source::patch(s, &r.target_asset_id, &changes)?;
    check_dsl(&candidate_dsl)?;
    // Re-measure the parsed output so text precision cannot invalidate the proposal.
    metrics = projection::evaluate(
        &asset(&candidate_dsl, &r.target_asset_id)?,
        r,
        &anchors,
        metrics.diagnostics,
    )?;
    if metrics.objective > before.objective + 1e-9 {
        return Err(HeadFitError::Candidate(
            "serialized candidate regressed".into(),
        ));
    }
    let mut diagnostics = validation.diagnostics;
    diagnostics.push(diagnostic("info", "MODEL_LIMIT", "Ellipsoid with local displacement; eyes/mouth corners unsupported. Stagnation is not proof of geometric incapacity; inspect held-out three-quarter view, occlusion and conflicting annotations."));
    Ok(HeadFitProposal {
        schema_version: SCHEMA_VERSION.into(),
        measurement: MEASUREMENT.into(),
        target_asset_id: r.target_asset_id.clone(),
        status: if changes.is_empty() {
            "unchanged"
        } else {
            "improved"
        }
        .into(),
        stop_reason: stop.into(),
        iterations,
        source_fingerprint: fingerprint(s),
        changes,
        before_metrics: before,
        after_metrics: metrics,
        diagnostics,
        candidate_dsl,
    })
}

/// Replay only the reviewed head edits; reject stale sources and altered candidate text.
pub fn apply_head_fit_proposal(s: &str, p: &HeadFitProposal) -> Result<String, HeadFitError> {
    if p.schema_version != SCHEMA_VERSION || p.source_fingerprint != fingerprint(s) {
        return Err(HeadFitError::Source(
            "version or fingerprint mismatch".into(),
        ));
    }
    let result = source::patch(s, &p.target_asset_id, &p.changes)?;
    if result != p.candidate_dsl {
        return Err(HeadFitError::Source(
            "candidate differs from reviewed changes".into(),
        ));
    }
    let old = asset(s, &p.target_asset_id)?;
    let new = asset(&result, &p.target_asset_id)?;
    if serde_json::to_value(source::changes(&old, &new, &p.target_asset_id))?
        != serde_json::to_value(&p.changes)?
    {
        return Err(HeadFitError::Source(
            "before/after values do not match source".into(),
        ));
    }
    check_dsl(&result)?;
    Ok(result)
}

/// JSON transport wrappers share the native computation with WASM.
pub fn validate_head_reference_set_json(r: &str) -> Result<String, HeadFitError> {
    Ok(serde_json::to_string(&validate_head_reference_set(
        &serde_json::from_str(r)?,
    ))?)
}
pub fn evaluate_head_reference_fit_json(s: &str, r: &str) -> Result<String, HeadFitError> {
    Ok(serde_json::to_string(&evaluate_head_reference_fit(
        s,
        &serde_json::from_str(r)?,
    )?)?)
}
pub fn fit_head_asset_to_references_json(s: &str, r: &str) -> Result<String, HeadFitError> {
    Ok(serde_json::to_string(&fit_head_asset_to_references(
        s,
        &serde_json::from_str(r)?,
        || false,
    )?)?)
}
pub fn apply_head_fit_proposal_json(s: &str, p: &str) -> Result<String, HeadFitError> {
    apply_head_fit_proposal(s, &serde_json::from_str(p)?)
}

// Access existing parameters only; dimensions are expressed relative to authored height.
fn parameter(a: &PrimitiveAssetNode, i: usize) -> Option<f32> {
    let PrimitiveGeometry::HeadSurface {
        head_shape: s,
        face_layout: f,
        ..
    } = &a.geometry
    else {
        return None;
    };
    Some(match i {
        0 => s.size[0] / s.size[1],
        1 => s.size[2] / s.size[1],
        2 => s.forehead,
        3 => s.cheek_width,
        4 => s.jaw_width,
        5 => s.chin_length,
        6 => s.chin_roundness,
        7 => f.as_ref()?.eyes.first()?.position[1],
        8 => f.as_ref()?.eyes.first()?.position[0].abs() * 2.0,
        9 => f.as_ref()?.noses.first()?.projection,
        10 => f.as_ref()?.noses.first()?.length,
        11 => f.as_ref()?.mouths.first()?.position[1],
        12 => f.as_ref()?.eyes.first()?.width,
        13 => f.as_ref()?.eyes.first()?.opening,
        14 => f.as_ref()?.eyes.first()?.tilt,
        15 => f.as_ref()?.eyes.first()?.iris.as_ref()?.radius,
        16 => f.as_ref()?.eyes.first()?.iris.as_ref()?.pupil_radius,
        17 => f.as_ref()?.eyes.first()?.position[1],
        18 => f.as_ref()?.eyes.first()?.position[0].abs() * 2.0,
        19 => f.as_ref()?.noses.first()?.projection,
        20 => f.as_ref()?.mouths.first()?.lower_lip,
        21 => f.as_ref()?.eyes.first()?.socket_width,
        22 => f.as_ref()?.eyes.first()?.socket_height,
        23 => f.as_ref()?.eyes.first()?.socket_depth,
        24 => f.as_ref()?.noses.first()?.position[1],
        _ => return None,
    })
}
fn set_parameter(a: &mut PrimitiveAssetNode, i: usize, v: f32) {
    let PrimitiveGeometry::HeadSurface {
        head_shape: s,
        face_layout: f,
        ..
    } = &mut a.geometry
    else {
        return;
    };
    match i {
        0 => s.size[0] = v * s.size[1],
        1 => s.size[2] = v * s.size[1],
        2 => s.forehead = v,
        3 => s.cheek_width = v,
        4 => s.jaw_width = v,
        5 => s.chin_length = v,
        6 => s.chin_roundness = v,
        _ => {
            if let Some(f) = f {
                match i {
                    7 => {
                        for c in &mut f.eyes {
                            c.position[1] = v;
                        }
                    }
                    8 => {
                        for c in &mut f.eyes {
                            c.position[0] = c.position[0].signum() * v * 0.5;
                        }
                    }
                    9 => {
                        for c in &mut f.noses {
                            c.projection = v;
                        }
                    }
                    10 => {
                        for c in &mut f.noses {
                            c.length = v;
                        }
                    }
                    11 => {
                        for c in &mut f.mouths {
                            c.position[1] = v;
                        }
                    }
                    12 => {
                        for c in &mut f.eyes {
                            c.width = v;
                        }
                    }
                    13 => {
                        for c in &mut f.eyes {
                            c.opening = v;
                        }
                    }
                    14 => {
                        for c in &mut f.eyes {
                            c.tilt = v;
                        }
                    }
                    15 => {
                        for c in &mut f.eyes {
                            if let Some(iris) = &mut c.iris {
                                iris.radius = v;
                            }
                        }
                    }
                    16 => {
                        for c in &mut f.eyes {
                            if let Some(iris) = &mut c.iris {
                                iris.pupil_radius = v.min(iris.radius);
                            }
                        }
                    }
                    17 => {
                        for c in &mut f.eyes {
                            c.position[1] = v;
                        }
                    }
                    18 => {
                        for c in &mut f.eyes {
                            c.position[0] = c.position[0].signum() * v * 0.5;
                        }
                    }
                    19 => {
                        for c in &mut f.noses {
                            c.projection = v;
                        }
                    }
                    20 => {
                        for c in &mut f.mouths {
                            c.lower_lip = v;
                        }
                    }
                    21 => {
                        for c in &mut f.eyes {
                            c.socket_width = v;
                        }
                    }
                    22 => {
                        for c in &mut f.eyes {
                            c.socket_height = v;
                        }
                    }
                    23 => {
                        for c in &mut f.eyes {
                            c.socket_depth = v;
                        }
                    }
                    24 => {
                        for c in &mut f.noses {
                            c.position[1] = v;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Make separate visual-review scenes. Perspective GPU views are not fitting measurements.
pub fn head_fit_preview_scripts(s: &str, id: &str) -> Result<Vec<(String, String)>, HeadFitError> {
    let a = asset(s, id)?;
    let block = &s[source::target_range(s, id)?];
    let material = a.material.as_deref().unwrap_or("head_review_clay");
    let head_height = crate::world::primitive::generate_primitive_mesh(&a).bounds_max[1]
        - crate::world::primitive::generate_primitive_mesh(&a).bounds_min[1];
    let distance = head_height * 2.4;
    let mut out = vec![];
    for (name, x, z) in [
        ("front", 0.0, 1.0),
        ("left", 1.0, 0.0),
        ("back", 0.0, -1.0),
        ("three-quarter", 0.70710677, 0.70710677),
    ] {
        let script=format!(r##"<Graph fps={{30}} duration="1s" size={{[512,512]}}>
<Assets><MaterialAsset id={material:?} baseColor="#C8B8AD" roughness="0.75" />
{block}
</Assets>
<Background color="#17202D" />
<Scene id="head_fit_review"><Timeline><Track id="head_fit_track" space="3d">
<Sequence from="0s" duration="1s"><CompositeGroup id="head_fit_stage" space="3d" depth="true">
<Camera3D position={{[{x},0,{z}]}} target={{[0,0,0]}} fov="34" />
<DirectionalLight direction={{[-0.4,-0.6,-1]}} intensity="2.2" />
<Model id="head_fit_model" asset={id:?} />
</CompositeGroup></Sequence></Track></Timeline></Scene><Present from="head_fit_review" /></Graph>"##,x=x*distance,z=z*distance).replace("><", ">\n<");
        check_dsl(&script)?;
        out.push((name.into(), script));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
