// =========================================
// =========================================
// crates/motionloom-action-tool/src/target_reference.rs

//! Independent FBX evaluator used only for offline validation, never playback.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceComparison {
    pub evaluator: String,
    pub samples: usize,
    pub metrics: Metrics,
    pub keyed_metrics: Metrics,
    pub duration_error_sec: f64,
    pub passed: bool,
    pub first_hips_source: [f32; 3],
    pub first_hips_reference: [f32; 3],
}

pub(super) fn compare(
    source: &AnimationSource,
    clip: &AnimationClip,
    mapping: &BTreeMap<String, usize>,
    target: &TargetOptions,
) -> Result<Option<SourceComparison>, ActionToolError> {
    if !source
        .path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
    {
        return Ok(None);
    }
    let bytes = read(&source.path)?;
    let evaluated_source = source.backend == "ufbx-evaluated";
    let scene = ufbx::load_memory(
        &bytes,
        ufbx::LoadOpts {
            target_axes: if evaluated_source {
                ufbx::CoordinateAxes::right_handed_y_up()
            } else {
                Default::default()
            },
            target_unit_meters: 1.,
            ignore_geometry: true,
            ignore_embedded: true,
            load_external_files: false,
            ..Default::default()
        },
    )
    .map_err(|e| err(format!("independent ufbx load: {}", e.description)))?;
    let stack = scene
        .anim_stacks
        .iter()
        .find(|s| s.element.name.as_ref() == clip.name)
        .ok_or_else(|| err("independent evaluator clip not found"))?;
    let duration_error_sec = ((stack.time_end - stack.time_begin) - clip.duration_sec as f64).abs();
    let order = order(&source.nodes);
    let mut pairs = vec![];
    for (bone, &i) in mapping {
        let found = scene
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.element.name.as_ref() == source.nodes[i].name)
            .map(|(j, _)| j)
            .collect::<Vec<_>>();
        if found.len() != 1 {
            return Err(err(format!(
                "independent evaluator missing/ambiguous node {}",
                source.nodes[i].name
            )));
        }
        pairs.push((bone, i, found[0]));
    }
    let samples = (clip.duration_sec * 120.).ceil() as usize + 1;
    let mut metrics = Metrics::default();
    let mut keyed_metrics = Metrics::default();
    let mut first_hips_source = [0.; 3];
    let mut first_hips_reference = [0.; 3];
    for f in 0..samples {
        let t = (f as f32 / 120.).min(clip.duration_sec);
        let actual = source_world(source, clip, t, &order)?;
        let evaluated = ufbx::evaluate_scene(
            &scene,
            &stack.anim,
            t as f64 + stack.time_begin,
            Default::default(),
        )
        .map_err(|e| err(format!("independent evaluation: {}", e.description)))?;
        for (bone, i, j) in &pairs {
            let m = &evaluated.nodes[*j].node_to_world;
            // Evaluate independently in original FBX axes, then compare in the
            // tool's declared basis. ufbx's named "front" convention differs.
            let raw = [
                [m.m00, m.m01, m.m02],
                [m.m10, m.m11, m.m12],
                [m.m20, m.m21, m.m22],
            ];
            let c = source.world_basis;
            let mut matrix = [0_f32; 16];
            matrix[15] = 1.;
            for r in 0..3 {
                for col in 0..3 {
                    for (a, raw_row) in raw.iter().enumerate() {
                        for (b, raw_value) in raw_row.iter().enumerate() {
                            matrix[col * 4 + r] += c[r][a] * *raw_value as f32 * c[col][b];
                        }
                    }
                }
            }
            let p = world_vector(source, [m.m03 as f32, m.m13 as f32, m.m23 as f32]);
            matrix[12..15].copy_from_slice(&p);
            let (p, r) = matrix_error(actual[*i], matrix, 1000.);
            if f == 0 && bone.as_str() == "hips" {
                first_hips_source = actual[*i].p;
                first_hips_reference = [matrix[12], matrix[13], matrix[14]];
            }
            merge(
                &mut metrics,
                Metrics {
                    max_position_mm: p,
                    max_rotation_deg: r,
                    worst_bone: (*bone).clone(),
                    worst_time_sec: t,
                },
            );
            if f % 4 == 0 {
                merge(
                    &mut keyed_metrics,
                    Metrics {
                        max_position_mm: p,
                        max_rotation_deg: r,
                        worst_bone: (*bone).clone(),
                        worst_time_sec: t,
                    },
                );
            }
        }
    }
    let passed = duration_error_sec < 0.001
        && metrics.max_position_mm <= target.max_position_mm
        && metrics.max_rotation_deg <= target.max_rotation_deg;
    Ok(Some(SourceComparison {
        evaluator: "ufbx 0.10.1 / evaluate_scene / metres / declared source basis".into(),
        samples,
        metrics,
        keyed_metrics,
        duration_error_sec,
        passed,
        first_hips_source,
        first_hips_reference,
    }))
}
