// =========================================
// =========================================
// crates/motionloom/src/authoring.rs

//! Machine-readable authoring analysis for LLM repair loops and showcase learning.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::compat::{GpuCompatibilitySeverity, inspect_gpu_compatibility};
use crate::dsl::{GraphAssetKind, GraphScript, parse_graph_script};
use crate::process::runtime::compile_runtime_program;
use crate::scene::animation::{AnimationDiagnosticSeverity, inspect_animation_targets};

const REPORT_VERSION: &str = "1.0";
const SCHEMA_VERSION: &str = "1.0";

/// Overall repair state for one authored script.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthoringStatus {
    Clean,
    NeedsReview,
    NeedsRepair,
    Unrenderable,
}

/// Severity shared by parse, semantic, compatibility, and authoring diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AuthoringDiagnosticSeverity {
    Error,
    Warning,
    Info,
}

/// One concrete repair action that an LLM can apply or explain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoringSuggestion {
    pub kind: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// A source-addressed diagnostic designed for deterministic LLM repair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoringDiagnostic {
    pub severity: AuthoringDiagnosticSeverity,
    pub code: String,
    pub phase: String,
    pub line: usize,
    pub column: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authored_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_value: Option<String>,
    pub message: String,
    pub effect: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggestions: Vec<AuthoringSuggestion>,
}

/// Stable counters that let hosts decide whether automatic rendering is safe.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoringSummary {
    pub errors: usize,
    pub warnings: usize,
    pub info: usize,
    pub ignored_attributes: usize,
    pub unknown_tags: usize,
    pub no_op_nodes: usize,
    pub animation_conflicts: usize,
    pub missing_assets: usize,
}

/// A compact description of what the parser accepted and the renderer will see.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EffectiveGraphSummary {
    pub fps: f32,
    pub duration_ms: u64,
    pub size: [u32; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render_size: Option<[u32; 2]>,
    pub scenes: usize,
    pub scene_nodes: usize,
    pub assets: usize,
    pub models: usize,
    pub cameras_3d: usize,
    pub environment_lights: usize,
    pub active_environment_lights: usize,
    pub animation_targets: usize,
    pub process_passes: usize,
    pub present_from: String,
}

/// One attribute observed in a showcase, enriched with known engine capability data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShowcaseAttributeSchema {
    pub name: String,
    pub occurrences: usize,
    pub examples: Vec<String>,
    pub recognized: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_inline_expression: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_animation_target: Option<bool>,
}

/// The small language slice demonstrated by one showcase script.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShowcaseTagSchema {
    pub tag: String,
    pub occurrences: usize,
    pub recognized: bool,
    pub validation_coverage: String,
    pub attributes: Vec<ShowcaseAttributeSchema>,
}

/// Per-showcase schema used by an LLM to learn the syntax actually present in an example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionLoomShowcaseSchema {
    pub schema_version: String,
    pub engine_version: String,
    pub root_tag: Option<String>,
    pub tags: Vec<ShowcaseTagSchema>,
    pub animation_properties: Vec<String>,
    pub asset_kinds: Vec<String>,
}

/// Complete parse/semantic report returned to an LLM after every authored revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionLoomAuthoringReport {
    pub report_version: String,
    pub engine_version: String,
    pub target: String,
    pub status: AuthoringStatus,
    pub parse_succeeded: bool,
    pub compile_succeeded: bool,
    pub renderable: bool,
    pub summary: AuthoringSummary,
    pub diagnostics: Vec<AuthoringDiagnostic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_graph: Option<EffectiveGraphSummary>,
    pub showcase_schema: MotionLoomShowcaseSchema,
}

#[derive(Debug, Clone)]
struct ScannedAttribute {
    name: String,
    value: Option<String>,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone)]
struct ScannedTag {
    name: String,
    attributes: Vec<ScannedAttribute>,
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy)]
struct TagCapability {
    attributes: &'static [&'static str],
    open_attributes: bool,
}

/// Analyze a script without changing permissive parser behavior used by existing projects.
pub fn analyze_motionloom_script(script: &str) -> MotionLoomAuthoringReport {
    analyze_motionloom_script_for_target(script, "auto")
}

/// Analyze a script for a named host target such as `wasm-webgpu` or `native-webgpu`.
pub fn analyze_motionloom_script_for_target(
    script: &str,
    target: &str,
) -> MotionLoomAuthoringReport {
    let scanned = scan_tags(script);
    let mut diagnostics = lexical_diagnostics(&scanned);

    let graph = match parse_graph_script(script) {
        Ok(graph) => Some(graph),
        Err(error) => {
            diagnostics.push(parse_error_diagnostic(error.line, &error.message));
            None
        }
    };

    let mut effective_graph = None;
    let mut compile_succeeded = false;
    if let Some(graph) = graph.as_ref() {
        append_animation_diagnostics(graph, &scanned, &mut diagnostics);
        append_semantic_diagnostics(graph, &scanned, &mut diagnostics);
        append_compatibility_diagnostics(script, target, &mut diagnostics);
        match compile_runtime_program(graph.clone()) {
            Ok(_) => compile_succeeded = true,
            Err(error) => diagnostics.push(compile_error_diagnostic(&error.message, &scanned)),
        }
        effective_graph = Some(effective_graph_summary(graph, &scanned));
    }

    sort_and_deduplicate_diagnostics(&mut diagnostics);
    let summary = summarize_diagnostics(&diagnostics);
    let parse_succeeded = graph.is_some();
    let renderable = parse_succeeded && summary.errors == 0;
    let status = if !parse_succeeded {
        AuthoringStatus::Unrenderable
    } else if summary.errors > 0 {
        AuthoringStatus::NeedsRepair
    } else if summary.warnings > 0 {
        AuthoringStatus::NeedsReview
    } else {
        AuthoringStatus::Clean
    };
    let showcase_schema = build_showcase_schema(&scanned, graph.as_ref());

    MotionLoomAuthoringReport {
        report_version: REPORT_VERSION.to_string(),
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        target: target.to_string(),
        status,
        parse_succeeded,
        compile_succeeded,
        renderable,
        summary,
        diagnostics,
        effective_graph,
        showcase_schema,
    }
}

fn compile_error_diagnostic(message: &str, tags: &[ScannedTag]) -> AuthoringDiagnostic {
    let pass_id = message
        .strip_prefix("pass ")
        .and_then(|rest| rest.split_whitespace().next());
    let source = pass_id.and_then(|id| {
        tags.iter()
            .find(|tag| tag.name == "Pass" && attribute_value(tag, "id").as_deref() == Some(id))
    });
    AuthoringDiagnostic {
        severity: AuthoringDiagnosticSeverity::Error,
        code: "PROCESS_COMPILE_ERROR".to_string(),
        phase: "compile".to_string(),
        line: source.map_or(0, |tag| tag.line),
        column: source.map_or(0, |tag| tag.column),
        tag: Some("Pass".to_string()),
        node_id: pass_id.map(str::to_string),
        attribute: None,
        authored_value: None,
        effective_value: None,
        message: message.to_string(),
        effect: "The Process graph parsed but its runtime program could not be compiled."
            .to_string(),
        suggestions: vec![AuthoringSuggestion {
            kind: "repair-process-pass".to_string(),
            message: "Check the named Pass effect, kernel, texture dependencies, and parameter expressions, then analyze the script again.".to_string(),
            replacement: None,
            attribute: None,
            confidence: Some(0.9),
        }],
    }
}

/// Serialize the complete authoring report for Rust hosts, WASM, and LLM tools.
pub fn motionloom_analyze_script_json(script: &str) -> String {
    serde_json::to_string_pretty(&analyze_motionloom_script(script))
        .expect("MotionLoom authoring reports are serializable")
}

/// Serialize a target-specific report without introducing a transport-specific error type.
pub fn motionloom_analyze_script_for_target_json(script: &str, target: &str) -> String {
    serde_json::to_string_pretty(&analyze_motionloom_script_for_target(script, target))
        .expect("MotionLoom authoring reports are serializable")
}

/// Generate the language slice demonstrated by one valid or partially valid showcase.
pub fn motionloom_showcase_schema_json(script: &str) -> String {
    let scanned = scan_tags(script);
    let graph = parse_graph_script(script).ok();
    serde_json::to_string_pretty(&build_showcase_schema(&scanned, graph.as_ref()))
        .expect("MotionLoom showcase schemas are serializable")
}

fn lexical_diagnostics(tags: &[ScannedTag]) -> Vec<AuthoringDiagnostic> {
    let mut diagnostics = Vec::new();
    for tag in tags {
        let Some(capability) = tag_capability(&tag.name) else {
            let suggestion =
                nearest_name(&tag.name, KNOWN_TAGS).map(|(name, confidence)| AuthoringSuggestion {
                    kind: "replace-tag".to_string(),
                    message: format!("Replace <{}> with the recognized <{name}> tag.", tag.name),
                    replacement: Some(name.to_string()),
                    attribute: None,
                    confidence: Some(confidence),
                });
            diagnostics.push(AuthoringDiagnostic {
                severity: AuthoringDiagnosticSeverity::Error,
                code: "UNKNOWN_TAG".to_string(),
                phase: "authoring".to_string(),
                line: tag.line,
                column: tag.column,
                tag: Some(tag.name.clone()),
                node_id: attribute_value(tag, "id"),
                attribute: None,
                authored_value: None,
                effective_value: None,
                message: format!("MotionLoom does not define the <{}> tag.", tag.name),
                effect: "The parser cannot reliably compile this node.".to_string(),
                suggestions: suggestion.into_iter().collect(),
            });
            continue;
        };

        let mut seen = HashMap::<&str, &ScannedAttribute>::new();
        for attribute in &tag.attributes {
            if let Some(first) = seen.insert(attribute.name.as_str(), attribute) {
                diagnostics.push(AuthoringDiagnostic {
                    severity: AuthoringDiagnosticSeverity::Error,
                    code: "DUPLICATE_ATTRIBUTE".to_string(),
                    phase: "authoring".to_string(),
                    line: attribute.line,
                    column: attribute.column,
                    tag: Some(tag.name.clone()),
                    node_id: attribute_value(tag, "id"),
                    attribute: Some(attribute.name.clone()),
                    authored_value: attribute.value.clone(),
                    effective_value: first.value.clone(),
                    message: format!(
                        "<{}> declares attribute {} more than once.",
                        tag.name, attribute.name
                    ),
                    effect: "Only one value can be authoritative; parser behavior is ambiguous."
                        .to_string(),
                    suggestions: vec![AuthoringSuggestion {
                        kind: "remove-duplicate".to_string(),
                        message: format!(
                            "Keep one {} attribute and remove the duplicate.",
                            attribute.name
                        ),
                        replacement: None,
                        attribute: Some(attribute.name.clone()),
                        confidence: Some(1.0),
                    }],
                });
                continue;
            }

            if capability.attributes.contains(&attribute.name.as_str()) {
                continue;
            }
            // Open tags have dynamic payload attributes; globally unknown names are still reported.
            if capability.open_attributes
                && (capability.attributes.is_empty()
                    || is_known_attribute_anywhere(&attribute.name))
            {
                continue;
            }
            let candidates = if capability.attributes.is_empty() {
                ALL_KNOWN_ATTRIBUTES
            } else {
                capability.attributes
            };
            let suggestion = nearest_name(&attribute.name, candidates).map(|(name, confidence)| {
                AuthoringSuggestion {
                    kind: "replace-attribute".to_string(),
                    message: format!(
                        "Replace {} with the supported {} attribute.",
                        attribute.name, name
                    ),
                    replacement: Some(name.to_string()),
                    attribute: Some(attribute.name.clone()),
                    confidence: Some(confidence),
                }
            });
            diagnostics.push(AuthoringDiagnostic {
                severity: AuthoringDiagnosticSeverity::Warning,
                code: "UNKNOWN_ATTRIBUTE".to_string(),
                phase: "authoring".to_string(),
                line: attribute.line,
                column: attribute.column,
                tag: Some(tag.name.clone()),
                node_id: attribute_value(tag, "id"),
                attribute: Some(attribute.name.clone()),
                authored_value: attribute.value.clone(),
                effective_value: None,
                message: format!(
                    "<{}> does not define attribute {}.",
                    tag.name, attribute.name
                ),
                effect: "The current permissive parser ignores this attribute.".to_string(),
                suggestions: suggestion.into_iter().collect(),
            });
        }
    }
    diagnostics
}

fn append_animation_diagnostics(
    graph: &GraphScript,
    tags: &[ScannedTag],
    diagnostics: &mut Vec<AuthoringDiagnostic>,
) {
    for diagnostic in inspect_animation_targets(graph).diagnostics {
        let severity = match diagnostic.severity {
            AnimationDiagnosticSeverity::Error => AuthoringDiagnosticSeverity::Error,
            AnimationDiagnosticSeverity::Warning => AuthoringDiagnosticSeverity::Warning,
        };
        let source = tags.iter().find(|tag| {
            tag.name == "AnimationTarget"
                && attribute_value(tag, "node").as_deref() == Some(diagnostic.node.as_str())
                && attribute_value(tag, "property").as_deref() == Some(diagnostic.property.as_str())
        });
        diagnostics.push(AuthoringDiagnostic {
            severity,
            code: diagnostic.code.to_ascii_uppercase(),
            phase: "animation".to_string(),
            line: source.map_or(0, |tag| tag.line),
            column: source.map_or(0, |tag| tag.column),
            tag: Some("AnimationTarget".to_string()),
            node_id: Some(diagnostic.node),
            attribute: Some(diagnostic.property),
            authored_value: None,
            effective_value: None,
            message: diagnostic.message,
            effect: "The authored animation channel may be ignored or target the wrong node."
                .to_string(),
            suggestions: diagnostic
                .suggestion
                .map(|message| AuthoringSuggestion {
                    kind: "repair-animation-target".to_string(),
                    message,
                    replacement: None,
                    attribute: None,
                    confidence: Some(1.0),
                })
                .into_iter()
                .collect(),
        });
    }
}

fn append_semantic_diagnostics(
    graph: &GraphScript,
    tags: &[ScannedTag],
    diagnostics: &mut Vec<AuthoringDiagnostic>,
) {
    // EnvironmentLight now feeds the existing Scene 3D PBR exposure bridge.
    // HDR image sampling is host/backend-dependent, but intensity is no longer
    // silently ignored and therefore must not be reported as a no-op.

    let surface_ids = tags
        .iter()
        .filter(|tag| tag.name == "Surface")
        .filter_map(|tag| attribute_value(tag, "id"))
        .collect::<BTreeSet<_>>();
    for tag in tags.iter().filter(|tag| tag.name == "Environment") {
        if let Some(collision) = attribute_value(tag, "collision")
            && !matches!(collision.as_str(), "mesh" | "none" | "bounds")
        {
            diagnostics.push(AuthoringDiagnostic {
                severity: AuthoringDiagnosticSeverity::Warning,
                code: "ENVIRONMENT_COLLISION_UNKNOWN".to_string(),
                phase: "environment".to_string(),
                line: tag.line,
                column: tag.column,
                tag: Some("Environment".to_string()),
                node_id: attribute_value(tag, "id"),
                attribute: Some("collision".to_string()),
                authored_value: Some(collision),
                effective_value: Some("none".to_string()),
                message: "Environment collision must be mesh, bounds, or none.".to_string(),
                effect: "Grounding and obstacle queries may not use this Environment.".to_string(),
                suggestions: vec![AuthoringSuggestion {
                    kind: "replace-attribute".to_string(),
                    message: "Use collision=\"mesh\" for authored environment surfaces."
                        .to_string(),
                    replacement: Some("mesh".to_string()),
                    attribute: Some("collision".to_string()),
                    confidence: Some(0.98),
                }],
            });
        }
    }
    for tag in tags.iter().filter(|tag| tag.name == "Model") {
        if let Some(collision) = attribute_value(tag, "collision")
            && !matches!(
                collision.as_str(),
                "kinematic" | "character" | "character_controller" | "none"
            )
        {
            diagnostics.push(AuthoringDiagnostic {
                severity: AuthoringDiagnosticSeverity::Warning,
                code: "MODEL_COLLISION_UNKNOWN".to_string(),
                phase: "environment".to_string(),
                line: tag.line,
                column: tag.column,
                tag: Some("Model".to_string()),
                node_id: attribute_value(tag, "id"),
                attribute: Some("collision".to_string()),
                authored_value: Some(collision),
                effective_value: Some("none".to_string()),
                message: "Model collision must be kinematic or none.".to_string(),
                effect: "The model will render without humanoid environment collision.".to_string(),
                suggestions: vec![AuthoringSuggestion {
                    kind: "replace-attribute".to_string(),
                    message: "Use collision=\"kinematic\" for an auto-fitted humanoid controller."
                        .to_string(),
                    replacement: Some("kinematic".to_string()),
                    attribute: Some("collision".to_string()),
                    confidence: Some(0.98),
                }],
            });
        }
    }
    for tag in tags.iter().filter(|tag| tag.name == "ApplyAction") {
        let Some(ground) = attribute_value(tag, "ground") else {
            continue;
        };
        if !surface_ids.contains(&ground) {
            diagnostics.push(AuthoringDiagnostic {
                severity: AuthoringDiagnosticSeverity::Error,
                code: "ACTION_GROUND_SURFACE_NOT_FOUND".to_string(),
                phase: "environment".to_string(),
                line: tag.line,
                column: tag.column,
                tag: Some("ApplyAction".to_string()),
                node_id: attribute_value(tag, "target"),
                attribute: Some("ground".to_string()),
                authored_value: Some(ground.clone()),
                effective_value: None,
                message: format!("ApplyAction ground references unknown Surface '{ground}'."),
                effect: "The actor cannot be deterministically grounded.".to_string(),
                suggestions: vec![AuthoringSuggestion {
                    kind: "choose-known-surface".to_string(),
                    message: if surface_ids.is_empty() {
                        "Add <Surface id=\"ground\" kind=\"ground\" ... /> inside Environment."
                            .to_string()
                    } else {
                        format!(
                            "Use one of the known surface ids: {}.",
                            surface_ids.iter().cloned().collect::<Vec<_>>().join(", ")
                        )
                    },
                    replacement: surface_ids.iter().next().cloned(),
                    attribute: Some("ground".to_string()),
                    confidence: Some(0.95),
                }],
            });
        }
    }

    // Video and audio declarations are valid assets but do not yet create playable Scene nodes.
    for asset in &graph.assets {
        if matches!(asset.kind, GraphAssetKind::Video | GraphAssetKind::Audio) {
            let kind = match asset.kind {
                GraphAssetKind::Video => "VideoAsset",
                GraphAssetKind::Audio => "AudioAsset",
                _ => unreachable!(),
            };
            diagnostics.push(AuthoringDiagnostic {
                severity: AuthoringDiagnosticSeverity::Warning,
                code: "HOST_ONLY_ASSET".to_string(),
                phase: "renderer-capability".to_string(),
                line: find_tag_line(tags, kind, Some(&asset.id)),
                column: 1,
                tag: Some(kind.to_string()),
                node_id: Some(asset.id.clone()),
                attribute: Some("src".to_string()),
                authored_value: Some(asset.src.clone()),
                effective_value: None,
                message: format!(
                    "{kind} '{}' is registered but has no native Scene playback node.",
                    asset.id
                ),
                effect: "The asset is available to the host but is not rendered by the Scene timeline."
                    .to_string(),
                suggestions: vec![AuthoringSuggestion {
                    kind: "host-integration".to_string(),
                    message: "Use the host media timeline for playback, or replace the asset with currently renderable Scene content.".to_string(),
                    replacement: None,
                    attribute: None,
                    confidence: Some(1.0),
                }],
            });
        }
    }

    append_inline_animation_override_diagnostics(graph, tags, diagnostics);
}

fn append_inline_animation_override_diagnostics(
    graph: &GraphScript,
    tags: &[ScannedTag],
    diagnostics: &mut Vec<AuthoringDiagnostic>,
) {
    for target in &graph.animation_targets {
        let Some(tag) = tags.iter().find(|tag| {
            attribute_value(tag, "id").as_deref() == Some(target.node.as_str())
                && tag
                    .attributes
                    .iter()
                    .any(|attr| attr.name == target.property)
        }) else {
            continue;
        };
        let authored = tag
            .attributes
            .iter()
            .find(|attribute| attribute.name == target.property)
            .and_then(|attribute| attribute.value.clone());
        if !authored.as_deref().is_some_and(is_inline_animation_value) {
            continue;
        }
        diagnostics.push(AuthoringDiagnostic {
            severity: AuthoringDiagnosticSeverity::Warning,
            code: "ANIMATION_OVERRIDE".to_string(),
            phase: "animation".to_string(),
            line: tag.line,
            column: tag.column,
            tag: Some(tag.name.clone()),
            node_id: Some(target.node.clone()),
            attribute: Some(target.property.clone()),
            authored_value: authored,
            effective_value: Some("AnimationTarget".to_string()),
            message: format!(
                "AnimationTarget overrides the inline {} value on node '{}'.",
                target.property, target.node
            ),
            effect: "The inline value remains a fallback, but AnimationTarget is authoritative."
                .to_string(),
            suggestions: vec![AuthoringSuggestion {
                kind: "remove-redundant-animation".to_string(),
                message: "Keep the inline value as an intentional fallback, or remove its curve/expression to avoid two apparent animation sources.".to_string(),
                replacement: None,
                attribute: Some(target.property.clone()),
                confidence: Some(1.0),
            }],
        });
    }
}

fn is_inline_animation_value(value: &str) -> bool {
    ["curve(", "$time", "sin(", "cos(", "random(", "noise("]
        .iter()
        .any(|marker| value.contains(marker))
}

fn append_compatibility_diagnostics(
    script: &str,
    target: &str,
    diagnostics: &mut Vec<AuthoringDiagnostic>,
) {
    let Ok(report) = inspect_gpu_compatibility(script) else {
        return;
    };
    for issue in report.issues {
        let relevant = match target {
            "wasm-webgpu" | "wasm" => format!("{:?}", issue.target).starts_with("Wasm"),
            "native-webgpu" | "native" => {
                matches!(
                    issue.target,
                    crate::compat::GpuCompatibilityTarget::NativeScenePreview
                        | crate::compat::GpuCompatibilityTarget::WgpuTextureOutput
                )
            }
            _ => true,
        };
        if !relevant || issue.severity == GpuCompatibilitySeverity::Info {
            continue;
        }
        let severity = match issue.severity {
            GpuCompatibilitySeverity::Blocking => AuthoringDiagnosticSeverity::Error,
            GpuCompatibilitySeverity::Warning => AuthoringDiagnosticSeverity::Warning,
            GpuCompatibilitySeverity::Info => AuthoringDiagnosticSeverity::Info,
        };
        diagnostics.push(AuthoringDiagnostic {
            severity,
            code: format!("GPU_{}", issue.code.to_ascii_uppercase()),
            phase: "renderer-capability".to_string(),
            line: 0,
            column: 0,
            tag: None,
            node_id: None,
            attribute: None,
            authored_value: None,
            effective_value: Some(format!("{:?}", issue.target)),
            message: issue.message,
            effect: "The selected renderer may fall back to CPU or reject this graph.".to_string(),
            suggestions: vec![AuthoringSuggestion {
                kind: "choose-compatible-renderer".to_string(),
                message: "Use a supported render target or replace the incompatible feature before export.".to_string(),
                replacement: None,
                attribute: None,
                confidence: Some(0.9),
            }],
        });
    }
}

fn parse_error_diagnostic(line: usize, message: &str) -> AuthoringDiagnostic {
    let (code, suggestion) = classify_parse_error(message);
    AuthoringDiagnostic {
        severity: AuthoringDiagnosticSeverity::Error,
        code: code.to_string(),
        phase: "parse".to_string(),
        line,
        column: 1,
        tag: extract_angle_tag(message),
        node_id: None,
        attribute: extract_mentioned_attribute(message),
        authored_value: None,
        effective_value: None,
        message: message.to_string(),
        effect: "The graph did not compile and cannot be rendered.".to_string(),
        suggestions: vec![AuthoringSuggestion {
            kind: "repair-parse-error".to_string(),
            message: suggestion.to_string(),
            replacement: None,
            attribute: None,
            confidence: Some(0.9),
        }],
    }
}

fn classify_parse_error(message: &str) -> (&'static str, &'static str) {
    let lower = message.to_ascii_lowercase();
    if lower.contains("missing required attribute") || lower.contains("requires ") {
        return (
            "MISSING_REQUIRED_ATTRIBUTE",
            "Add the required attribute using the canonical spelling shown in the error message.",
        );
    }
    if lower.contains("duplicate") {
        return (
            "DUPLICATE_DEFINITION",
            "Keep one authoritative declaration and rename or remove the duplicate.",
        );
    }
    if lower.contains("not found") || lower.contains("missing target") {
        return (
            "UNRESOLVED_REFERENCE",
            "Define the referenced id before use or update the reference to an existing id.",
        );
    }
    if lower.contains("invalid") && lower.contains("curve") {
        return (
            "INVALID_CURVE",
            "Use numeric curve points in time:value[:ease] form; move procedural functions to a standalone expression.",
        );
    }
    if lower.contains("expected") || lower.contains("invalid") {
        return (
            "INVALID_VALUE",
            "Replace the value with one of the expected forms or enum values named in the error.",
        );
    }
    if lower.contains("close") || lower.contains("closed") || lower.contains("unclosed") {
        return (
            "UNCLOSED_TAG",
            "Close the named tag and keep <Present /> as the final direct child of <Graph>.",
        );
    }
    (
        "PARSE_ERROR",
        "Repair the syntax at the reported line, then run the authoring analyzer again.",
    )
}

fn effective_graph_summary(graph: &GraphScript, tags: &[ScannedTag]) -> EffectiveGraphSummary {
    EffectiveGraphSummary {
        fps: graph.fps,
        duration_ms: graph.duration_ms,
        size: [graph.size.0, graph.size.1],
        render_size: graph.render_size.map(|size| [size.0, size.1]),
        scenes: graph.scenes.len(),
        scene_nodes: tags
            .iter()
            .filter(|tag| is_visual_scene_tag(&tag.name))
            .count(),
        assets: graph.assets.len(),
        models: tags
            .iter()
            .filter(|tag| matches!(tag.name.as_str(), "Model" | "Environment"))
            .count(),
        cameras_3d: tags.iter().filter(|tag| tag.name == "Camera3D").count(),
        environment_lights: tags
            .iter()
            .filter(|tag| tag.name == "EnvironmentLight")
            .count(),
        active_environment_lights: tags
            .iter()
            .filter(|tag| tag.name == "EnvironmentLight")
            .count(),
        animation_targets: graph.animation_targets.len(),
        process_passes: graph.passes.len(),
        present_from: graph.present.from.clone(),
    }
}

fn build_showcase_schema(
    tags: &[ScannedTag],
    graph: Option<&GraphScript>,
) -> MotionLoomShowcaseSchema {
    let mut grouped = BTreeMap::<String, Vec<&ScannedTag>>::new();
    for tag in tags {
        grouped.entry(tag.name.clone()).or_default().push(tag);
    }
    let mut tag_schemas = Vec::new();
    for (name, occurrences) in grouped {
        let capability = tag_capability(&name);
        let mut attributes = BTreeMap::<String, Vec<&ScannedAttribute>>::new();
        for tag in &occurrences {
            for attribute in &tag.attributes {
                attributes
                    .entry(attribute.name.clone())
                    .or_default()
                    .push(attribute);
            }
        }
        let attributes = attributes
            .into_iter()
            .map(|(attribute_name, values)| {
                let recognized = capability.is_some_and(|capability| {
                    capability.attributes.contains(&attribute_name.as_str())
                        || (capability.open_attributes
                            && (capability.attributes.is_empty()
                                || is_known_attribute_anywhere(&attribute_name)))
                });
                let descriptor = crate::scene::animation::animation_property_descriptor(
                    canonical_attribute_name(&attribute_name),
                );
                let mut examples = BTreeSet::new();
                for attribute in &values {
                    if let Some(value) = &attribute.value {
                        examples.insert(shorten_example(value));
                    }
                }
                ShowcaseAttributeSchema {
                    name: attribute_name.clone(),
                    occurrences: values.len(),
                    examples: examples.into_iter().take(4).collect(),
                    recognized,
                    canonical_name: Some(canonical_attribute_name(&attribute_name).to_string())
                        .filter(|canonical| canonical != &attribute_name),
                    value_type: descriptor.map(|descriptor| {
                        format!("{:?}", descriptor.value_type).to_ascii_lowercase()
                    }),
                    supports_inline_expression: supports_inline_expression(&name, &attribute_name),
                    supports_animation_target: Some(descriptor.is_some()),
                }
            })
            .collect();
        tag_schemas.push(ShowcaseTagSchema {
            tag: name,
            occurrences: occurrences.len(),
            recognized: capability.is_some(),
            validation_coverage: match capability {
                Some(capability) if capability.open_attributes => "open".to_string(),
                Some(_) => "strict".to_string(),
                None => "unknown".to_string(),
            },
            attributes,
        });
    }
    MotionLoomShowcaseSchema {
        schema_version: SCHEMA_VERSION.to_string(),
        engine_version: env!("CARGO_PKG_VERSION").to_string(),
        root_tag: tags.first().map(|tag| tag.name.clone()),
        tags: tag_schemas,
        animation_properties: graph
            .map(|graph| {
                graph
                    .animation_targets
                    .iter()
                    .map(|target| target.property.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default(),
        asset_kinds: graph
            .map(|graph| {
                graph
                    .assets
                    .iter()
                    .map(|asset| format!("{:?}", asset.kind).to_ascii_lowercase())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn supports_inline_expression(tag: &str, attribute: &str) -> Option<bool> {
    if matches!(
        tag,
        "ParticleEmitter"
            | "SpringChain"
            | "DistanceConstraint"
            | "Hinge"
            | "RigidBody2D"
            | "Cloth"
            | "HairStrandField"
    ) && matches!(
        attribute,
        "x" | "y" | "rate" | "radius" | "stiffness" | "damping" | "amount"
    ) {
        return Some(false);
    }
    crate::scene::animation::animation_property_descriptor(canonical_attribute_name(attribute))
        .map(|_| true)
}

fn canonical_attribute_name(attribute: &str) -> &str {
    match attribute {
        "scale_x" => "scaleX",
        "scale_y" => "scaleY",
        "skew_x" => "skewX",
        "skew_y" => "skewY",
        "rotation_x" => "rotationX",
        "rotation_y" => "rotationY",
        "rotation_z" => "rotationZ",
        "position_x" => "positionX",
        "position_y" => "positionY",
        "position_z" => "positionZ",
        "font_size" => "fontSize",
        "line_height" => "lineHeight",
        "stroke_width" => "strokeWidth",
        "transform_origin_x" => "transformOriginX",
        "transform_origin_y" => "transformOriginY",
        other => other,
    }
}

fn summarize_diagnostics(diagnostics: &[AuthoringDiagnostic]) -> AuthoringSummary {
    let mut summary = AuthoringSummary::default();
    for diagnostic in diagnostics {
        match diagnostic.severity {
            AuthoringDiagnosticSeverity::Error => summary.errors += 1,
            AuthoringDiagnosticSeverity::Warning => summary.warnings += 1,
            AuthoringDiagnosticSeverity::Info => summary.info += 1,
        }
        match diagnostic.code.as_str() {
            "UNKNOWN_ATTRIBUTE" => summary.ignored_attributes += 1,
            "UNKNOWN_TAG" => summary.unknown_tags += 1,
            "RENDERER_NO_OP" => summary.no_op_nodes += 1,
            "ANIMATION_OVERRIDE" | "DUPLICATE_CHANNEL" => summary.animation_conflicts += 1,
            "ASSET_FETCH_FAILED" | "MISSING_ASSET" => summary.missing_assets += 1,
            _ => {}
        }
    }
    summary
}

fn sort_and_deduplicate_diagnostics(diagnostics: &mut Vec<AuthoringDiagnostic>) {
    diagnostics.sort_by(|a, b| {
        (a.line, a.column, &a.code, &a.attribute).cmp(&(b.line, b.column, &b.code, &b.attribute))
    });
    diagnostics.dedup_by(|a, b| {
        a.line == b.line
            && a.column == b.column
            && a.code == b.code
            && a.attribute == b.attribute
            && a.node_id == b.node_id
    });
}

fn scan_tags(script: &str) -> Vec<ScannedTag> {
    let bytes = script.as_bytes();
    let mut tags = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let Some(relative) = script[cursor..].find('<') else {
            break;
        };
        let start = cursor + relative;
        if script[start..].starts_with("<!--") {
            cursor = script[start + 4..]
                .find("-->")
                .map(|end| start + 4 + end + 3)
                .unwrap_or(bytes.len());
            continue;
        }
        if matches!(bytes.get(start + 1), Some(b'/') | Some(b'!') | Some(b'?')) {
            cursor = script[start..]
                .find('>')
                .map(|end| start + end + 1)
                .unwrap_or(bytes.len());
            continue;
        }
        let Some(end) = find_tag_end(script, start + 1) else {
            break;
        };
        let body = &script[start + 1..end];
        let name_end = body
            .char_indices()
            .find(|(_, ch)| ch.is_whitespace() || matches!(ch, '/' | '>'))
            .map(|(index, _)| index)
            .unwrap_or(body.len());
        let name = body[..name_end].trim();
        if name.is_empty() {
            cursor = end + 1;
            continue;
        }
        let (line, column) = line_column(script, start);
        tags.push(ScannedTag {
            name: name.to_string(),
            attributes: scan_attributes(script, body, start + 1, name_end),
            line,
            column,
        });
        cursor = end + 1;
    }
    tags
}

fn find_tag_end(script: &str, start: usize) -> Option<usize> {
    let mut quote = None;
    let mut brace_depth = 0usize;
    for (relative, ch) in script[start..].char_indices() {
        if let Some(active) = quote {
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '>' if brace_depth == 0 => return Some(start + relative),
            _ => {}
        }
    }
    None
}

fn scan_attributes(
    script: &str,
    body: &str,
    body_offset: usize,
    mut cursor: usize,
) -> Vec<ScannedAttribute> {
    let bytes = body.as_bytes();
    let mut attributes = Vec::new();
    while cursor < bytes.len() {
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if cursor >= bytes.len() || bytes[cursor] == b'/' {
            break;
        }
        let name_start = cursor;
        while cursor < bytes.len()
            && (bytes[cursor].is_ascii_alphanumeric()
                || matches!(bytes[cursor], b'_' | b'-' | b':' | b'.'))
        {
            cursor += 1;
        }
        if cursor == name_start {
            cursor += 1;
            continue;
        }
        let name = body[name_start..cursor].to_string();
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        let mut value = None;
        if cursor < bytes.len() && bytes[cursor] == b'=' {
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let value_start = cursor;
            cursor = consume_attribute_value(body, cursor);
            value = Some(body[value_start..cursor].trim().to_string());
        }
        let (line, column) = line_column(script, body_offset + name_start);
        attributes.push(ScannedAttribute {
            name,
            value,
            line,
            column,
        });
    }
    attributes
}

fn consume_attribute_value(body: &str, start: usize) -> usize {
    let bytes = body.as_bytes();
    if start >= bytes.len() {
        return start;
    }
    if matches!(bytes[start], b'"' | b'\'') {
        let quote = bytes[start];
        let mut cursor = start + 1;
        while cursor < bytes.len() {
            if bytes[cursor] == quote && bytes.get(cursor.wrapping_sub(1)) != Some(&b'\\') {
                return cursor + 1;
            }
            cursor += 1;
        }
        return bytes.len();
    }
    if bytes[start] == b'{' {
        let mut cursor = start;
        let mut depth = 0usize;
        let mut quote = None;
        while cursor < bytes.len() {
            let ch = bytes[cursor] as char;
            if let Some(active) = quote {
                if ch == active && bytes.get(cursor.wrapping_sub(1)) != Some(&b'\\') {
                    quote = None;
                }
            } else {
                match ch {
                    '"' | '\'' => quote = Some(ch),
                    '{' => depth += 1,
                    '}' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            return cursor + 1;
                        }
                    }
                    _ => {}
                }
            }
            cursor += 1;
        }
        return bytes.len();
    }
    let mut cursor = start;
    while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() && bytes[cursor] != b'/' {
        cursor += 1;
    }
    cursor
}

fn line_column(script: &str, offset: usize) -> (usize, usize) {
    let prefix = &script[..offset.min(script.len())];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = prefix
        .rfind('\n')
        .map(|index| prefix[index + 1..].chars().count() + 1)
        .unwrap_or_else(|| prefix.chars().count() + 1);
    (line, column)
}

fn attribute_value(tag: &ScannedTag, name: &str) -> Option<String> {
    tag.attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_deref())
        .map(strip_authored_wrapper)
}

fn strip_authored_wrapper(value: &str) -> String {
    value
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn nearest_name<'a>(name: &str, candidates: &'a [&str]) -> Option<(&'a str, f32)> {
    let normalized = name.to_ascii_lowercase();
    let mut best = None;
    for candidate in candidates {
        let candidate_normalized = candidate.to_ascii_lowercase();
        let distance = levenshtein(&normalized, &candidate_normalized);
        let longest = normalized.len().max(candidate.len()).max(1);
        let similarity = 1.0 - distance as f32 / longest as f32;
        let confidence = if candidate_normalized.len() >= 3
            && normalized.ends_with(&candidate_normalized)
        {
            similarity.max(0.92)
        } else if candidate_normalized.len() >= 3 && normalized.starts_with(&candidate_normalized) {
            similarity.max(0.85)
        } else {
            similarity
        };
        if confidence >= 0.45
            && best
                .as_ref()
                .is_none_or(|(_, best_confidence)| confidence > *best_confidence)
        {
            best = Some((*candidate, confidence));
        }
    }
    best
}

fn levenshtein(a: &str, b: &str) -> usize {
    let mut costs = (0..=b.chars().count()).collect::<Vec<_>>();
    for (i, ca) in a.chars().enumerate() {
        let mut previous = i;
        costs[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let current = costs[j + 1];
            costs[j + 1] = if ca == cb {
                previous
            } else {
                1 + previous.min(current).min(costs[j])
            };
            previous = current;
        }
    }
    *costs.last().unwrap_or(&0)
}

fn extract_angle_tag(message: &str) -> Option<String> {
    let start = message.find('<')? + 1;
    let end = message[start..].find(|ch: char| ch == '>' || ch.is_whitespace())? + start;
    Some(message[start..end].trim_matches('/').to_string())
}

fn extract_mentioned_attribute(message: &str) -> Option<String> {
    let marker = "attribute: ";
    let start = message.find(marker)? + marker.len();
    Some(
        message[start..]
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_matches(['\'', '"', '.', ','])
            .to_string(),
    )
}

fn find_tag_line(tags: &[ScannedTag], name: &str, id: Option<&str>) -> usize {
    tags.iter()
        .find(|tag| {
            tag.name == name
                && id.is_none_or(|id| attribute_value(tag, "id").as_deref() == Some(id))
        })
        .map(|tag| tag.line)
        .unwrap_or(0)
}

fn shorten_example(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.chars().count() <= 120 {
        return trimmed.to_string();
    }
    format!("{}…", trimmed.chars().take(119).collect::<String>())
}

fn is_visual_scene_tag(tag: &str) -> bool {
    matches!(
        tag,
        "Text"
            | "Image"
            | "Svg"
            | "SVG"
            | "Rect"
            | "Circle"
            | "Ellipse"
            | "Line"
            | "Polyline"
            | "Path"
            | "Group"
            | "Model"
            | "ParticleEmitter"
    )
}

fn tag_capability(tag: &str) -> Option<TagCapability> {
    let strict = |attributes| TagCapability {
        attributes,
        open_attributes: false,
    };
    let open = |attributes| TagCapability {
        attributes,
        open_attributes: true,
    };
    Some(match tag {
        "Graph" => strict(&[
            "id",
            "version",
            "fps",
            "apply",
            "scope",
            "duration",
            "size",
            "renderSize",
            "render_size",
        ]),
        "Assets" | "Defs" | "Timeline" | "Effects" | "PostEffects" | "Variants" => strict(&["id"]),
        "VideoAsset" | "ImageAsset" | "ModelAsset" | "AudioAsset" => {
            strict(&["id", "src", "decoder", "colorSpace", "color_space"])
        }
        "AnimationAsset" => strict(&["id", "src"]),
        "Background" => strict(&["id", "color"]),
        "Scene" => strict(&["id", "size"]),
        "Track" => strict(&[
            "id",
            "space",
            "z",
            "zDepth",
            "z_depth",
            "role",
            "compositeOrder",
            "composite_order",
        ]),
        "Sequence" => strict(&["id", "from", "duration", "out"]),
        "Layer" | "Layer3D" => open(LAYER_ATTRIBUTES),
        "Group" => open(GROUP_ATTRIBUTES),
        "CompositeGroup" => strict(&[
            "id",
            "space",
            "compositeOrder",
            "composite_order",
            "depth",
            "format",
        ]),
        "Camera3D" => strict(&[
            "id",
            "position",
            "target",
            "fov",
            "up",
            "roll",
            "horizonLock",
            "horizon_lock",
        ]),
        "Anchor" => strict(&[
            "id",
            "relativeTo",
            "relative_to",
            "offset",
            "space",
            "node",
            "surface",
            "uv",
        ]),
        "EnvironmentLight" => strict(&["id", "asset", "intensity"]),
        "EnvironmentDebug" => strict(&[
            "axes",
            "bounds",
            "surfaces",
            "anchors",
            "actionPath",
            "cameras",
        ]),
        "Model" | "ModelLayer" | "Environment" => strict(&[
            "id",
            "asset",
            "profile",
            "rig",
            "retarget",
            "position",
            "positionX",
            "positionY",
            "positionZ",
            "position_x",
            "position_y",
            "position_z",
            "rotation",
            "rotationX",
            "rotationY",
            "rotationZ",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "scale",
            "exposure",
            "static",
            "collision",
            "castShadow",
            "receiveShadow",
            "up",
            "forward",
            "unitScale",
            "unit_scale",
            "scaleMode",
            "scale_mode",
        ]),
        "Surface" => strict(&[
            "id",
            "node",
            "kind",
            "height",
            "normal",
            "space",
            "centroid",
            "boundsMin",
            "boundsMax",
            "bounds_min",
            "bounds_max",
        ]),
        "MaterialBinding" => strict(&["material", "definition", "texture"]),
        "Play" => strict(&[
            "clip",
            "loop",
            "speed",
            "weight",
            "blendIn",
            "blendOut",
            "blend_in",
            "blend_out",
            "mask",
        ]),
        "BoneOverride" => strict(&[
            "bone",
            "x",
            "y",
            "z",
            "rotationX",
            "rotationY",
            "rotationZ",
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "scale",
        ]),
        "Rect" => strict(RECT_ATTRIBUTES),
        "Circle" => strict(CIRCLE_ATTRIBUTES),
        "Ellipse" => strict(ELLIPSE_ATTRIBUTES),
        "Line" => open(LINE_ATTRIBUTES),
        "Polyline" | "Curve" => open(POLYLINE_ATTRIBUTES),
        "Path" | "FaceJaw" => open(PATH_ATTRIBUTES),
        "Text" => strict(TEXT_ATTRIBUTES),
        "TextLayout" => open(TEXT_LAYOUT_ATTRIBUTES),
        "TextAnimator" => open(TEXT_ANIMATOR_ATTRIBUTES),
        "Transform" => strict(TRANSFORM_ATTRIBUTES),
        "Style" => open(STYLE_ATTRIBUTES),
        "Glow" => strict(&["radius", "intensity", "color"]),
        "Shadow" => strict(&["id", "x", "y", "blur", "color", "opacity"]),
        "Repeat" => open(REPEAT_ATTRIBUTES),
        "Layout" => open(LAYOUT_ATTRIBUTES),
        "AnimationTarget" => strict(&["node", "property"]),
        "Key" => strict(&["time", "frame", "value", "ease"]),
        "ParticleEmitter" => strict(&[
            "id", "target", "x", "y", "rate", "lifetime", "velocity", "gravity", "radius", "color",
        ]),
        "SpringChain" => strict(&[
            "id",
            "target",
            "pin",
            "segments",
            "stiffness",
            "damping",
            "gravity",
            "wind",
            "attraction",
            "colliders",
            "collisionRadius",
        ]),
        "DynamicCurve" => strict(&["id", "target", "simulation"]),
        "DistanceConstraint" => strict(&["id", "a", "b", "distance", "stiffness"]),
        "Hinge" => strict(&["id", "a", "b", "anchor", "stiffness"]),
        "RigidBody2D" => strict(&["id", "target", "mass", "velocity", "angularVelocity"]),
        "Cloth" => strict(&[
            "id",
            "target",
            "columns",
            "rows",
            "stiffness",
            "damping",
            "amplitude",
            "frequency",
        ]),
        "HairStrandField" => strict(&[
            "id",
            "target",
            "strands",
            "segments",
            "stiffness",
            "damping",
        ]),
        "CacheBake" => strict(&["id", "target", "fromFrame", "toFrame"]),
        "Gravity" => strict(&["id", "vector"]),
        "Wind" => strict(&["id", "direction", "strength", "turbulence", "noiseScale"]),
        "Attraction" => strict(&["id", "target", "point", "strength", "radius"]),
        "Collider" => strict(&[
            "id", "target", "shape", "x", "y", "radius", "radiusX", "radiusY", "from", "to",
        ]),
        "Process" => strict(&["id", "output"]),
        "Input" => open(&["id", "type", "from", "fmt", "size"]),
        "Tex" => strict(&[
            "id",
            "fmt",
            "from",
            "src",
            "input",
            "size",
            "usage",
            "transient",
            "pingpong",
        ]),
        "Buffer" => strict(&[
            "id",
            "elemType",
            "elem_type",
            "length",
            "stride",
            "usage",
            "transient",
            "pingpong",
        ]),
        "Pass" => open(&[
            "id",
            "kind",
            "role",
            "kernel",
            "mode",
            "effect",
            "transition",
            "transitionFallback",
            "transitionEasing",
            "transitionClips",
            "transition_fallback",
            "transition_easing",
            "transition_clips",
            "in",
            "out",
            "params",
            "mask",
            "maskMode",
            "maskInvert",
            "mask_mode",
            "mask_invert",
            "iterate",
            "pingpong",
            "cache",
            "blend",
            "loadOp",
            "storeOp",
            "load_op",
            "store_op",
        ]),
        "Output" => strict(&[
            "id",
            "from",
            "to",
            "fmt",
            "size",
            "colorSpace",
            "color_space",
            "alpha",
        ]),
        "Present" => strict(&["from", "to", "format", "colorSpace", "color_space", "alpha"]),
        "LinearGradient" => strict(&["id", "x1", "y1", "x2", "y2", "stops", "units"]),
        "RadialGradient" => strict(&["id", "cx", "cy", "r", "fx", "fy", "stops", "units"]),
        "Filter" => strict(&["id"]),
        "Blur" => strict(&["radius"]),
        "ColorMatrix" => strict(&["values"]),
        "Material" => open(&[
            "id",
            "textureAmount",
            "specular",
            "roughness",
            "displacementAmount",
            "displacement",
            "refraction",
            "glass",
            "dispersion",
        ]),
        "Texture" => open(&["id", "src"]),
        "Noise" => open(&[
            "id",
            "kind",
            "seed",
            "scale",
            "strength",
            "octaves",
            "evolution",
            "contrast",
        ]),
        "Font" => strict(&["id", "family", "path", "fallback"]),
        "Palette" => strict(&["id"]),
        "Color" => strict(&["name", "value"]),
        "Effect" => strict(&["process"]),
        "Component" => strict(&["id"]),
        "Param" => strict(&["name", "type", "default", "values"]),
        "Derived" => strict(&["name", "value"]),
        "Slot" => strict(&["name"]),
        "Fill" => strict(&["slot"]),
        "Use" => open(&["id", "ref"]),
        "Vary" => open(&["property", "values", "range", "choose"]),
        "Action" => open(&[
            "id",
            "source",
            "clip",
            "sourceProfile",
            "source_profile",
            "profile",
            "skeleton",
            "intent",
            "duration",
        ]),
        "Pose" => strict(&["t", "label"]),
        "Marker" => strict(&["id", "time", "t", "role"]),
        "Bone" => open(&[
            "id",
            "parent",
            "x",
            "y",
            "z",
            "rotation",
            "rotationX",
            "rotationY",
            "rotationZ",
            "scale",
            "length",
            "role",
            "side",
            "forward",
            "twist",
            "bend",
            "turn",
            "opacity",
        ]),
        "IK" => open(&[
            "root",
            "mid",
            "end",
            "chain",
            "targetX",
            "targetY",
            "targetZ",
            "target_x",
            "target_y",
            "target_z",
            "pole",
            "poleX",
            "poleY",
            "poleZ",
            "bend",
            "weight",
            "iterations",
            "plane",
        ]),
        "ApplyAction" => strict(&[
            "target",
            "action",
            "at",
            "loop",
            "weight",
            "speed",
            "blendIn",
            "blendOut",
            "blend_in",
            "blend_out",
            "mode",
            "mask",
            "duration",
            "rootMotion",
            "root_motion",
            "destination",
            "takeoff",
            "contact",
            "landing",
            "face",
            "syncGroup",
            "sync_group",
            "syncMarker",
            "sync_marker",
            "ground",
            "groundOffset",
            "ground_offset",
            "footLock",
            "foot_lock",
        ]),
        "ModelProfile" => strict(&["id", "kind", "model", "preset"]),
        "Retarget" => strict(&["preset"]),
        "Map" => strict(&["from", "to"]),
        "BoneAxisMap" => strict(&[]),
        "Axis" => open(&[
            "bone",
            "id",
            "forward",
            "side",
            "twist",
            "bend",
            "turn",
            "restForward",
            "restSide",
            "restTwist",
            "restBend",
            "restTurn",
            "rest_forward",
            "rest_side",
            "rest_twist",
            "rest_bend",
            "rest_turn",
        ]),
        "Skeleton" => strict(&[
            "id",
            "profile",
            "height",
            "facing",
            "symmetryAxis",
            "symmetry_axis",
            "validation",
            "autoCorrect",
            "auto_correct",
            "overlay",
        ]),
        "Landmark" => strict(&["id", "bone", "offset"]),
        "Measure" => strict(&["id", "from", "to"]),
        "Ratio" => strict(&["measure", "relativeTo", "relative_to", "value", "tolerance"]),
        "Region" => open(&[
            "id", "role", "type", "center", "from", "to", "radiusX", "radiusY", "radius_x",
            "radius_y", "width",
        ]),
        "Constraint" => strict(&[
            "type",
            "kind",
            "source",
            "target",
            "at",
            "duration",
            "solver",
            "weight",
            "left",
            "right",
            "axis",
            "from",
            "to",
            "bone",
            "relativeTo",
            "relative_to",
            "value",
            "min",
            "max",
        ]),
        "Guide" => strict(&["id", "type", "through", "angle"]),
        "Control" => strict(&[
            "id",
            "type",
            "target",
            "targets",
            "chainLength",
            "chain_length",
        ]),
        name if KNOWN_TAGS.contains(&name) => open(&[]),
        _ => return None,
    })
}

fn is_known_attribute_anywhere(attribute: &str) -> bool {
    ALL_KNOWN_ATTRIBUTES.contains(&attribute)
        || COMMON_TRANSFORM_ATTRIBUTES.contains(&attribute)
        || PAINT_ATTRIBUTES.contains(&attribute)
}

const COMMON_TRANSFORM_ATTRIBUTES: &[&str] = &[
    "id",
    "x",
    "y",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
    "opacity",
    "blend",
];
const PAINT_ATTRIBUTES: &[&str] = &[
    "color",
    "fill",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "texture",
    "textureOpacity",
    "textureScale",
    "textureMask",
    "texture_opacity",
    "texture_scale",
    "texture_mask",
    "strokeDasharray",
    "trimStart",
    "trimEnd",
];
const RECT_ATTRIBUTES: &[&str] = &[
    "id",
    "x",
    "y",
    "width",
    "height",
    "radius",
    "color",
    "fill",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "opacity",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
    "blend",
    "texture",
    "textureOpacity",
    "textureScale",
    "textureMask",
    "texture_opacity",
    "texture_scale",
    "texture_mask",
    "strokeDasharray",
    "trimStart",
    "trimEnd",
];
const CIRCLE_ATTRIBUTES: &[&str] = &[
    "id",
    "x",
    "y",
    "radius",
    "color",
    "fill",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "opacity",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
    "blend",
    "texture",
    "textureOpacity",
    "textureScale",
    "textureMask",
    "texture_opacity",
    "texture_scale",
    "texture_mask",
    "strokeDasharray",
    "trimStart",
    "trimEnd",
];
const ELLIPSE_ATTRIBUTES: &[&str] = &[
    "id",
    "x",
    "y",
    "cx",
    "cy",
    "radiusX",
    "radiusY",
    "radius_x",
    "radius_y",
    "rx",
    "ry",
    "color",
    "fill",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "opacity",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
    "blend",
];
const LINE_ATTRIBUTES: &[&str] = &[
    "id",
    "x1",
    "y1",
    "x2",
    "y2",
    "stroke",
    "color",
    "strokeWidth",
    "stroke_width",
    "opacity",
    "lineCap",
    "lineJoin",
    "line_cap",
    "line_join",
    "trimStart",
    "trimEnd",
    "trim_start",
    "trim_end",
    "taperStart",
    "taperEnd",
    "taper_start",
    "taper_end",
];
const POLYLINE_ATTRIBUTES: &[&str] = &[
    "id",
    "points",
    "closed",
    "fill",
    "color",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "opacity",
    "trimStart",
    "trimEnd",
    "trim_start",
    "trim_end",
    "taperStart",
    "taperEnd",
];
const PATH_ATTRIBUTES: &[&str] = &[
    "id",
    "d",
    "fill",
    "color",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "opacity",
    "fillRule",
    "fill_rule",
    "booleanOp",
    "boolean_op",
    "trimStart",
    "trimEnd",
    "trim_start",
    "trim_end",
    "taperStart",
    "taperEnd",
    "taper_start",
    "taper_end",
    "offsetPath",
    "offset_path",
    "lineCap",
    "lineJoin",
    "line_cap",
    "line_join",
    "blend",
    "strokeWidthStart",
    "strokeWidthEnd",
    "strokePressure",
    "strokeDasharray",
    "normalize",
    "roundCorners",
    "brush",
    "rotation",
    "scale",
    "x",
    "y",
];
const TEXT_ATTRIBUTES: &[&str] = &[
    "id",
    "value",
    "x",
    "y",
    "width",
    "maxWidth",
    "max_width",
    "align",
    "tracking",
    "textGap",
    "text_gap",
    "fontSize",
    "font_size",
    "size",
    "renderScale",
    "render_scale",
    "antialias",
    "antiAlias",
    "aa",
    "edgeSmoothing",
    "edge_smoothing",
    "softEdge",
    "soft_edge",
    "blur",
    "lineHeight",
    "line_height",
    "color",
    "fill",
    "blend",
    "opacity",
    "box",
    "boxColor",
    "box_color",
    "boxPadding",
    "box_padding",
    "boxPaddingX",
    "boxPaddingY",
    "box_padding_x",
    "box_padding_y",
    "boxRadius",
    "box_radius",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "strokeJoin",
    "stroke_join",
    "strokePosition",
    "stroke_position",
    "fontFamily",
    "font_family",
    "fontWeight",
    "font_weight",
    "font",
    "fontPath",
    "font_path",
    "visibleChars",
    "visible_chars",
    "maxLines",
    "max_lines",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
];
const GROUP_ATTRIBUTES: &[&str] = &[
    "id",
    "brush",
    "material",
    "x",
    "y",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
    "opacity",
    "blend",
    "filter",
    "effects",
    "mask",
    "maskMode",
    "maskInvert",
    "mask_mode",
    "mask_invert",
    "deformAmount",
    "deform_amount",
    "maskFrom",
    "maskFeather",
    "maskExpansion",
    "deformGrid",
    "gridFrom",
    "gridTo",
];
const LAYER_ATTRIBUTES: &[&str] = &[
    "id",
    "space",
    "x",
    "y",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "scale_x",
    "scale_y",
    "skewX",
    "skewY",
    "skew_x",
    "skew_y",
    "transformOriginX",
    "transformOriginY",
    "transform_origin_x",
    "transform_origin_y",
    "opacity",
    "blend",
    "z",
    "zDepth",
    "z_depth",
    "perspective",
    "rotationX",
    "rotationY",
    "rotation_x",
    "rotation_y",
    "translateZ",
    "translate_z",
    "source",
    "ref",
    "playbackRate",
    "playback_rate",
    "sourceTime",
    "source_time",
    "timeOffset",
    "time_offset",
    "out",
    "mask",
    "maskMode",
    "maskInvert",
    "maskFrom",
    "mask_mode",
    "mask_invert",
    "matte",
    "matteFrom",
    "matte_from",
    "matteMode",
    "matte_mode",
    "matteInvert",
    "matte_invert",
];
const TEXT_LAYOUT_ATTRIBUTES: &[&str] = &[
    "width",
    "maxWidth",
    "max_width",
    "maxLines",
    "max_lines",
    "wrap",
    "overflow",
    "lineHeight",
    "line_height",
    "align",
];
const TEXT_ANIMATOR_ATTRIBUTES: &[&str] = &[
    "id",
    "selector",
    "range",
    "from",
    "duration",
    "stagger",
    "order",
    "mode",
    "activeWord",
    "active_word",
    "preRoll",
    "pre_roll",
    "postRoll",
    "post_roll",
    "randomSeed",
    "random_seed",
];
const TRANSFORM_ATTRIBUTES: &[&str] = &[
    "x", "y", "rotation", "scale", "scaleX", "scaleY", "scale_x", "scale_y", "skewX", "skewY",
    "skew_x", "skew_y",
];
const STYLE_ATTRIBUTES: &[&str] = &[
    "color",
    "opacity",
    "blur",
    "stroke",
    "strokeWidth",
    "stroke_width",
    "strokeJoin",
    "strokePosition",
    "shadowColor",
    "shadowX",
    "shadowY",
    "shadowBlur",
    "shadow_color",
    "shadow_x",
    "shadow_y",
    "shadow_blur",
];
const REPEAT_ATTRIBUTES: &[&str] = &[
    "id",
    "count",
    "x",
    "y",
    "xStep",
    "yStep",
    "x_step",
    "y_step",
    "rotation",
    "rotationStep",
    "rotation_step",
    "scale",
    "scaleStep",
    "scale_step",
    "opacity",
    "opacityStep",
    "opacity_step",
    "seed",
    "distribution",
    "bounds",
    "scaleRange",
    "rotationRange",
    "opacityRange",
];
const LAYOUT_ATTRIBUTES: &[&str] = &[
    "id",
    "mode",
    "width",
    "height",
    "padding",
    "gap",
    "align",
    "justify",
    "columns",
    "rows",
    "x",
    "y",
    "itemWidth",
    "itemHeight",
];

const KNOWN_TAGS: &[&str] = &[
    "Graph",
    "Assets",
    "VideoAsset",
    "ImageAsset",
    "ModelAsset",
    "AudioAsset",
    "AnimationAsset",
    "Background",
    "Scene",
    "Defs",
    "Timeline",
    "Track",
    "Sequence",
    "Layer",
    "Layer3D",
    "Group",
    "CompositeGroup",
    "Camera3D",
    "Anchor",
    "EnvironmentLight",
    "Environment",
    "EnvironmentDebug",
    "Surface",
    "Model",
    "ModelLayer",
    "MaterialBinding",
    "Play",
    "BoneOverride",
    "Rect",
    "Circle",
    "Ellipse",
    "Line",
    "Polyline",
    "Curve",
    "Path",
    "FaceJaw",
    "Text",
    "TextLayout",
    "TextAnimator",
    "Transform",
    "Style",
    "Glow",
    "Shadow",
    "Repeat",
    "Layout",
    "AnimationTarget",
    "Key",
    "ParticleEmitter",
    "SpringChain",
    "DynamicCurve",
    "DistanceConstraint",
    "Hinge",
    "RigidBody2D",
    "Cloth",
    "HairStrandField",
    "CacheBake",
    "Gravity",
    "Wind",
    "Attraction",
    "Collider",
    "Process",
    "Input",
    "Tex",
    "Buffer",
    "Pass",
    "Output",
    "Present",
    "LinearGradient",
    "RadialGradient",
    "Filter",
    "Blur",
    "ColorMatrix",
    "Material",
    "Texture",
    "Noise",
    "Font",
    "Palette",
    "Color",
    "Effect",
    "Effects",
    "PostEffects",
    "Component",
    "Param",
    "Derived",
    "Slot",
    "Fill",
    "Use",
    "Variants",
    "Vary",
    "Action",
    "Pose",
    "Marker",
    "Bone",
    "IK",
    "ApplyAction",
    "ModelProfile",
    "Retarget",
    "Map",
    "BoneAxisMap",
    "Axis",
    "Skeleton",
    "Landmark",
    "Measure",
    "Ratio",
    "Constraint",
    "Guide",
    "Control",
    "Character",
    "Part",
    "Camera",
    "Mask",
    "Precompose",
    "Puppet",
    "PuppetWarp",
    "PuppetPin",
    "Pin",
    "MeshTopology",
    "Vertex",
    "Triangle",
    "Edge",
    "Region",
    "LimbEnvelope",
    "LimbRegion",
    "PixelGrid",
    "SVG",
    "Svg",
    "Image",
    "Brush",
    "Solid",
    "ParticleField",
    "RadialRays",
    "LightStreak",
    "ChromaticAberration",
    "ColorBleed",
    "HighlightCompression",
    "EdgeSoftness",
    "EdgeRoughness",
];

const ALL_KNOWN_ATTRIBUTES: &[&str] = &[
    "id",
    "version",
    "fps",
    "apply",
    "scope",
    "duration",
    "size",
    "renderSize",
    "src",
    "decoder",
    "colorSpace",
    "color",
    "from",
    "out",
    "space",
    "z",
    "zDepth",
    "role",
    "compositeOrder",
    "x",
    "y",
    "width",
    "height",
    "radius",
    "fill",
    "stroke",
    "strokeWidth",
    "opacity",
    "rotation",
    "scale",
    "scaleX",
    "scaleY",
    "skewX",
    "skewY",
    "transformOriginX",
    "transformOriginY",
    "blend",
    "texture",
    "value",
    "fontSize",
    "fontFamily",
    "fontWeight",
    "lineHeight",
    "tracking",
    "position",
    "target",
    "fov",
    "asset",
    "profile",
    "rig",
    "retarget",
    "rotationX",
    "rotationY",
    "rotationZ",
    "positionX",
    "positionY",
    "positionZ",
    "exposure",
    "material",
    "definition",
    "clip",
    "loop",
    "speed",
    "weight",
    "mask",
    "node",
    "property",
    "time",
    "frame",
    "ease",
    "rate",
    "lifetime",
    "velocity",
    "gravity",
    "kind",
    "effect",
    "kernel",
    "in",
    "params",
    "format",
    "to",
    "stops",
    "units",
    "count",
    "xStep",
    "yStep",
    "rotationStep",
    "opacityStep",
    "seed",
    "selector",
    "stagger",
    "order",
    "name",
    "type",
    "default",
    "values",
    "ref",
    "action",
    "at",
    "skeleton",
    "intent",
    "t",
    "label",
    "parent",
    "root",
    "mid",
    "end",
    "chain",
    "targetX",
    "targetY",
    "targetZ",
    "preset",
];

#[cfg(test)]
mod tests {
    use super::{AuthoringStatus, motionloom_analyze_script_json, motionloom_showcase_schema_json};

    #[test]
    fn report_exposes_ignored_attributes_and_llm_repair_suggestions() {
        let script = r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <Scene id="main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
      <Text id="title" value="Hello" x="20" y="40" strokeOpacity="0.5" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="main" />
</Graph>"##;
        let value: serde_json::Value =
            serde_json::from_str(&motionloom_analyze_script_json(script)).unwrap();
        assert_eq!(value["status"], "needs-review");
        assert_eq!(value["summary"]["ignoredAttributes"], 1);
        assert!(
            value["diagnostics"].as_array().unwrap().iter().any(|item| {
                item["code"] == "UNKNOWN_ATTRIBUTE"
                    && item["attribute"] == "strokeOpacity"
                    && item["line"] == 7
                    && item["suggestions"][0]["replacement"] == "opacity"
                    && !item["suggestions"].as_array().unwrap().is_empty()
            }),
            "{value:#}"
        );
    }

    #[test]
    fn report_keeps_parse_errors_inside_json_instead_of_throwing() {
        let script = r#"<Graph fps={30} duration="1s" size={[320,180]}><Scene>"#;
        let value: serde_json::Value =
            serde_json::from_str(&motionloom_analyze_script_json(script)).unwrap();
        assert_eq!(value["status"], "unrenderable");
        assert_eq!(value["parseSucceeded"], false);
        assert!(value["diagnostics"].as_array().unwrap().iter().any(|item| {
            item["phase"] == "parse" && !item["suggestions"].as_array().unwrap().is_empty()
        }));
    }

    #[test]
    fn showcase_schema_records_used_syntax_and_animation_capability() {
        let script = r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <Scene id="main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
    <Rect id="card" x={curve("0:0,1:20")} y="0" width="20" height="20" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="main" />
</Graph>"##;
        let value: serde_json::Value =
            serde_json::from_str(&motionloom_showcase_schema_json(script)).unwrap();
        let tags = value["tags"].as_array().unwrap();
        let rect = tags.iter().find(|item| item["tag"] == "Rect").unwrap();
        let x = rect["attributes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["name"] == "x")
            .unwrap();
        assert_eq!(x["supportsInlineExpression"], true);
        assert_eq!(x["supportsAnimationTarget"], true);
    }

    #[test]
    fn clean_script_has_clean_status() {
        let script = r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <Scene id="main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
    <Circle x="40" y="40" radius="12" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="main" />
</Graph>"##;
        let report = super::analyze_motionloom_script(script);
        assert_eq!(report.status, AuthoringStatus::Clean);
    }

    #[test]
    fn semantic_bone_and_axis_attributes_are_not_reported_as_ignored() {
        let script = r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets><ModelAsset id="girl_asset" src="girl.glb" /></Assets>
  <ModelProfile id="girl_profile" kind="3d" model="girl_asset" preset="humanoid_v1">
    <Retarget preset="humanoid_v1"><Map from="Arm.R" to="upper_arm_r" /></Retarget>
    <BoneAxisMap>
      <Axis bone="upper_arm_r" forward="rotationZ:-1" side="rotationX:1"
            twist="rotationY:1" restSide="-55" />
    </BoneAxisMap>
  </ModelProfile>
  <Action id="wave" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s"><Bone id="upper_arm_r" forward="0" bend="0" /></Pose>
    <Pose t="1s"><Bone id="upper_arm_r" forward="70" bend="45" /></Pose>
  </Action>
  <Scene id="main"><Timeline><Track><Sequence duration="1s"><Layer>
    <Circle x="40" y="40" radius="12" color="#fff" />
  </Layer></Sequence></Track></Timeline></Scene>
  <Present from="main" />
</Graph>"##;
        let report = super::analyze_motionloom_script(script);
        assert_eq!(report.summary.ignored_attributes, 0, "{report:#?}");
    }

    #[test]
    fn process_compile_failures_are_reported_after_successful_parse() {
        let script = r#"<Graph fps={30} duration="1s" size={[320,180]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[320,180]} />
  <Pass id="broken" kernel="invert_mix.wgsl" effect="invert_mix"
        in={["src"]} out={["out"]} params={{ mix: "not_a_function(" }} />
  <Present from="out" />
</Graph>"#;
        let report = super::analyze_motionloom_script(script);
        assert!(report.parse_succeeded);
        assert!(!report.compile_succeeded);
        assert_eq!(report.status, AuthoringStatus::NeedsRepair);
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "PROCESS_COMPILE_ERROR"
                    && diagnostic.line == 4
                    && diagnostic.node_id.as_deref() == Some("broken"))
        );
    }
}
