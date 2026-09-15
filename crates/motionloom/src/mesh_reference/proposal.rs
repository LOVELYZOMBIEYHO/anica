// =========================================
// =========================================
// crates/motionloom/src/mesh_reference/proposal.rs

use super::{
    ApplyMeshProposalResult, MESH_REFERENCE_SCHEMA_VERSION, MeshAssetProposal, MeshReferenceError,
    validate_mesh_topology,
};
use crate::{ControlCageNode, PrimitiveGeometry, parse_graph_script};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};

pub fn mesh_source_fingerprint(source: &str) -> String {
    format!("{:x}", Sha256::digest(source.as_bytes()))
}

pub fn mesh_topology_signature(cage: &ControlCageNode) -> String {
    let mut hash = Sha256::new();
    hash.update((cage.positions.len() as u64).to_le_bytes());
    hash.update((cage.faces.len() as u64).to_le_bytes());
    for face in &cage.faces {
        hash.update((face.len() as u64).to_le_bytes());
        for &index in face {
            hash.update(index.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

pub fn apply_mesh_asset_proposal(
    source: &str,
    proposal: &MeshAssetProposal,
) -> Result<ApplyMeshProposalResult, MeshReferenceError> {
    if proposal.schema_version != MESH_REFERENCE_SCHEMA_VERSION {
        return Err(MeshReferenceError::Request(format!(
            "unsupported schemaVersion {}; expected {MESH_REFERENCE_SCHEMA_VERSION}",
            proposal.schema_version
        )));
    }
    if mesh_source_fingerprint(source) != proposal.source_fingerprint {
        return Err(MeshReferenceError::Source(
            "stale source fingerprint".into(),
        ));
    }
    let mut cage = mesh_asset(source, &proposal.target_asset_id)?;
    if mesh_topology_signature(&cage) != proposal.topology_signature {
        return Err(MeshReferenceError::Source(
            "stale topology signature".into(),
        ));
    }
    if proposal.changes.is_empty() {
        return Err(MeshReferenceError::Request(
            "proposal contains no vertex changes".into(),
        ));
    }
    if proposal.changes.len() > proposal.validation.max_changed_vertices.max(1) {
        return Err(MeshReferenceError::Request(format!(
            "proposal changes {} vertices, above the configured batch limit of {}",
            proposal.changes.len(),
            proposal.validation.max_changed_vertices.max(1)
        )));
    }
    validate_evaluation_context(proposal)?;
    let original = cage.clone();
    let mut seen = BTreeSet::new();
    let extent = cage_extent(&cage).max(1e-6);
    for change in &proposal.changes {
        if !seen.insert(change.vertex) {
            return Err(MeshReferenceError::Request(format!(
                "vertex {} appears more than once",
                change.vertex
            )));
        }
        let Some(current) = cage.positions.get(change.vertex).copied() else {
            return Err(MeshReferenceError::Request(format!(
                "vertex {} is out of range",
                change.vertex
            )));
        };
        if cage.pinned.get(change.vertex).copied().unwrap_or(false) {
            return Err(MeshReferenceError::Candidate(format!(
                "vertex {} is pinned",
                change.vertex
            )));
        }
        if current != change.before {
            return Err(MeshReferenceError::Source(format!(
                "vertex {} before value does not match source",
                change.vertex
            )));
        }
        if change.after.iter().any(|value| !value.is_finite()) {
            return Err(MeshReferenceError::Candidate(format!(
                "vertex {} after value is not finite",
                change.vertex
            )));
        }
        let movement = distance(change.before, change.after);
        if movement > extent * proposal.validation.max_move_relative_to_bounds.max(0.0) {
            return Err(MeshReferenceError::Candidate(format!(
                "vertex {} moves {:.6}, above the configured relative bound",
                change.vertex, movement
            )));
        }
        cage.positions[change.vertex] = change.after;
    }
    validate_deformation(&original, &cage, &seen, proposal)?;
    let topology = validate_mesh_topology(&cage, &proposal.validation);
    if !topology.valid {
        return Err(MeshReferenceError::Candidate(format!(
            "proposal failed topology validation: {}",
            topology
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.severity == "error")
                .map(|diagnostic| diagnostic.code.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let replacements: HashMap<_, _> = proposal
        .changes
        .iter()
        .map(|change| (change.vertex, change.after))
        .collect();
    let next_source = rewrite_vertex_positions(source, &proposal.target_asset_id, &replacements)?;
    let reparsed = mesh_asset(&next_source, &proposal.target_asset_id)?;
    if reparsed.positions != cage.positions || reparsed.faces != cage.faces {
        return Err(MeshReferenceError::Source(
            "patched source does not reproduce the validated candidate".into(),
        ));
    }
    Ok(ApplyMeshProposalResult {
        schema_version: MESH_REFERENCE_SCHEMA_VERSION.into(),
        source_fingerprint: mesh_source_fingerprint(&next_source),
        topology_signature: mesh_topology_signature(&reparsed),
        source: next_source,
        applied_changes: proposal.changes.len(),
        topology,
    })
}

fn validate_evaluation_context(proposal: &MeshAssetProposal) -> Result<(), MeshReferenceError> {
    let supplied = [
        proposal.camera_fingerprint.as_ref(),
        proposal.reference_set_fingerprint.as_ref(),
        proposal.metric_profile_fingerprint.as_ref(),
        proposal.evaluation_fingerprint.as_ref(),
    ];
    if supplied.iter().all(|value| value.is_none()) {
        return Ok(());
    }
    if supplied.iter().any(|value| value.is_none()) {
        return Err(MeshReferenceError::Request(
            "camera, reference-set, metric-profile, and evaluation fingerprints must be supplied together".into(),
        ));
    }
    let expected = super::evaluation::evaluation_context_fingerprint(
        &proposal.source_fingerprint,
        &proposal.topology_signature,
        proposal.camera_fingerprint.as_deref().unwrap_or_default(),
        proposal
            .reference_set_fingerprint
            .as_deref()
            .unwrap_or_default(),
        proposal
            .metric_profile_fingerprint
            .as_deref()
            .unwrap_or_default(),
    );
    if proposal.evaluation_fingerprint.as_deref() != Some(expected.as_str()) {
        return Err(MeshReferenceError::Source(
            "stale or inconsistent evaluation fingerprint".into(),
        ));
    }
    Ok(())
}

fn validate_deformation(
    before: &ControlCageNode,
    after: &ControlCageNode,
    changed: &BTreeSet<usize>,
    proposal: &MeshAssetProposal,
) -> Result<(), MeshReferenceError> {
    let extent = cage_extent(before).max(1e-6);
    let edges = mesh_edges(before);
    let maximum_ratio = proposal.validation.max_edge_length_ratio.max(1.0);
    for &(a, b) in &edges {
        if !changed.contains(&a) && !changed.contains(&b) {
            continue;
        }
        let old = distance(before.positions[a], before.positions[b]).max(extent * 1e-7);
        let new = distance(after.positions[a], after.positions[b]);
        let ratio = (new / old).max(old / new.max(extent * 1e-7));
        if ratio > maximum_ratio {
            return Err(MeshReferenceError::Candidate(format!(
                "edge {a}-{b} changes length by {ratio:.3}x, above the configured limit"
            )));
        }
    }
    let adjacency = mesh_adjacency(before, &edges);
    let maximum_laplacian = extent
        * proposal
            .validation
            .max_laplacian_delta_relative_to_bounds
            .max(0.0);
    for &vertex in changed {
        let old = laplacian(before, vertex, &adjacency);
        let new = laplacian(after, vertex, &adjacency);
        if distance(old, new) > maximum_laplacian {
            return Err(MeshReferenceError::Candidate(format!(
                "vertex {vertex} introduces a local deformation spike above the configured limit"
            )));
        }
    }
    Ok(())
}

fn mesh_edges(cage: &ControlCageNode) -> BTreeSet<(usize, usize)> {
    cage.faces
        .iter()
        .flat_map(|face| {
            (0..face.len()).map(move |index| {
                let a = face[index] as usize;
                let b = face[(index + 1) % face.len()] as usize;
                (a.min(b), a.max(b))
            })
        })
        .collect()
}

fn mesh_adjacency(cage: &ControlCageNode, edges: &BTreeSet<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut adjacency = vec![vec![]; cage.positions.len()];
    for &(a, b) in edges {
        if a < adjacency.len() && b < adjacency.len() {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }
    adjacency
}

fn laplacian(cage: &ControlCageNode, vertex: usize, adjacency: &[Vec<usize>]) -> [f32; 3] {
    let neighbors = &adjacency[vertex];
    if neighbors.is_empty() {
        return cage.positions[vertex];
    }
    let mut average = [0.0; 3];
    for &neighbor in neighbors {
        for (axis, component) in average.iter_mut().enumerate() {
            *component += cage.positions[neighbor][axis] / neighbors.len() as f32;
        }
    }
    std::array::from_fn(|axis| cage.positions[vertex][axis] - average[axis])
}

pub fn apply_mesh_asset_proposal_json(
    source: &str,
    proposal_json: &str,
) -> Result<String, MeshReferenceError> {
    let proposal = serde_json::from_str(proposal_json)?;
    Ok(serde_json::to_string(&apply_mesh_asset_proposal(
        source, &proposal,
    )?)?)
}

pub(crate) fn mesh_asset(source: &str, id: &str) -> Result<ControlCageNode, MeshReferenceError> {
    let graph = parse_graph_script(source)
        .map_err(|error| MeshReferenceError::Candidate(error.to_string()))?;
    let matches: Vec<_> = graph.assets.iter().filter(|asset| asset.id == id).collect();
    if matches.len() != 1 {
        return Err(MeshReferenceError::Candidate(format!(
            "expected one asset named {id}, found {}",
            matches.len()
        )));
    }
    let asset = matches[0]
        .primitive()
        .ok_or_else(|| MeshReferenceError::Candidate(format!("{id} is not a primitive asset")))?;
    let PrimitiveGeometry::Mesh { cage } = &asset.geometry else {
        return Err(MeshReferenceError::Candidate(format!(
            "{id} is not a MeshAsset"
        )));
    };
    Ok(cage.clone())
}

pub(crate) fn rewrite_mesh_asset_cage(
    source: &str,
    asset_id: &str,
    cage: &ControlCageNode,
) -> Result<String, MeshReferenceError> {
    let tags = scan_tags(source);
    let open = tags
        .iter()
        .find(|tag| {
            !tag.closing
                && tag.name == "MeshAsset"
                && attribute(&tag.raw, "id").as_deref() == Some(asset_id)
        })
        .ok_or_else(|| MeshReferenceError::Source(format!("MeshAsset {asset_id} was not found")))?;
    let close = tags
        .iter()
        .find(|tag| tag.closing && tag.name == "MeshAsset" && tag.start > open.start)
        .ok_or_else(|| MeshReferenceError::Source(format!("MeshAsset {asset_id} is not closed")))?;
    let open_end = open.start + open.raw.len();
    let close_end = close.start + close.raw.len();
    let mut body = String::new();
    for (index, position) in cage.positions.iter().enumerate() {
        let uv = cage.uvs.get(index).copied().unwrap_or([0.0; 2]);
        let pinned = cage.pinned.get(index).copied().unwrap_or(false);
        body.push_str(&format!(
            "\n  <Vertex position={{[{}]}} uv={{[{}]}} pinned=\"{}\" />",
            format_vector(position),
            format_vector(&uv),
            pinned
        ));
    }
    for face in &cage.faces {
        body.push_str(&format!(
            "\n  <Face indices={{[{}]}} />",
            format_indices(face)
        ));
    }
    body.push('\n');
    let mut result = String::with_capacity(source.len() + body.len());
    result.push_str(&source[..open_end]);
    result.push_str(&body);
    result.push_str(&source[close.start..close_end]);
    result.push_str(&source[close_end..]);
    Ok(result)
}

fn rewrite_vertex_positions(
    source: &str,
    asset_id: &str,
    changes: &HashMap<usize, [f32; 3]>,
) -> Result<String, MeshReferenceError> {
    let tags = scan_tags(source);
    let mut active = false;
    let mut vertex = 0;
    let mut patches = vec![];
    for tag in tags {
        if tag.name == "MeshAsset" {
            active = !tag.closing && attribute(&tag.raw, "id").as_deref() == Some(asset_id);
            continue;
        }
        if !active || tag.closing || tag.name != "Vertex" {
            continue;
        }
        if let Some(value) = changes.get(&vertex) {
            let relative = position_span(&tag.raw).ok_or_else(|| {
                MeshReferenceError::Source(format!("Vertex {vertex} has no literal position"))
            })?;
            patches.push((
                tag.start + relative.0,
                tag.start + relative.1,
                format_position(*value),
            ));
        }
        vertex += 1;
    }
    if patches.len() != changes.len() {
        return Err(MeshReferenceError::Source(
            "source changed or requested vertices were not found".into(),
        ));
    }
    let mut result = source.to_owned();
    for (start, end, replacement) in patches.into_iter().rev() {
        result.replace_range(start..end, &replacement);
    }
    Ok(result)
}

#[derive(Debug)]
struct Tag {
    name: String,
    raw: String,
    start: usize,
    closing: bool,
}

fn scan_tags(source: &str) -> Vec<Tag> {
    let bytes = source.as_bytes();
    let mut result = vec![];
    let mut index = 0;
    while index < bytes.len() {
        if source[index..].starts_with("<!--") {
            index = source[index + 4..]
                .find("-->")
                .map_or(bytes.len(), |offset| index + 4 + offset + 3);
            continue;
        }
        if bytes[index] != b'<' {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        let closing = bytes.get(index) == Some(&b'/');
        if closing {
            index += 1;
        }
        let name_start = index;
        while bytes
            .get(index)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            index += 1;
        }
        if index == name_start {
            continue;
        }
        let name = source[name_start..index].to_owned();
        let mut quote = None;
        let mut braces = 0_i32;
        while index < bytes.len() {
            let byte = bytes[index];
            if let Some(expected) = quote {
                if byte == expected && bytes.get(index.wrapping_sub(1)) != Some(&b'\\') {
                    quote = None;
                }
            } else if byte == b'\'' || byte == b'"' {
                quote = Some(byte);
            } else if byte == b'{' {
                braces += 1;
            } else if byte == b'}' {
                braces -= 1;
            } else if byte == b'>' && braces == 0 {
                index += 1;
                result.push(Tag {
                    name,
                    raw: source[start..index].to_owned(),
                    start,
                    closing,
                });
                break;
            }
            index += 1;
        }
    }
    result
}

fn attribute(raw: &str, key: &str) -> Option<String> {
    let marker = format!("{key}=");
    let start = raw.find(&marker)? + marker.len();
    let quote = *raw.as_bytes().get(start)?;
    if quote != b'\'' && quote != b'"' {
        return None;
    }
    let tail = &raw[start + 1..];
    let end = tail.find(quote as char)?;
    Some(tail[..end].into())
}

fn position_span(raw: &str) -> Option<(usize, usize)> {
    let start = raw.find("position")?;
    let equals = raw[start..].find('=')? + start;
    let open = raw[equals..].find('{')? + equals;
    let close = raw[open..].find('}')? + open + 1;
    Some((start, close))
}

fn format_position(value: [f32; 3]) -> String {
    let values = value.map(|number| {
        let mut text = format!("{number:.7}");
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        if text == "-0" {
            text = "0".into();
        }
        text
    });
    format!("position={{[{}, {}, {}]}}", values[0], values[1], values[2])
}

fn format_vector<const N: usize>(value: &[f32; N]) -> String {
    value
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_indices(value: &[u32]) -> String {
    value
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn cage_extent(cage: &ControlCageNode) -> f32 {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for position in &cage.positions {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(position[axis]);
            maximum[axis] = maximum[axis].max(position[axis]);
        }
    }
    (0..3)
        .map(|axis| maximum[axis] - minimum[axis])
        .fold(0.0, f32::max)
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
