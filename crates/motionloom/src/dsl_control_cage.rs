// =========================================
// =========================================
// crates/motionloom/src/dsl_control_cage.rs

use super::*;

/// Parse a standalone polygon mesh with optional Catmull-Clark subdivision.
pub(super) fn parse_mesh_asset(
    lines: &[&str],
    start: usize,
) -> Result<(PrimitiveAssetNode, usize), GraphParseError> {
    parse_asset(lines, start, "MeshAsset", "Vertex", "Face")
}

/// Parse a HeadCage child through the same topology validator as generic surfaces.
pub(super) fn parse_head_cage(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(ControlCageNode, usize), GraphParseError> {
    let (tag, open) = collect_tag_block(lines, start, '>', false)?;
    validate_head_attributes(&tag, &["subdivision"], "HeadCage", start + 1)?;
    if is_self_closing_tag(&tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: "HeadCage requires Vertex and Face children.".into(),
        });
    }
    let subdivision =
        parse_optional_primitive_u32(&tag, "subdivision", asset_id, start + 1)?.unwrap_or(2);
    let end = find_matching_close_tag(lines, open + 1, "HeadCage")?;
    let cage = parse_cage_children(
        lines,
        open + 1,
        end,
        start,
        asset_id,
        "HeadCage",
        "Vertex",
        "Face",
        subdivision,
    )?;
    Ok((cage, end))
}

fn parse_asset(
    lines: &[&str],
    start: usize,
    asset_tag: &str,
    vertex_tag: &str,
    face_tag: &str,
) -> Result<(PrimitiveAssetNode, usize), GraphParseError> {
    let (tag, open) = collect_tag_block(lines, start, '>', false)?;
    let fail = |message: &str| GraphParseError {
        line: start + 1,
        message: message.into(),
    };
    validate_head_attributes(
        &tag,
        &["id", "material", "subdivision", "subdivisionScheme"],
        asset_tag,
        start + 1,
    )?;
    if is_self_closing_tag(&tag) {
        return Err(fail(&format!("{asset_tag} requires a control cage.")));
    }
    let id = strip_wrappers(&required_attr_value(&tag, "id", start + 1)?).to_string();
    let material = strip_wrappers(&required_attr_value(&tag, "material", start + 1)?).to_string();
    let subdivision =
        parse_optional_primitive_u32(&tag, "subdivision", &id, start + 1)?.unwrap_or(0);
    if subdivision > 2 {
        return Err(fail(&format!("{asset_tag} subdivision must be 0..2.")));
    }
    let subdivision_scheme = primitive_string_attribute(&tag, "subdivisionScheme", "catmullclark");
    if subdivision_scheme != "catmullclark" {
        return Err(fail(
            "MeshAsset subdivisionScheme currently accepts catmullClark only.",
        ));
    }
    let end = find_matching_close_tag(lines, open + 1, asset_tag)?;
    let cage = parse_cage_children(
        lines,
        open + 1,
        end,
        start,
        &id,
        asset_tag,
        vertex_tag,
        face_tag,
        subdivision,
    )?;
    Ok((
        PrimitiveAssetNode {
            id,
            geometry: PrimitiveGeometry::Mesh { cage },
            color: [1.0; 4],
            material: Some(material),
            material_definition: None,
            bevel_radius: 0.0,
            bevel_segments: 0,
            material_seed: None,
            collision: PrimitiveCollisionNode::default(),
            modifiers: vec![],
            mesh_build: PrimitiveMeshBuildNode::default(),
            lod: PrimitiveLodNode::default(),
        },
        end,
    ))
}

#[allow(clippy::too_many_arguments)]
fn parse_cage_children(
    lines: &[&str],
    first: usize,
    end: usize,
    start: usize,
    id: &str,
    asset_tag: &str,
    vertex_tag: &str,
    face_tag: &str,
    subdivision: u32,
) -> Result<ControlCageNode, GraphParseError> {
    let fail = |message: &str| GraphParseError {
        line: start + 1,
        message: message.into(),
    };
    if subdivision > 2 {
        return Err(fail(&format!("{asset_tag} subdivision must be 0..2.")));
    }
    let mut cage = ControlCageNode {
        positions: vec![],
        uvs: vec![],
        pinned: vec![],
        faces: vec![],
        subdivision,
    };
    let mut i = first;
    while i < end {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("<!--") || line.starts_with("//") {
            i += 1;
            continue;
        }
        let (child, next) = collect_self_closing_block(lines, i)?;
        if starts_open_tag(line, vertex_tag) {
            validate_head_attributes(&child, &["position", "uv", "pinned"], vertex_tag, i + 1)?;
            let raw = required_attr_value(&child, "position", i + 1)?;
            let position = parse_primitive_vec_value::<3>(&raw, "position", id, i + 1, false)?;
            let uv = parse_optional_primitive_vec::<2>(&child, "uv", id, i + 1, false)?
                .unwrap_or([position[0], position[1]]);
            cage.positions.push(position);
            cage.uvs.push(uv);
            let pin = primitive_string_attribute(&child, "pinned", "false");
            if !matches!(pin.as_str(), "true" | "false") {
                return Err(fail(&format!("{vertex_tag} pinned must be true or false.")));
            }
            cage.pinned.push(pin == "true");
        } else if starts_open_tag(line, face_tag) {
            validate_head_attributes(&child, &["indices"], face_tag, i + 1)?;
            let raw = required_attr_value(&child, "indices", i + 1)?;
            let value = strip_wrappers(&raw)
                .trim()
                .trim_start_matches('[')
                .trim_end_matches(']');
            let face = value
                .split(',')
                .map(|v| v.trim().parse::<u32>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| fail(&format!("{face_tag} indices must be unsigned integers.")))?;
            if !(3..=4).contains(&face.len()) {
                return Err(fail(&format!("{face_tag} requires three or four indices.")));
            }
            cage.faces.push(face);
        } else {
            return Err(fail(&format!(
                "{asset_tag} accepts {vertex_tag} and {face_tag} only."
            )));
        }
        if cage.positions.len() > 30000 || cage.faces.len() > 30000 {
            return Err(fail(&format!(
                "{asset_tag} cage exceeds 30000 vertices/faces."
            )));
        }
        i = next + 1;
    }
    if cage.positions.len() < 3 || cage.faces.is_empty() {
        return Err(fail(&format!("{asset_tag} requires vertices and faces.")));
    }
    // Reject inconsistent winding and non-manifold joins before subdivision.
    let mut edges = std::collections::HashMap::<(u32, u32), Vec<(u32, u32)>>::new();
    for face in &cage.faces {
        if face.iter().any(|&v| v as usize >= cage.positions.len())
            || face.iter().collect::<HashSet<_>>().len() != face.len()
        {
            return Err(fail(&format!(
                "{face_tag} has invalid or repeated indices."
            )));
        }
        for j in 0..face.len() {
            let (a, b) = (face[j], face[(j + 1) % face.len()]);
            let adjacent = edges.entry((a.min(b), a.max(b))).or_default();
            if adjacent.len() == 2 || adjacent.contains(&(a, b)) {
                return Err(fail(&format!(
                    "{asset_tag} has non-manifold or inconsistently wound edges."
                )));
            }
            adjacent.push((a, b));
        }
    }
    Ok(cage)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> String {
        r##"<Graph fps={30} duration="1s" size={[64,64]}>
<Assets>
<MaterialAsset id="clay" baseColor="#aaaaaa" />
<MeshAsset id="surface" material="clay" subdivision="1" subdivisionScheme="catmullClark">
<Vertex position={[-1,0,0]} pinned="true" />
<Vertex position={[1,0,0]} uv={[1,0]} pinned="true" />
<Vertex position={[1,1,0]} />
<Vertex position={[-1,1,0]} />
<Face indices={[0,1,2,3]} />
</MeshAsset>
</Assets>
<Background color="#111111" />
<Present from="scene" />
</Graph>"##
            .to_string()
    }
    #[test]
    fn control_cage_parses_renders_and_invalidates_cache() {
        let graph = parse_graph_script(&source()).unwrap();
        let asset = graph.assets[0].primitive().unwrap();
        let PrimitiveGeometry::Mesh { cage } = &asset.geometry else {
            panic!("MeshAsset must compile to the shared polygon cage IR");
        };
        assert_eq!(
            cage.positions,
            vec![
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0]
            ]
        );
        assert_eq!(
            cage.uvs,
            vec![[-1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]
        );
        assert_eq!(cage.pinned, vec![true, true, false, false]);
        assert_eq!(cage.faces, vec![vec![0, 1, 2, 3]]);
        let mesh = crate::world::primitive::generate_primitive_mesh(asset);
        assert_eq!(mesh.indices.len(), 24);
        let report = crate::authoring::analyze_motionloom_script_for_target(&source(), "auto");
        assert_eq!(report.summary.ignored_attributes, 0);
        let mut changed = asset.clone();
        if let PrimitiveGeometry::Mesh { cage } = &mut changed.geometry {
            cage.uvs[0][0] += 0.25;
        }
        assert_ne!(
            crate::world::primitive::primitive_geometry_cache_key(asset),
            crate::world::primitive::primitive_geometry_cache_key(&changed)
        );
    }

    #[test]
    fn mesh_asset_supports_zero_through_two_subdivision_levels() {
        for (level, expected_indices) in [(0, 6), (1, 24), (2, 96)] {
            let script = source().replace("subdivision=\"1\"", &format!("subdivision=\"{level}\""));
            let graph = parse_graph_script(&script).unwrap();
            let asset = graph.assets[0].primitive().unwrap();
            assert_eq!(
                crate::world::primitive::generate_primitive_mesh(asset)
                    .indices
                    .len(),
                expected_indices
            );
        }
        let defaulted = source().replace(" subdivision=\"1\"", "");
        let graph = parse_graph_script(&defaulted).unwrap();
        let PrimitiveGeometry::Mesh { cage } = &graph.assets[0].primitive().unwrap().geometry
        else {
            panic!()
        };
        assert_eq!(cage.subdivision, 0);
    }

    #[test]
    fn removed_mesh_tag_names_are_rejected() {
        let cases = [
            source().replace("MeshAsset", "SubdivisionSurfaceAsset"),
            source().replacen("Vertex", "ControlVertex", 1),
            source().replace("Face indices", "ControlFace indices"),
        ];
        for old in cases {
            assert!(parse_graph_script(&old).is_err());
        }
    }

    #[test]
    fn mesh_asset_rejects_unknown_subdivision_scheme() {
        assert!(parse_graph_script(&source().replace("catmullClark", "loop")).is_err());
    }
    #[test]
    fn control_cage_rejects_bad_topology_and_limits() {
        for (a, b) in [
            ("0,1,2,3", "0,1,2,99"),
            ("0,1,2,3", "0,1,1,3"),
            ("subdivision=\"1\"", "subdivision=\"3\""),
            ("pinned=\"true\"", "pinned=\"perhaps\""),
            ("</MeshAsset>", "<Face indices={[0,1,2]} />\n</MeshAsset>"),
        ] {
            assert!(
                parse_graph_script(&source().replace(a, b)).is_err(),
                "accepted {b}"
            );
        }
    }
}
