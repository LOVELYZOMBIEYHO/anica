// =========================================
// =========================================
// crates/motionloom/src/mesh_authoring/geometry.rs

use super::{
    Axis, GeometryOperation, GeometryRecipe, MESH_AUTHORING_SCHEMA_VERSION, MeshAuthoringError,
    MeshAuthoringResult, OperationResult, SemanticRegion, TopologyCorrespondence, UvProjection,
};
use crate::ControlCageNode;
use crate::mesh_reference::{
    MeshProposalValidationOptions, mesh_topology_signature, validate_mesh_topology,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f32::consts::{PI, TAU};

#[derive(Clone)]
struct MeshState {
    cage: ControlCageNode,
    regions: BTreeMap<String, SemanticRegion>,
    correspondence: TopologyCorrespondence,
    operations: Vec<OperationResult>,
}

impl MeshState {
    fn empty(subdivision: u32) -> Self {
        Self {
            cage: ControlCageNode {
                positions: vec![],
                uvs: vec![],
                pinned: vec![],
                faces: vec![],
                subdivision,
            },
            regions: BTreeMap::new(),
            correspondence: TopologyCorrespondence::default(),
            operations: vec![],
        }
    }

    fn from_cage(cage: ControlCageNode, regions: Vec<SemanticRegion>) -> Self {
        let mut correspondence = TopologyCorrespondence::default();
        for index in 0..cage.positions.len() as u64 {
            correspondence
                .old_to_new_vertices
                .insert(index, vec![index]);
        }
        for index in 0..cage.faces.len() as u64 {
            correspondence.old_to_new_faces.insert(index, vec![index]);
        }
        Self {
            cage,
            regions: regions
                .into_iter()
                .map(|region| (region.id.clone(), region))
                .collect(),
            correspondence,
            operations: vec![],
        }
    }

    fn add_vertex(&mut self, position: [f32; 3]) -> u64 {
        let id = self.cage.positions.len() as u64;
        self.cage.positions.push(position);
        self.cage.uvs.push([0.0; 2]);
        self.cage.pinned.push(false);
        self.correspondence.created_vertices.push(id);
        id
    }

    fn add_face(&mut self, face: Vec<u64>) -> Result<u64, MeshAuthoringError> {
        if !(3..=4).contains(&face.len()) {
            return Err(MeshAuthoringError::Operation(
                "generated faces must have three or four vertices".into(),
            ));
        }
        let id = self.cage.faces.len() as u64;
        self.cage
            .faces
            .push(face.into_iter().map(|value| value as u32).collect());
        self.correspondence.created_faces.push(id);
        Ok(id)
    }

    fn region(&self, id: &str) -> Result<&SemanticRegion, MeshAuthoringError> {
        self.regions
            .get(id)
            .ok_or_else(|| MeshAuthoringError::UnknownRegion(id.into()))
    }

    fn record(&mut self, id: &str, operation: &str, before_v: usize, before_f: usize) {
        self.operations.push(OperationResult {
            id: id.into(),
            operation: operation.into(),
            created_vertices: (before_v as u64..self.cage.positions.len() as u64).collect(),
            created_faces: (before_f as u64..self.cage.faces.len() as u64).collect(),
            affected_regions: vec![id.into()],
        });
    }
}

pub fn execute_geometry_recipe(
    recipe: &GeometryRecipe,
) -> Result<MeshAuthoringResult, MeshAuthoringError> {
    if recipe.schema_version != MESH_AUTHORING_SCHEMA_VERSION {
        return Err(MeshAuthoringError::SchemaVersion(
            recipe.schema_version.clone(),
        ));
    }
    if recipe.subdivision > 2 {
        return Err(MeshAuthoringError::Operation(
            "subdivision must be in the range 0..=2".into(),
        ));
    }
    let mut state = MeshState::empty(recipe.subdivision);
    execute_operations(&mut state, &recipe.operations)?;
    finish(
        recipe.id.clone(),
        state,
        MeshProposalValidationOptions::default(),
    )
}

pub(crate) fn execute_operations_on_cage(
    cage: ControlCageNode,
    regions: Vec<SemanticRegion>,
    operations: &[GeometryOperation],
    validation: MeshProposalValidationOptions,
) -> Result<MeshAuthoringResult, MeshAuthoringError> {
    let mut state = MeshState::from_cage(cage, regions);
    execute_operations(&mut state, operations)?;
    finish("topology-proposal".into(), state, validation)
}

fn finish(
    recipe_id: String,
    state: MeshState,
    validation: MeshProposalValidationOptions,
) -> Result<MeshAuthoringResult, MeshAuthoringError> {
    if state.cage.positions.len() > 30_000 || state.cage.faces.len() > 30_000 {
        return Err(MeshAuthoringError::LimitExceeded {
            vertices: state.cage.positions.len(),
            faces: state.cage.faces.len(),
        });
    }
    let topology = validate_mesh_topology(&state.cage, &validation);
    if !topology.valid {
        return Err(MeshAuthoringError::Topology(topology));
    }
    Ok(MeshAuthoringResult {
        schema_version: MESH_AUTHORING_SCHEMA_VERSION.into(),
        recipe_id,
        topology_signature: mesh_topology_signature(&state.cage),
        cage: state.cage,
        regions: state.regions,
        correspondence: state.correspondence,
        operations: state.operations,
        topology,
    })
}

fn execute_operations(
    state: &mut MeshState,
    operations: &[GeometryOperation],
) -> Result<(), MeshAuthoringError> {
    let mut operation_ids = BTreeSet::new();
    for operation in operations {
        let id = operation_id(operation);
        if !operation_ids.insert(id.to_string()) {
            return Err(MeshAuthoringError::DuplicateOperation(id.into()));
        }
        match operation {
            GeometryOperation::CreateLoop { spec } => create_loop(state, spec)?,
            GeometryOperation::DefineRegion {
                id,
                vertices,
                faces,
            } => {
                validate_indices(state, vertices, faces)?;
                state.regions.insert(
                    id.clone(),
                    SemanticRegion {
                        id: id.clone(),
                        vertices: vertices.clone(),
                        faces: faces.clone(),
                    },
                );
                state.record(
                    id,
                    "defineRegion",
                    state.cage.positions.len(),
                    state.cage.faces.len(),
                );
            }
            GeometryOperation::LoftLoops {
                id,
                loops,
                cap_start,
                cap_end,
            } => {
                loft_loops(state, id, loops, *cap_start, *cap_end)?;
            }
            GeometryOperation::CreateSurface {
                id,
                rows,
                close_columns,
            } => {
                create_surface(state, id, rows, *close_columns)?;
            }
            GeometryOperation::SweepProfile {
                id,
                path,
                radius,
                segments,
            } => {
                sweep_profile(state, id, path, *radius, *segments)?;
            }
            GeometryOperation::ExtrudeRegion {
                id,
                region,
                vector,
                segments,
            } => {
                extrude_region(state, id, region, *vector, *segments)?;
            }
            GeometryOperation::ThickenSurface {
                id,
                region,
                thickness,
            } => {
                thicken_surface(state, id, region, *thickness)?;
            }
            GeometryOperation::TransformRegion {
                id,
                region,
                translate,
                scale,
            } => {
                transform_region(state, id, region, *translate, *scale)?;
            }
            GeometryOperation::MirrorRegion {
                id,
                region,
                axis,
                weld_center,
                tolerance,
            } => {
                mirror_region(state, id, region, *axis, *weld_center, *tolerance)?;
            }
            GeometryOperation::WeldVertices {
                id,
                region,
                tolerance,
            } => {
                weld_vertices(state, id, region.as_deref(), *tolerance)?;
            }
            GeometryOperation::CapBoundary { id, loop_id } => cap_loop(state, id, loop_id)?,
            GeometryOperation::GenerateUv {
                id,
                region,
                projection,
            } => {
                generate_uv(state, id, region.as_deref(), projection)?;
            }
        }
    }
    Ok(())
}

fn operation_id(operation: &GeometryOperation) -> &str {
    match operation {
        GeometryOperation::CreateLoop { spec } => &spec.id,
        GeometryOperation::DefineRegion { id, .. }
        | GeometryOperation::LoftLoops { id, .. }
        | GeometryOperation::CreateSurface { id, .. }
        | GeometryOperation::SweepProfile { id, .. }
        | GeometryOperation::ExtrudeRegion { id, .. }
        | GeometryOperation::ThickenSurface { id, .. }
        | GeometryOperation::TransformRegion { id, .. }
        | GeometryOperation::MirrorRegion { id, .. }
        | GeometryOperation::WeldVertices { id, .. }
        | GeometryOperation::CapBoundary { id, .. }
        | GeometryOperation::GenerateUv { id, .. } => id,
    }
}

fn create_loop(state: &mut MeshState, spec: &super::LoopSpec) -> Result<(), MeshAuthoringError> {
    if spec.segments < 3 || spec.radius_a <= 0.0 || spec.radius_b <= 0.0 {
        return Err(MeshAuthoringError::Operation(format!(
            "loop {} requires at least three segments and positive radii",
            spec.id
        )));
    }
    let before = state.cage.positions.len();
    let rotation = spec.rotation_degrees.to_radians();
    let mut vertices = vec![];
    for index in 0..spec.segments {
        let angle = index as f32 / spec.segments as f32 * TAU + rotation;
        let (a, b) = (spec.radius_a * angle.cos(), spec.radius_b * angle.sin());
        let position = match spec.normal_axis {
            Axis::X => [spec.center[0], spec.center[1] + a, spec.center[2] + b],
            Axis::Y => [spec.center[0] + a, spec.center[1], spec.center[2] + b],
            Axis::Z => [spec.center[0] + a, spec.center[1] + b, spec.center[2]],
        };
        vertices.push(state.add_vertex(position));
    }
    state.regions.insert(
        spec.id.clone(),
        SemanticRegion {
            id: spec.id.clone(),
            vertices,
            faces: vec![],
        },
    );
    state.record(&spec.id, "createLoop", before, state.cage.faces.len());
    Ok(())
}

fn loft_loops(
    state: &mut MeshState,
    id: &str,
    loop_ids: &[String],
    cap_start: bool,
    cap_end: bool,
) -> Result<(), MeshAuthoringError> {
    if loop_ids.len() < 2 {
        return Err(MeshAuthoringError::Operation(
            "loft requires at least two loops".into(),
        ));
    }
    let loops: Vec<_> = loop_ids
        .iter()
        .map(|name| state.region(name).map(|region| region.vertices.clone()))
        .collect::<Result<_, _>>()?;
    let count = loops[0].len();
    if count < 3 || loops.iter().any(|vertices| vertices.len() != count) {
        return Err(MeshAuthoringError::Operation(
            "loft loops must have the same vertex count of at least three".into(),
        ));
    }
    let before_f = state.cage.faces.len();
    for pair in loops.windows(2) {
        for index in 0..count {
            state.add_face(vec![
                pair[0][index],
                pair[1][index],
                pair[1][(index + 1) % count],
                pair[0][(index + 1) % count],
            ])?;
        }
    }
    if cap_start {
        cap_vertices(state, &loops[0], false)?;
    }
    if cap_end {
        cap_vertices(state, loops.last().expect("loft has loops"), true)?;
    }
    let faces = (before_f as u64..state.cage.faces.len() as u64).collect();
    state.regions.insert(
        id.into(),
        SemanticRegion {
            id: id.into(),
            vertices: loops.into_iter().flatten().collect(),
            faces,
        },
    );
    state.record(id, "loftLoops", state.cage.positions.len(), before_f);
    Ok(())
}

fn create_surface(
    state: &mut MeshState,
    id: &str,
    rows: &[Vec<[f32; 3]>],
    close_columns: bool,
) -> Result<(), MeshAuthoringError> {
    if rows.len() < 2 || rows[0].len() < 2 || rows.iter().any(|row| row.len() != rows[0].len()) {
        return Err(MeshAuthoringError::Operation(
            "surface requires at least two equally sized rows with two points".into(),
        ));
    }
    let before_v = state.cage.positions.len();
    let before_f = state.cage.faces.len();
    let grid: Vec<Vec<u64>> = rows
        .iter()
        .map(|row| row.iter().map(|&point| state.add_vertex(point)).collect())
        .collect();
    let columns = rows[0].len();
    let edges = if close_columns { columns } else { columns - 1 };
    for pair in grid.windows(2) {
        for column in 0..edges {
            let next = (column + 1) % columns;
            state.add_face(vec![
                pair[0][column],
                pair[1][column],
                pair[1][next],
                pair[0][next],
            ])?;
        }
    }
    state.regions.insert(
        id.into(),
        SemanticRegion {
            id: id.into(),
            vertices: grid.into_iter().flatten().collect(),
            faces: (before_f as u64..state.cage.faces.len() as u64).collect(),
        },
    );
    state.record(id, "createSurface", before_v, before_f);
    Ok(())
}

fn sweep_profile(
    state: &mut MeshState,
    id: &str,
    path: &[[f32; 3]],
    radius: f32,
    segments: usize,
) -> Result<(), MeshAuthoringError> {
    if path.len() < 2 || segments < 3 || radius <= 0.0 {
        return Err(MeshAuthoringError::Operation(
            "sweep requires two path points, three profile segments, and positive radius".into(),
        ));
    }
    let before_v = state.cage.positions.len();
    let before_f = state.cage.faces.len();
    let mut loops = vec![];
    for (index, &point) in path.iter().enumerate() {
        let tangent = normalize(sub(
            path[(index + 1).min(path.len() - 1)],
            path[index.saturating_sub(1)],
        ))?;
        let reference = if tangent[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let right = normalize(cross(tangent, reference))?;
        let up = cross(right, tangent);
        let mut ring = vec![];
        for segment in 0..segments {
            let angle = segment as f32 / segments as f32 * TAU;
            ring.push(state.add_vertex(add(
                point,
                add(
                    scale(right, radius * angle.cos()),
                    scale(up, radius * angle.sin()),
                ),
            )));
        }
        loops.push(ring);
    }
    for pair in loops.windows(2) {
        for index in 0..segments {
            state.add_face(vec![
                pair[0][index],
                pair[1][index],
                pair[1][(index + 1) % segments],
                pair[0][(index + 1) % segments],
            ])?;
        }
    }
    cap_vertices(state, &loops[0], false)?;
    cap_vertices(state, loops.last().expect("sweep has loops"), true)?;
    state.regions.insert(
        id.into(),
        SemanticRegion {
            id: id.into(),
            vertices: loops.into_iter().flatten().collect(),
            faces: (before_f as u64..state.cage.faces.len() as u64).collect(),
        },
    );
    state.record(id, "sweepProfile", before_v, before_f);
    Ok(())
}

fn extrude_region(
    state: &mut MeshState,
    id: &str,
    region_id: &str,
    vector: [f32; 3],
    segments: usize,
) -> Result<(), MeshAuthoringError> {
    if segments == 0 || length(vector) <= 1e-8 {
        return Err(MeshAuthoringError::Operation(
            "extrusion requires non-zero vector and at least one segment".into(),
        ));
    }
    let region = state.region(region_id)?.clone();
    if region.faces.is_empty() {
        return Err(MeshAuthoringError::Operation(
            "extrusion region has no faces".into(),
        ));
    }
    let selected_faces: BTreeSet<usize> =
        region.faces.iter().map(|&value| value as usize).collect();
    if selected_faces
        .iter()
        .any(|&index| index >= state.cage.faces.len())
    {
        return Err(MeshAuthoringError::Operation(
            "extrusion face is out of range".into(),
        ));
    }
    let selected_vertices: BTreeSet<u32> = selected_faces
        .iter()
        .flat_map(|&index| state.cage.faces[index].iter().copied())
        .collect();
    let boundary = boundary_edges(&state.cage.faces, &selected_faces);
    let top_template: Vec<Vec<u32>> = selected_faces
        .iter()
        .map(|&i| state.cage.faces[i].clone())
        .collect();
    let before_v = state.cage.positions.len();
    let old_faces = state.cage.faces.clone();
    let mut face_mapping = BTreeMap::new();
    state.cage.faces = state
        .cage
        .faces
        .iter()
        .enumerate()
        .filter(|(index, _)| !selected_faces.contains(index))
        .map(|(_, face)| face.clone())
        .collect();
    let mut next_face = 0_u64;
    for old_face in 0..old_faces.len() {
        if !selected_faces.contains(&old_face) {
            face_mapping.insert(old_face as u64, vec![next_face]);
            next_face += 1;
        }
    }
    let created_face_start = state.cage.faces.len();
    let mut previous: HashMap<u32, u64> =
        selected_vertices.iter().map(|&v| (v, v as u64)).collect();
    for step in 1..=segments {
        let mut current = HashMap::new();
        for &vertex in &selected_vertices {
            let position = add(
                state.cage.positions[vertex as usize],
                scale(vector, step as f32 / segments as f32),
            );
            current.insert(vertex, state.add_vertex(position));
        }
        for &(a, b) in &boundary {
            state.add_face(vec![previous[&a], previous[&b], current[&b], current[&a]])?;
        }
        previous = current;
    }
    let top_start = state.cage.faces.len();
    for (&old_face, face) in selected_faces.iter().zip(&top_template) {
        let new_face = state.add_face(face.iter().map(|vertex| previous[vertex]).collect())?;
        face_mapping.insert(old_face as u64, vec![new_face]);
    }
    let new_region = SemanticRegion {
        id: id.into(),
        vertices: previous.values().copied().collect(),
        faces: (top_start as u64..state.cage.faces.len() as u64).collect(),
    };
    for existing in state.regions.values_mut() {
        existing.faces = existing
            .faces
            .iter()
            .filter_map(|old| face_mapping.get(old))
            .flatten()
            .copied()
            .collect();
    }
    state.regions.remove(region_id);
    state.regions.insert(id.into(), new_region);
    state.correspondence.old_to_new_faces = face_mapping;
    state
        .correspondence
        .invalidated_regions
        .push(region_id.into());
    state.record(id, "extrudeRegion", before_v, created_face_start);
    Ok(())
}

fn thicken_surface(
    state: &mut MeshState,
    id: &str,
    region_id: &str,
    thickness: f32,
) -> Result<(), MeshAuthoringError> {
    if thickness.abs() <= 1e-8 {
        return Err(MeshAuthoringError::Operation(
            "thickness must be non-zero".into(),
        ));
    }
    let region = state.region(region_id)?.clone();
    let selected_faces: BTreeSet<usize> =
        region.faces.iter().map(|&value| value as usize).collect();
    let vertices: BTreeSet<u32> = selected_faces
        .iter()
        .flat_map(|&face| state.cage.faces.get(face).into_iter().flatten().copied())
        .collect();
    if vertices.is_empty() {
        return Err(MeshAuthoringError::Operation(
            "thicken region has no faces".into(),
        ));
    }
    let mut normals = HashMap::<u32, [f32; 3]>::new();
    for &face_id in &selected_faces {
        let face = &state.cage.faces[face_id];
        let normal = face_normal(&state.cage, face)?;
        for &vertex in face {
            let entry = normals.entry(vertex).or_insert([0.0; 3]);
            *entry = add(*entry, normal);
        }
    }
    let before_v = state.cage.positions.len();
    let before_f = state.cage.faces.len();
    let mut duplicate = HashMap::new();
    for vertex in vertices {
        let normal = normalize(normals[&vertex])?;
        duplicate.insert(
            vertex,
            state.add_vertex(add(
                state.cage.positions[vertex as usize],
                scale(normal, thickness),
            )),
        );
    }
    for &face_id in &selected_faces {
        let face = &state.cage.faces[face_id].clone();
        state.add_face(face.iter().rev().map(|vertex| duplicate[vertex]).collect())?;
    }
    for (a, b) in boundary_edges(&state.cage.faces[..before_f], &selected_faces) {
        state.add_face(vec![a as u64, duplicate[&a], duplicate[&b], b as u64])?;
    }
    state.regions.insert(
        id.into(),
        SemanticRegion {
            id: id.into(),
            vertices: (before_v as u64..state.cage.positions.len() as u64).collect(),
            faces: (before_f as u64..state.cage.faces.len() as u64).collect(),
        },
    );
    state.record(id, "thickenSurface", before_v, before_f);
    Ok(())
}

fn transform_region(
    state: &mut MeshState,
    id: &str,
    region_id: &str,
    translate: [f32; 3],
    scale_value: [f32; 3],
) -> Result<(), MeshAuthoringError> {
    let region = state.region(region_id)?.clone();
    let vertices = region_vertices(state, &region)?;
    if vertices.is_empty() {
        return Err(MeshAuthoringError::Operation(
            "transform region is empty".into(),
        ));
    }
    let center = scale(
        vertices.iter().fold([0.0; 3], |sum, &vertex| {
            add(sum, state.cage.positions[vertex])
        }),
        1.0 / vertices.len() as f32,
    );
    for vertex in vertices {
        let local = sub(state.cage.positions[vertex], center);
        state.cage.positions[vertex] = add(
            add(
                center,
                [
                    local[0] * scale_value[0],
                    local[1] * scale_value[1],
                    local[2] * scale_value[2],
                ],
            ),
            translate,
        );
    }
    state.record(
        id,
        "transformRegion",
        state.cage.positions.len(),
        state.cage.faces.len(),
    );
    Ok(())
}

fn mirror_region(
    state: &mut MeshState,
    id: &str,
    region_id: &str,
    axis: Axis,
    weld_center: bool,
    tolerance: f32,
) -> Result<(), MeshAuthoringError> {
    let region = state.region(region_id)?.clone();
    let vertices = region_vertices(state, &region)?;
    let before_v = state.cage.positions.len();
    let before_f = state.cage.faces.len();
    let mut mapped = HashMap::<usize, u64>::new();
    for vertex in vertices {
        let mut point = state.cage.positions[vertex];
        point[axis_index(axis)] *= -1.0;
        mapped.insert(vertex, state.add_vertex(point));
    }
    let mut faces = vec![];
    for &face_id in &region.faces {
        let face = state.cage.faces.get(face_id as usize).ok_or_else(|| {
            MeshAuthoringError::Operation(format!("mirror face {face_id} is out of range"))
        })?;
        if face
            .iter()
            .all(|vertex| mapped.contains_key(&(*vertex as usize)))
        {
            let new_face: Vec<_> = face
                .iter()
                .rev()
                .map(|vertex| mapped[&(*vertex as usize)])
                .collect();
            faces.push(state.add_face(new_face)?);
        }
    }
    state.regions.insert(
        id.into(),
        SemanticRegion {
            id: id.into(),
            vertices: mapped.values().copied().collect(),
            faces,
        },
    );
    if weld_center {
        weld_vertices(state, &format!("{id}.weld"), None, tolerance)?;
    }
    state.record(id, "mirrorRegion", before_v, before_f);
    Ok(())
}

fn weld_vertices(
    state: &mut MeshState,
    id: &str,
    region_id: Option<&str>,
    tolerance: f32,
) -> Result<(), MeshAuthoringError> {
    if tolerance <= 0.0 {
        return Err(MeshAuthoringError::Operation(
            "weld tolerance must be positive".into(),
        ));
    }
    let candidates: BTreeSet<usize> = if let Some(region_id) = region_id {
        region_vertices(state, &state.region(region_id)?.clone())?
            .into_iter()
            .collect()
    } else {
        (0..state.cage.positions.len()).collect()
    };
    let old_count = state.cage.positions.len();
    let mut representatives: Vec<usize> = (0..old_count).collect();
    let list: Vec<_> = candidates.into_iter().collect();
    for (offset, &a) in list.iter().enumerate() {
        for &b in &list[offset + 1..] {
            if distance(state.cage.positions[a], state.cage.positions[b]) <= tolerance {
                representatives[b] = representatives[a];
            }
        }
    }
    for index in 0..old_count {
        let mut root = representatives[index];
        while representatives[root] != root {
            root = representatives[root];
        }
        representatives[index] = root;
    }
    let mut new_index = HashMap::new();
    let mut positions = vec![];
    let mut uvs = vec![];
    let mut pinned = vec![];
    for root in representatives.iter().copied() {
        if let std::collections::hash_map::Entry::Vacant(entry) = new_index.entry(root) {
            let index = positions.len();
            entry.insert(index);
            positions.push(state.cage.positions[root]);
            uvs.push(state.cage.uvs.get(root).copied().unwrap_or([0.0; 2]));
            pinned.push(state.cage.pinned.get(root).copied().unwrap_or(false));
        }
    }
    let mapping: Vec<usize> = representatives.iter().map(|root| new_index[root]).collect();
    let old_faces = state.cage.faces.clone();
    let mut faces = vec![];
    let mut face_mapping = BTreeMap::<u64, Vec<u64>>::new();
    for (face_id, face) in old_faces.iter().enumerate() {
        let face: Vec<u32> = face
            .iter()
            .map(|index| mapping[*index as usize] as u32)
            .collect();
        if face.iter().copied().collect::<BTreeSet<_>>().len() >= 3 {
            face_mapping.insert(face_id as u64, vec![faces.len() as u64]);
            faces.push(face);
        } else {
            state.correspondence.removed_faces.push(face_id as u64);
        }
    }
    state.cage.positions = positions;
    state.cage.uvs = uvs;
    state.cage.pinned = pinned;
    state.cage.faces = faces;
    for (old, &new) in mapping.iter().enumerate() {
        state
            .correspondence
            .old_to_new_vertices
            .insert(old as u64, vec![new as u64]);
        if old != new {
            state.correspondence.removed_vertices.push(old as u64);
        }
    }
    state.correspondence.old_to_new_faces = face_mapping;
    for region in state.regions.values_mut() {
        region.vertices = region
            .vertices
            .iter()
            .filter_map(|old| mapping.get(*old as usize).map(|&new| new as u64))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        region.faces = region
            .faces
            .iter()
            .filter_map(|old| state.correspondence.old_to_new_faces.get(old))
            .flatten()
            .copied()
            .collect();
    }
    state.record(
        id,
        "weldVertices",
        state.cage.positions.len(),
        state.cage.faces.len(),
    );
    Ok(())
}

fn cap_loop(state: &mut MeshState, id: &str, loop_id: &str) -> Result<(), MeshAuthoringError> {
    let vertices = state.region(loop_id)?.vertices.clone();
    let before_v = state.cage.positions.len();
    let before_f = state.cage.faces.len();
    cap_vertices(state, &vertices, false)?;
    state.regions.insert(
        id.into(),
        SemanticRegion {
            id: id.into(),
            vertices,
            faces: (before_f as u64..state.cage.faces.len() as u64).collect(),
        },
    );
    state.record(id, "capBoundary", before_v, before_f);
    Ok(())
}

fn cap_vertices(
    state: &mut MeshState,
    vertices: &[u64],
    reverse: bool,
) -> Result<(), MeshAuthoringError> {
    if vertices.len() < 3 {
        return Err(MeshAuthoringError::Operation(
            "cap requires at least three vertices".into(),
        ));
    }
    let center = scale(
        vertices.iter().fold([0.0; 3], |sum, &id| {
            add(sum, state.cage.positions[id as usize])
        }),
        1.0 / vertices.len() as f32,
    );
    let pole = state.add_vertex(center);
    for index in 0..vertices.len() {
        let mut face = vec![
            pole,
            vertices[index],
            vertices[(index + 1) % vertices.len()],
        ];
        if reverse {
            face.reverse();
        }
        state.add_face(face)?;
    }
    Ok(())
}

fn generate_uv(
    state: &mut MeshState,
    id: &str,
    region_id: Option<&str>,
    projection: &UvProjection,
) -> Result<(), MeshAuthoringError> {
    let vertices: Vec<usize> = if let Some(region_id) = region_id {
        region_vertices(state, &state.region(region_id)?.clone())?
    } else {
        (0..state.cage.positions.len()).collect()
    };
    for vertex in vertices {
        let point = state.cage.positions[vertex];
        state.cage.uvs[vertex] = project_uv(point, projection)?;
    }
    state.record(
        id,
        "generateUv",
        state.cage.positions.len(),
        state.cage.faces.len(),
    );
    Ok(())
}

fn project_uv(point: [f32; 3], projection: &UvProjection) -> Result<[f32; 2], MeshAuthoringError> {
    Ok(match projection {
        UvProjection::Planar {
            u_axis,
            v_axis,
            scale,
            offset,
        } => [
            point[axis_index(*u_axis)] * scale[0] + offset[0],
            point[axis_index(*v_axis)] * scale[1] + offset[1],
        ],
        UvProjection::Cylindrical {
            axis,
            scale,
            offset,
        } => {
            let (a, b) = perpendicular_axes(*axis);
            [
                (point[b].atan2(point[a]) / TAU + 0.5) * scale[0] + offset[0],
                point[axis_index(*axis)] * scale[1] + offset[1],
            ]
        }
        UvProjection::Spherical { scale, offset } => {
            let radius = length(point).max(1e-8);
            [
                (point[2].atan2(point[0]) / TAU + 0.5) * scale[0] + offset[0],
                ((point[1] / radius).clamp(-1.0, 1.0).asin() / PI + 0.5) * scale[1] + offset[1],
            ]
        }
        UvProjection::ReferenceCamera {
            origin,
            right,
            up,
            forward,
            focal,
            center,
            image_size,
        } => {
            let relative = sub(point, *origin);
            let depth = dot(relative, *forward);
            if depth.abs() <= 1e-8 {
                return Err(MeshAuthoringError::Operation(
                    "reference-camera UV has zero depth".into(),
                ));
            }
            [
                (center[0] + dot(relative, *right) * focal / depth) / image_size[0] as f32,
                (center[1] - dot(relative, *up) * focal / depth) / image_size[1] as f32,
            ]
        }
    })
}

fn validate_indices(
    state: &MeshState,
    vertices: &[u64],
    faces: &[u64],
) -> Result<(), MeshAuthoringError> {
    if vertices
        .iter()
        .any(|&id| id as usize >= state.cage.positions.len())
        || faces
            .iter()
            .any(|&id| id as usize >= state.cage.faces.len())
    {
        return Err(MeshAuthoringError::Operation(
            "region contains an out-of-range handle".into(),
        ));
    }
    Ok(())
}

fn region_vertices(
    state: &MeshState,
    region: &SemanticRegion,
) -> Result<Vec<usize>, MeshAuthoringError> {
    let mut vertices: BTreeSet<usize> = region
        .vertices
        .iter()
        .map(|&value| value as usize)
        .collect();
    for &face in &region.faces {
        let face = state.cage.faces.get(face as usize).ok_or_else(|| {
            MeshAuthoringError::Operation(format!("region {} has out-of-range face", region.id))
        })?;
        vertices.extend(face.iter().map(|&value| value as usize));
    }
    if vertices
        .iter()
        .any(|&index| index >= state.cage.positions.len())
    {
        return Err(MeshAuthoringError::Operation(format!(
            "region {} has out-of-range vertex",
            region.id
        )));
    }
    Ok(vertices.into_iter().collect())
}

fn boundary_edges(faces: &[Vec<u32>], selected: &BTreeSet<usize>) -> Vec<(u32, u32)> {
    let mut edges = BTreeMap::<(u32, u32), Vec<(u32, u32)>>::new();
    for &face_id in selected {
        if let Some(face) = faces.get(face_id) {
            for index in 0..face.len() {
                let pair = (face[index], face[(index + 1) % face.len()]);
                edges
                    .entry((pair.0.min(pair.1), pair.0.max(pair.1)))
                    .or_default()
                    .push(pair);
            }
        }
    }
    edges
        .values()
        .filter(|uses| uses.len() == 1)
        .map(|uses| uses[0])
        .collect()
}

fn face_normal(cage: &ControlCageNode, face: &[u32]) -> Result<[f32; 3], MeshAuthoringError> {
    if face.len() < 3 {
        return Err(MeshAuthoringError::Operation(
            "face has fewer than three vertices".into(),
        ));
    }
    normalize(cross(
        sub(
            cage.positions[face[1] as usize],
            cage.positions[face[0] as usize],
        ),
        sub(
            cage.positions[face[2] as usize],
            cage.positions[face[0] as usize],
        ),
    ))
}

fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

fn perpendicular_axes(axis: Axis) -> (usize, usize) {
    match axis {
        Axis::X => (1, 2),
        Axis::Y => (0, 2),
        Axis::Z => (0, 1),
    }
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], value: f32) -> [f32; 3] {
    [a[0] * value, a[1] * value, a[2] * value]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    length(sub(a, b))
}

fn normalize(a: [f32; 3]) -> Result<[f32; 3], MeshAuthoringError> {
    let length = length(a);
    if length <= 1e-8 || !length.is_finite() {
        return Err(MeshAuthoringError::Operation(
            "cannot normalize a zero-length vector".into(),
        ));
    }
    Ok(scale(a, 1.0 / length))
}
