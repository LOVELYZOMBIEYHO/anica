// =========================================
// =========================================
// crates/motionloom/src/world/gltf_loader.rs

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine as _;
use draco_gltf::{
    ComponentType as DracoComponentType, MeshIndex as DracoMeshIndex, PackedAttribute,
    PackedGeometry, PrimitiveIndex as DracoPrimitiveIndex, PrimitiveMode as DracoPrimitiveMode,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

const GLB_MAGIC: &[u8; 4] = b"glTF";
const GLB_JSON_CHUNK: u32 = 0x4E4F534A;
const GLB_BIN_CHUNK: u32 = 0x004E4942;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbMetadata {
    pub path: PathBuf,
    pub version: u32,
    pub json_len: usize,
    pub bin_len: usize,
    pub scene_count: usize,
    pub node_count: usize,
    pub mesh_count: usize,
    pub material_count: usize,
    pub skin_count: usize,
    pub animation_count: usize,
    pub node_names: Vec<String>,
    pub joint_names: Vec<String>,
    pub animation_names: Vec<String>,
    pub has_skin: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbMeshData {
    pub path: PathBuf,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<Option<[f32; 3]>>,
    pub texcoords: Vec<Option<[f32; 2]>>,
    pub colors: Vec<Option<[f32; 4]>>,
    pub joints: Vec<Option<[u16; 4]>>,
    pub weights: Vec<Option<[f32; 4]>>,
    pub indices: Vec<u32>,
    pub triangles: Vec<GlbTriangle>,
    pub materials: Vec<GlbMaterialData>,
    pub textures: Vec<Option<GlbTextureData>>,
    pub mesh_names: Vec<Option<String>>,
    pub nodes: Vec<GlbNodeData>,
    pub skin: Option<GlbSkinData>,
    pub animations: Vec<GlbAnimationData>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbAnimationData {
    pub name: Option<String>,
    pub duration: f32,
    pub channels: Vec<GlbAnimationChannelData>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbAnimationChannelData {
    pub node_index: usize,
    pub property: GlbAnimationProperty,
    pub interpolation: GlbAnimationInterpolation,
    pub times: Vec<f32>,
    pub values: GlbAnimationValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlbAnimationProperty {
    Translation,
    Rotation,
    Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlbAnimationInterpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GlbAnimationValues {
    Vec3(Vec<[f32; 3]>),
    Quat(Vec<[f32; 4]>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlbTriangle {
    pub indices: [u32; 3],
    pub material: Option<usize>,
    pub mesh: Option<usize>,
    pub mesh_node: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbMaterialData {
    pub name: Option<String>,
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub normal_scale: f32,
    pub emissive_texture: Option<usize>,
    pub emissive_factor: [f32; 3],
    pub emissive_strength: f32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub specular_factor: f32,
    pub specular_color_factor: [f32; 3],
    pub alpha_mode: GlbAlphaMode,
    pub alpha_cutoff: f32,
    pub transmission_factor: f32,
    pub ior: f32,
    pub thickness_factor: f32,
    pub attenuation_color: [f32; 3],
    pub attenuation_distance: f32,
    pub depth_write: GlbDepthWriteMode,
    pub sort_priority: i32,
    pub double_sided: bool,
    pub unlit: bool,
    pub specular_glossiness: bool,
}

impl Default for GlbMaterialData {
    fn default() -> Self {
        Self {
            name: None,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            emissive_texture: None,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_strength: 1.0,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            specular_factor: 1.0,
            specular_color_factor: [1.0, 1.0, 1.0],
            alpha_mode: GlbAlphaMode::Opaque,
            alpha_cutoff: 0.5,
            transmission_factor: 0.0,
            ior: 1.5,
            thickness_factor: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: 1_000_000.0,
            depth_write: GlbDepthWriteMode::Auto,
            sort_priority: 0,
            double_sided: false,
            unlit: false,
            specular_glossiness: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlbAlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Auto follows material semantics while explicit modes remain available for
/// advanced effects that deliberately manage their own occlusion behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlbDepthWriteMode {
    Auto,
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlbTextureData {
    pub width: u32,
    pub height: u32,
    /// Decoded pixels are immutable and intentionally shared by retained
    /// primitive/material resources that reference the same ImageAsset.
    pub rgba: Arc<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbNodeData {
    pub index: usize,
    pub name: Option<String>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub matrix: Option<[f32; 16]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbSkinData {
    pub skeleton: Option<usize>,
    pub joints: Vec<GlbSkinJointData>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlbSkinJointData {
    pub node_index: usize,
    pub name: Option<String>,
    pub inverse_bind_matrix: [f32; 16],
}

#[derive(Debug, Error)]
pub enum GlbLoadError {
    #[error("failed to read GLB {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid GLB {path}: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("invalid GLB JSON {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to decode GLB image {path}: {source}")]
    ImageDecode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
}

struct GlbChunks {
    version: u32,
    json_len: usize,
    bin_len: usize,
    json: Value,
    bin: Vec<u8>,
}

#[derive(Debug, Clone)]
struct AccessorInfo {
    buffer_view: usize,
    byte_offset: usize,
    component_type: u32,
    count: usize,
    item_type: String,
}

#[derive(Debug, Clone, Copy)]
struct BufferViewInfo {
    byte_offset: usize,
    byte_stride: Option<usize>,
}

pub fn load_glb_metadata(path: impl AsRef<Path>) -> Result<GlbMetadata, GlbLoadError> {
    let path = path.as_ref();
    let bytes = read_file(path)?;
    parse_glb_metadata(path, &bytes)
}

pub fn load_glb_mesh_data(path: impl AsRef<Path>) -> Result<GlbMeshData, GlbLoadError> {
    let path = path.as_ref();
    let bytes = read_file(path)?;
    parse_glb_mesh_data(path, &bytes)
}

/// Load nodes, skinning data, and animation clips from a GLB that may not
/// contain renderable mesh primitives. Animation-only humanoid exports use
/// this shape after conversion to glTF.
pub fn load_glb_animation_data(path: impl AsRef<Path>) -> Result<GlbMeshData, GlbLoadError> {
    let path = path.as_ref();
    let bytes = read_file(path)?;
    parse_glb_animation_data(path, &bytes)
}

/// Load GLB metadata from an in-memory byte buffer.
pub fn load_glb_metadata_from_bytes(
    path: &Path,
    bytes: &[u8],
) -> Result<GlbMetadata, GlbLoadError> {
    parse_glb_metadata(path, bytes)
}

/// Load GLB mesh data from an in-memory byte buffer.
pub fn load_glb_mesh_data_from_bytes(
    path: &Path,
    bytes: &[u8],
) -> Result<GlbMeshData, GlbLoadError> {
    parse_glb_mesh_data(path, bytes)
}

/// In-memory counterpart to [`load_glb_animation_data`].
pub fn load_glb_animation_data_from_bytes(
    path: &Path,
    bytes: &[u8],
) -> Result<GlbMeshData, GlbLoadError> {
    parse_glb_animation_data(path, bytes)
}

pub fn parse_glb_metadata(path: &Path, bytes: &[u8]) -> Result<GlbMetadata, GlbLoadError> {
    let chunks = parse_gltf_or_glb_chunks(path, bytes)?;
    let skin_count = array_len(chunks.json.get("skins"));
    Ok(GlbMetadata {
        path: path.to_path_buf(),
        version: chunks.version,
        json_len: chunks.json_len,
        bin_len: chunks.bin_len,
        scene_count: array_len(chunks.json.get("scenes")),
        node_count: array_len(chunks.json.get("nodes")),
        mesh_count: array_len(chunks.json.get("meshes")),
        material_count: array_len(chunks.json.get("materials")),
        skin_count,
        animation_count: array_len(chunks.json.get("animations")),
        node_names: string_names(chunks.json.get("nodes")),
        joint_names: joint_names(&chunks.json),
        animation_names: string_names(chunks.json.get("animations")),
        has_skin: skin_count > 0,
    })
}

/// Return the source glTF JSON for in-crate inspectors that consume optional
/// metadata such as VRM humanoid declarations. Rendering continues to use the
/// typed mesh representation, so unknown extensions remain non-authoritative.
pub(crate) fn parse_gltf_json_value(path: &Path, bytes: &[u8]) -> Result<Value, GlbLoadError> {
    Ok(parse_gltf_or_glb_chunks(path, bytes)?.json)
}

pub fn parse_glb_mesh_data(path: &Path, bytes: &[u8]) -> Result<GlbMeshData, GlbLoadError> {
    parse_glb_data(path, bytes, true)
}

/// Parse an animation-only GLB while retaining the same data representation
/// used by model clips. Geometry arrays remain empty when the source has no
/// `meshes` entry.
pub fn parse_glb_animation_data(path: &Path, bytes: &[u8]) -> Result<GlbMeshData, GlbLoadError> {
    parse_glb_data(path, bytes, false)
}

fn parse_glb_data(
    path: &Path,
    bytes: &[u8],
    require_renderable_mesh: bool,
) -> Result<GlbMeshData, GlbLoadError> {
    let chunks = parse_gltf_or_glb_chunks(path, bytes)?;
    let draco_import = chunks
        .json
        .get("extensionsRequired")
        .and_then(Value::as_array)
        .is_some_and(|extensions| {
            extensions
                .iter()
                .any(|extension| extension.as_str() == Some("KHR_draco_mesh_compression"))
        })
        .then(|| {
            draco_gltf::import_slice(bytes, path.parent()).map_err(|error| {
                invalid_err(
                    path,
                    &format!("failed to read Draco glTF geometry: {error}"),
                )
            })
        })
        .transpose()?;
    let textures = read_textures(&chunks, path)?;
    let materials = read_materials(&chunks);
    let nodes = read_nodes(&chunks);
    let skin = read_skin(&chunks, path, &nodes)?;
    let animations = read_animations(&chunks, path)?;
    let mut positions = Vec::<[f32; 3]>::new();
    let mut normals = Vec::<Option<[f32; 3]>>::new();
    let mut texcoords = Vec::<Option<[f32; 2]>>::new();
    let mut colors = Vec::<Option<[f32; 4]>>::new();
    let mut joints = Vec::<Option<[u16; 4]>>::new();
    let mut weights = Vec::<Option<[f32; 4]>>::new();
    let mut indices = Vec::<u32>::new();
    let mut triangles = Vec::<GlbTriangle>::new();

    let meshes = chunks
        .json
        .get("meshes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mesh_names = meshes
        .iter()
        .map(|mesh| {
            mesh.get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect::<Vec<_>>();
    let mesh_node_lookup = nodes
        .iter()
        .filter_map(|node| node.mesh.map(|mesh| (mesh, node.index)))
        .collect::<std::collections::HashMap<_, _>>();
    for (mesh_index, mesh) in meshes.iter().enumerate() {
        let Some(primitives) = mesh.get("primitives").and_then(Value::as_array) else {
            continue;
        };
        let mesh_node = mesh_node_lookup.get(&mesh_index).copied();
        for (primitive_index, primitive) in primitives.iter().enumerate() {
            let mode = primitive.get("mode").and_then(Value::as_u64).unwrap_or(4);
            if mode != 4 {
                continue;
            }
            if primitive
                .get("extensions")
                .and_then(|extensions| extensions.get("KHR_draco_mesh_compression"))
                .is_some()
            {
                let import = draco_import.as_ref().ok_or_else(|| {
                    invalid_err(path, "Draco primitive found without a Draco decoder")
                })?;
                let geometry = import
                    .read_primitive(DracoPrimitiveIndex::new(
                        DracoMeshIndex(mesh_index),
                        primitive_index,
                    ))
                    .map_err(|error| {
                        invalid_err(
                            path,
                            &format!(
                                "failed to decode Draco mesh {mesh_index} primitive {primitive_index}: {error}"
                            ),
                        )
                    })?;
                append_packed_primitive(
                    path,
                    &geometry,
                    primitive,
                    mesh_index,
                    mesh_node,
                    &mut positions,
                    &mut normals,
                    &mut texcoords,
                    &mut colors,
                    &mut joints,
                    &mut weights,
                    &mut indices,
                    &mut triangles,
                )?;
                continue;
            }
            let Some(position_index) = primitive
                .get("attributes")
                .and_then(|attrs| attrs.get("POSITION"))
                .and_then(Value::as_u64)
                .map(|value| value as usize)
            else {
                continue;
            };
            let base = positions.len() as u32;
            let primitive_positions = read_positions(&chunks, position_index, path)?;
            let primitive_normals = primitive
                .get("attributes")
                .and_then(|attrs| attrs.get("NORMAL"))
                .and_then(Value::as_u64)
                .map(|index| read_normals(&chunks, index as usize, path))
                .transpose()?
                .unwrap_or_default();
            let primitive_texcoords = primitive
                .get("attributes")
                .and_then(|attrs| attrs.get("TEXCOORD_0"))
                .and_then(Value::as_u64)
                .map(|index| read_texcoords(&chunks, index as usize, path))
                .transpose()?
                .unwrap_or_default();
            let primitive_colors = primitive
                .get("attributes")
                .and_then(|attrs| attrs.get("COLOR_0"))
                .and_then(Value::as_u64)
                .map(|index| read_colors(&chunks, index as usize, path))
                .transpose()?
                .unwrap_or_default();
            let primitive_joints = primitive
                .get("attributes")
                .and_then(|attrs| attrs.get("JOINTS_0"))
                .and_then(Value::as_u64)
                .map(|index| read_joints(&chunks, index as usize, path))
                .transpose()?
                .unwrap_or_default();
            let primitive_weights = primitive
                .get("attributes")
                .and_then(|attrs| attrs.get("WEIGHTS_0"))
                .and_then(Value::as_u64)
                .map(|index| read_weights(&chunks, index as usize, path))
                .transpose()?
                .unwrap_or_default();
            let primitive_len = primitive_positions.len();
            positions.extend(primitive_positions);
            for idx in 0..primitive_len {
                normals.push(primitive_normals.get(idx).copied());
                texcoords.push(primitive_texcoords.get(idx).copied());
                colors.push(primitive_colors.get(idx).copied());
                joints.push(primitive_joints.get(idx).copied());
                weights.push(primitive_weights.get(idx).copied());
            }

            let primitive_indices =
                if let Some(index_index) = primitive.get("indices").and_then(Value::as_u64) {
                    read_indices(&chunks, index_index as usize, path)?
                        .into_iter()
                        .map(|index| base + index)
                        .collect::<Vec<_>>()
                } else {
                    (0..primitive_len as u32)
                        .map(|index| base + index)
                        .collect::<Vec<_>>()
                };
            let material = primitive
                .get("material")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            for chunk in primitive_indices.chunks(3) {
                if let [a, b, c] = *chunk {
                    indices.extend([a, b, c]);
                    triangles.push(GlbTriangle {
                        indices: [a, b, c],
                        material,
                        mesh: Some(mesh_index),
                        mesh_node,
                    });
                }
            }
        }
    }

    if require_renderable_mesh && (positions.is_empty() || triangles.is_empty()) {
        return invalid(path, "no triangle POSITION/index data found");
    }
    let (bounds_min, bounds_max) = if positions.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        compute_bounds(&positions)
    };
    Ok(GlbMeshData {
        path: path.to_path_buf(),
        positions,
        normals,
        texcoords,
        colors,
        joints,
        weights,
        indices,
        triangles,
        materials,
        textures,
        mesh_names,
        nodes,
        skin,
        animations,
        bounds_min,
        bounds_max,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_packed_primitive(
    path: &Path,
    geometry: &PackedGeometry,
    primitive: &Value,
    mesh_index: usize,
    mesh_node: Option<usize>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<Option<[f32; 3]>>,
    texcoords: &mut Vec<Option<[f32; 2]>>,
    colors: &mut Vec<Option<[f32; 4]>>,
    joints: &mut Vec<Option<[u16; 4]>>,
    weights: &mut Vec<Option<[f32; 4]>>,
    indices: &mut Vec<u32>,
    triangles: &mut Vec<GlbTriangle>,
) -> Result<(), GlbLoadError> {
    if geometry.mode() != DracoPrimitiveMode::Triangles {
        return Ok(());
    }
    let position_attribute = packed_attribute(geometry, "POSITION")
        .ok_or_else(|| invalid_err(path, "decoded Draco primitive is missing POSITION"))?;
    let primitive_positions = packed_f32_rows::<3>(path, position_attribute)?;
    let primitive_normals = packed_attribute(geometry, "NORMAL")
        .map(|attribute| packed_f32_rows::<3>(path, attribute))
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .map(normalize3)
        .collect::<Vec<_>>();
    let primitive_texcoords = packed_attribute(geometry, "TEXCOORD_0")
        .map(|attribute| packed_f32_rows::<2>(path, attribute))
        .transpose()?
        .unwrap_or_default();
    let primitive_colors = packed_attribute(geometry, "COLOR_0")
        .map(|attribute| packed_colors(path, attribute))
        .transpose()?
        .unwrap_or_default();
    let primitive_joints = packed_attribute(geometry, "JOINTS_0")
        .map(|attribute| packed_u16_rows::<4>(path, attribute))
        .transpose()?
        .unwrap_or_default();
    let primitive_weights = packed_attribute(geometry, "WEIGHTS_0")
        .map(|attribute| packed_f32_rows::<4>(path, attribute))
        .transpose()?
        .unwrap_or_default();

    let base = positions.len() as u32;
    let primitive_len = primitive_positions.len();
    positions.extend(primitive_positions);
    for idx in 0..primitive_len {
        normals.push(primitive_normals.get(idx).copied());
        texcoords.push(primitive_texcoords.get(idx).copied());
        colors.push(primitive_colors.get(idx).copied());
        joints.push(primitive_joints.get(idx).copied());
        weights.push(primitive_weights.get(idx).copied());
    }

    let primitive_indices = geometry
        .indices()
        .map(|packed| packed_indices(path, packed))
        .transpose()?
        .unwrap_or_else(|| (0..primitive_len as u32).collect())
        .into_iter()
        .map(|index| base + index)
        .collect::<Vec<_>>();
    let material = primitive
        .get("material")
        .and_then(Value::as_u64)
        .map(|value| value as usize);
    for chunk in primitive_indices.chunks(3) {
        if let [a, b, c] = *chunk {
            indices.extend([a, b, c]);
            triangles.push(GlbTriangle {
                indices: [a, b, c],
                material,
                mesh: Some(mesh_index),
                mesh_node,
            });
        }
    }
    Ok(())
}

fn packed_attribute<'a>(
    geometry: &'a PackedGeometry,
    semantic: &str,
) -> Option<&'a PackedAttribute> {
    geometry
        .attributes()
        .iter()
        .find(|attribute| attribute.semantic() == semantic)
}

fn packed_f32_rows<const N: usize>(
    path: &Path,
    attribute: &PackedAttribute,
) -> Result<Vec<[f32; N]>, GlbLoadError> {
    if usize::from(attribute.components()) != N {
        return invalid(
            path,
            &format!(
                "decoded Draco {} has {} components; expected {N}",
                attribute.semantic(),
                attribute.components()
            ),
        );
    }
    let stride = attribute.component_type().byte_width() * N;
    let mut rows = Vec::with_capacity(attribute.count());
    for row in 0..attribute.count() {
        let mut values = [0.0; N];
        for (component, value) in values.iter_mut().enumerate() {
            *value = packed_component_f32(
                path,
                attribute,
                row * stride + component * attribute.component_type().byte_width(),
            )?;
        }
        rows.push(values);
    }
    Ok(rows)
}

fn packed_u16_rows<const N: usize>(
    path: &Path,
    attribute: &PackedAttribute,
) -> Result<Vec<[u16; N]>, GlbLoadError> {
    if usize::from(attribute.components()) != N {
        return invalid(
            path,
            &format!(
                "decoded Draco {} has {} components; expected {N}",
                attribute.semantic(),
                attribute.components()
            ),
        );
    }
    let stride = attribute.component_type().byte_width() * N;
    let mut rows = Vec::with_capacity(attribute.count());
    for row in 0..attribute.count() {
        let mut values = [0; N];
        for (component, value) in values.iter_mut().enumerate() {
            *value = packed_component_u64(
                path,
                attribute.component_type(),
                attribute.bytes(),
                row * stride + component * attribute.component_type().byte_width(),
            )?
            .min(u16::MAX as u64) as u16;
        }
        rows.push(values);
    }
    Ok(rows)
}

fn packed_colors(path: &Path, attribute: &PackedAttribute) -> Result<Vec<[f32; 4]>, GlbLoadError> {
    let components = usize::from(attribute.components());
    if components != 3 && components != 4 {
        return invalid(path, "decoded Draco COLOR_0 must be VEC3 or VEC4");
    }
    let stride = attribute.component_type().byte_width() * components;
    let mut rows = Vec::with_capacity(attribute.count());
    for row in 0..attribute.count() {
        let mut values = [1.0; 4];
        for (component, value) in values.iter_mut().enumerate().take(components) {
            *value = packed_component_f32(
                path,
                attribute,
                row * stride + component * attribute.component_type().byte_width(),
            )?;
        }
        rows.push(values);
    }
    Ok(rows)
}

fn packed_indices(
    path: &Path,
    indices: &draco_gltf::PackedIndices,
) -> Result<Vec<u32>, GlbLoadError> {
    let width = indices.component_type().byte_width();
    let mut out = Vec::with_capacity(indices.count());
    for index in 0..indices.count() {
        out.push(
            packed_component_u64(
                path,
                indices.component_type(),
                indices.bytes(),
                index * width,
            )?
            .min(u32::MAX as u64) as u32,
        );
    }
    Ok(out)
}

fn packed_component_f32(
    path: &Path,
    attribute: &PackedAttribute,
    offset: usize,
) -> Result<f32, GlbLoadError> {
    let bytes = attribute.bytes();
    let component_type = attribute.component_type();
    let normalized = attribute.normalized();
    let value = match component_type {
        DracoComponentType::F32 => read_f32(bytes, offset)
            .ok_or_else(|| invalid_err(path, "truncated packed f32 Draco attribute"))?,
        DracoComponentType::U8 => {
            let value = *bytes
                .get(offset)
                .ok_or_else(|| invalid_err(path, "truncated packed u8 Draco attribute"))?
                as f32;
            if normalized { value / 255.0 } else { value }
        }
        DracoComponentType::I8 => {
            let value = *bytes
                .get(offset)
                .ok_or_else(|| invalid_err(path, "truncated packed i8 Draco attribute"))?
                as i8 as f32;
            if normalized {
                (value / 127.0).max(-1.0)
            } else {
                value
            }
        }
        DracoComponentType::U16 => {
            let value = read_u16(bytes, offset)
                .ok_or_else(|| invalid_err(path, "truncated packed u16 Draco attribute"))?
                as f32;
            if normalized { value / 65535.0 } else { value }
        }
        DracoComponentType::I16 => {
            let value = read_i16(bytes, offset)
                .ok_or_else(|| invalid_err(path, "truncated packed i16 Draco attribute"))?
                as f32;
            if normalized {
                (value / 32767.0).max(-1.0)
            } else {
                value
            }
        }
        DracoComponentType::U32 => read_u32(bytes, offset)
            .ok_or_else(|| invalid_err(path, "truncated packed u32 Draco attribute"))?
            as f32,
        other => {
            return invalid(
                path,
                &format!("unsupported decoded Draco component type {other:?}"),
            );
        }
    };
    Ok(value)
}

fn packed_component_u64(
    path: &Path,
    component_type: DracoComponentType,
    bytes: &[u8],
    offset: usize,
) -> Result<u64, GlbLoadError> {
    let value = match component_type {
        DracoComponentType::U8 => *bytes
            .get(offset)
            .ok_or_else(|| invalid_err(path, "truncated packed u8 Draco integer"))?
            as u64,
        DracoComponentType::U16 => read_u16(bytes, offset)
            .ok_or_else(|| invalid_err(path, "truncated packed u16 Draco integer"))?
            as u64,
        DracoComponentType::U32 => read_u32(bytes, offset)
            .ok_or_else(|| invalid_err(path, "truncated packed u32 Draco integer"))?
            as u64,
        other => {
            return invalid(
                path,
                &format!("unsupported decoded Draco integer type {other:?}"),
            );
        }
    };
    Ok(value)
}

fn read_file(path: &Path) -> Result<Vec<u8>, GlbLoadError> {
    std::fs::read(path).map_err(|source| GlbLoadError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn parse_gltf_or_glb_chunks(path: &Path, bytes: &[u8]) -> Result<GlbChunks, GlbLoadError> {
    if bytes.starts_with(GLB_MAGIC) {
        return parse_glb_chunks(path, bytes);
    }
    parse_gltf_json_chunks(path, bytes)
}

fn parse_gltf_json_chunks(path: &Path, bytes: &[u8]) -> Result<GlbChunks, GlbLoadError> {
    let json: Value = serde_json::from_slice(bytes).map_err(|source| GlbLoadError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    let version = json
        .get("asset")
        .and_then(|asset| asset.get("version"))
        .and_then(Value::as_str)
        .and_then(|version| version.split('.').next())
        .and_then(|major| major.parse::<u32>().ok())
        .unwrap_or(2);
    if version != 2 {
        return invalid(path, &format!("expected glTF version 2, got {version}"));
    }
    let buffers = json
        .get("buffers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if buffers.len() > 1 {
        return invalid(
            path,
            "text glTF currently supports one binary buffer; use GLB for multi-buffer assets",
        );
    }
    let bin = if let Some(buffer) = buffers.first() {
        let uri = buffer.get("uri").and_then(Value::as_str).ok_or_else(|| {
            invalid_err(
                path,
                "text glTF buffer is missing uri; use a data URI, external .bin, or GLB",
            )
        })?;
        read_gltf_uri(path, uri)?
    } else {
        Vec::new()
    };
    Ok(GlbChunks {
        version,
        json_len: bytes.len(),
        bin_len: bin.len(),
        json,
        bin,
    })
}

fn read_gltf_uri(path: &Path, uri: &str) -> Result<Vec<u8>, GlbLoadError> {
    if let Some((header, payload)) = uri.split_once(',')
        && header.starts_with("data:")
    {
        if header.ends_with(";base64") {
            return base64::engine::general_purpose::STANDARD
                .decode(payload)
                .map_err(|err| invalid_err(path, &format!("invalid base64 data URI: {err}")));
        }
        return Ok(payload.as_bytes().to_vec());
    }
    let asset_path = path.parent().unwrap_or_else(|| Path::new(".")).join(uri);
    read_file(&asset_path)
}

fn parse_glb_chunks(path: &Path, bytes: &[u8]) -> Result<GlbChunks, GlbLoadError> {
    if bytes.len() < 12 {
        return invalid(path, "file is shorter than the GLB header");
    }
    if &bytes[0..4] != GLB_MAGIC {
        return invalid(path, "magic is not glTF");
    }
    let version = read_u32(bytes, 4).ok_or_else(|| invalid_err(path, "missing version"))?;
    if version != 2 {
        return invalid(path, &format!("expected GLB version 2, got {version}"));
    }
    let declared_len =
        read_u32(bytes, 8).ok_or_else(|| invalid_err(path, "missing length"))? as usize;
    if declared_len > bytes.len() {
        return invalid(
            path,
            &format!(
                "declared length {declared_len} exceeds file length {}",
                bytes.len()
            ),
        );
    }

    let mut offset = 12usize;
    let mut json_bytes: Option<&[u8]> = None;
    let mut bin = Vec::<u8>::new();
    while offset + 8 <= declared_len {
        let chunk_len = read_u32(bytes, offset)
            .ok_or_else(|| invalid_err(path, "missing chunk length"))?
            as usize;
        let chunk_type =
            read_u32(bytes, offset + 4).ok_or_else(|| invalid_err(path, "missing chunk type"))?;
        offset += 8;
        let end = offset.saturating_add(chunk_len);
        if end > declared_len || end > bytes.len() {
            return invalid(path, "chunk length exceeds file length");
        }
        match chunk_type {
            GLB_JSON_CHUNK => json_bytes = Some(&bytes[offset..end]),
            GLB_BIN_CHUNK => bin.extend_from_slice(&bytes[offset..end]),
            _ => {}
        }
        offset = end;
    }

    let Some(json_bytes) = json_bytes else {
        return invalid(path, "missing JSON chunk");
    };
    let json_text = std::str::from_utf8(json_bytes)
        .map_err(|err| invalid_err(path, &format!("JSON chunk is not UTF-8: {err}")))?
        .trim_end_matches(['\0', ' ', '\n', '\r', '\t']);
    let json: Value = serde_json::from_str(json_text).map_err(|source| GlbLoadError::Json {
        path: path.to_path_buf(),
        source,
    })?;

    Ok(GlbChunks {
        version,
        json_len: json_bytes.len(),
        bin_len: bin.len(),
        json,
        bin,
    })
}

fn read_textures(
    chunks: &GlbChunks,
    path: &Path,
) -> Result<Vec<Option<GlbTextureData>>, GlbLoadError> {
    let mut images = Vec::<Option<GlbTextureData>>::new();
    if let Some(image_nodes) = chunks.json.get("images").and_then(Value::as_array) {
        for image_node in image_nodes {
            images.push(read_image_node(chunks, path, image_node)?);
        }
    }

    let mut textures = Vec::<Option<GlbTextureData>>::new();
    if let Some(texture_nodes) = chunks.json.get("textures").and_then(Value::as_array) {
        for texture_node in texture_nodes {
            let source = texture_source_index(texture_node);
            textures.push(source.and_then(|index| images.get(index).cloned().flatten()));
        }
    }
    Ok(textures)
}

fn texture_source_index(texture_node: &Value) -> Option<usize> {
    json_usize(texture_node, "source").or_else(|| {
        texture_node
            .get("extensions")
            .and_then(|extensions| {
                extensions
                    .get("EXT_texture_webp")
                    .or_else(|| extensions.get("KHR_texture_basisu"))
            })
            .and_then(|extension| json_usize(extension, "source"))
    })
}

fn read_image_node(
    chunks: &GlbChunks,
    path: &Path,
    image_node: &Value,
) -> Result<Option<GlbTextureData>, GlbLoadError> {
    let bytes = if let Some(buffer_view) = json_usize(image_node, "bufferView") {
        let view = parse_buffer_view(chunks, buffer_view, path)?;
        let byte_length = chunks
            .json
            .get("bufferViews")
            .and_then(Value::as_array)
            .and_then(|items| items.get(buffer_view))
            .and_then(|view| json_usize(view, "byteLength"))
            .ok_or_else(|| invalid_err(path, "image bufferView missing byteLength"))?;
        ensure_range(path, view.byte_offset, byte_length, chunks.bin.len())?;
        chunks.bin[view.byte_offset..view.byte_offset + byte_length].to_vec()
    } else if let Some(uri) = image_node.get("uri").and_then(Value::as_str) {
        read_gltf_uri(path, uri)?
    } else {
        return Ok(None);
    };
    let decoded = image::load_from_memory(&bytes)
        .map_err(|source| GlbLoadError::ImageDecode {
            path: path.to_path_buf(),
            source,
        })?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    Ok(Some(GlbTextureData {
        width,
        height,
        rgba: Arc::new(decoded.into_raw()),
    }))
}

fn read_materials(chunks: &GlbChunks) -> Vec<GlbMaterialData> {
    chunks
        .json
        .get("materials")
        .and_then(Value::as_array)
        .map(|materials| {
            materials
                .iter()
                .map(|material| {
                    let pbr = material.get("pbrMetallicRoughness");
                    let mut out = GlbMaterialData {
                        name: material
                            .get("name")
                            .and_then(Value::as_str)
                            .map(ToString::to_string),
                        ..Default::default()
                    };
                    if let Some(factor) = pbr
                        .and_then(|pbr| pbr.get("baseColorFactor"))
                        .and_then(Value::as_array)
                    {
                        for axis in 0..4 {
                            out.base_color_factor[axis] =
                                factor.get(axis).and_then(Value::as_f64).unwrap_or(1.0) as f32;
                        }
                    }
                    out.base_color_texture = pbr
                        .and_then(|pbr| pbr.get("baseColorTexture"))
                        .and_then(|tex| json_usize(tex, "index"));
                    out.metallic_roughness_texture = pbr
                        .and_then(|pbr| pbr.get("metallicRoughnessTexture"))
                        .and_then(|tex| json_usize(tex, "index"));
                    out.metallic_factor = pbr
                        .and_then(|pbr| pbr.get("metallicFactor"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32;
                    out.roughness_factor = pbr
                        .and_then(|pbr| pbr.get("roughnessFactor"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32;
                    out.normal_texture = material
                        .get("normalTexture")
                        .and_then(|tex| json_usize(tex, "index"));
                    out.normal_scale = material
                        .get("normalTexture")
                        .and_then(|tex| tex.get("scale"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32;
                    if let Some(spec_gloss) = material.get("extensions").and_then(|extensions| {
                        extensions.get("KHR_materials_pbrSpecularGlossiness")
                    }) {
                        out.specular_glossiness = true;
                        if let Some(factor) =
                            spec_gloss.get("diffuseFactor").and_then(Value::as_array)
                        {
                            for axis in 0..4 {
                                out.base_color_factor[axis] =
                                    factor.get(axis).and_then(Value::as_f64).unwrap_or(1.0) as f32;
                            }
                        }
                        if let Some(texture) = spec_gloss
                            .get("diffuseTexture")
                            .and_then(|tex| json_usize(tex, "index"))
                        {
                            out.base_color_texture = Some(texture);
                        }
                    }
                    out.emissive_texture = material
                        .get("emissiveTexture")
                        .and_then(|tex| json_usize(tex, "index"));
                    let emissive_factor = material
                        .get("emissiveFactor")
                        .and_then(Value::as_array)
                        .map(|factor| {
                            [
                                factor.first().and_then(Value::as_f64).unwrap_or(1.0) as f32,
                                factor.get(1).and_then(Value::as_f64).unwrap_or(1.0) as f32,
                                factor.get(2).and_then(Value::as_f64).unwrap_or(1.0) as f32,
                            ]
                        })
                        .unwrap_or([0.0, 0.0, 0.0]);
                    out.emissive_factor = emissive_factor;
                    out.emissive_strength = material
                        .get("extensions")
                        .and_then(|extensions| extensions.get("KHR_materials_emissive_strength"))
                        .and_then(|extension| extension.get("emissiveStrength"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.0) as f32;
                    if let Some(specular) = material
                        .get("extensions")
                        .and_then(|extensions| extensions.get("KHR_materials_specular"))
                    {
                        out.specular_factor = specular
                            .get("specularFactor")
                            .and_then(Value::as_f64)
                            .unwrap_or(1.0) as f32;
                        if let Some(color) = specular
                            .get("specularColorFactor")
                            .and_then(Value::as_array)
                        {
                            for axis in 0..3 {
                                out.specular_color_factor[axis] =
                                    color.get(axis).and_then(Value::as_f64).unwrap_or(1.0) as f32;
                            }
                        }
                    }
                    if let Some(transmission) = material
                        .get("extensions")
                        .and_then(|extensions| extensions.get("KHR_materials_transmission"))
                    {
                        out.transmission_factor = transmission
                            .get("transmissionFactor")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0)
                            as f32;
                    }
                    out.ior = material
                        .get("extensions")
                        .and_then(|extensions| extensions.get("KHR_materials_ior"))
                        .and_then(|extension| extension.get("ior"))
                        .and_then(Value::as_f64)
                        .unwrap_or(1.5) as f32;
                    if let Some(volume) = material
                        .get("extensions")
                        .and_then(|extensions| extensions.get("KHR_materials_volume"))
                    {
                        out.thickness_factor = volume
                            .get("thicknessFactor")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0) as f32;
                        if let Some(color) =
                            volume.get("attenuationColor").and_then(Value::as_array)
                        {
                            for axis in 0..3 {
                                out.attenuation_color[axis] =
                                    color.get(axis).and_then(Value::as_f64).unwrap_or(1.0) as f32;
                            }
                        }
                        out.attenuation_distance = volume
                            .get("attenuationDistance")
                            .and_then(Value::as_f64)
                            .filter(|distance| distance.is_finite() && *distance > 0.0)
                            .unwrap_or(1_000_000.0)
                            as f32;
                    }
                    out.alpha_mode = match material
                        .get("alphaMode")
                        .and_then(Value::as_str)
                        .unwrap_or("OPAQUE")
                    {
                        "MASK" => GlbAlphaMode::Mask,
                        "BLEND" => GlbAlphaMode::Blend,
                        _ => GlbAlphaMode::Opaque,
                    };
                    out.alpha_cutoff = material
                        .get("alphaCutoff")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.5) as f32;
                    out.double_sided = material
                        .get("doubleSided")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    out.unlit = material
                        .get("extensions")
                        .and_then(|extensions| extensions.get("KHR_materials_unlit"))
                        .is_some();
                    if out.base_color_texture.is_none() && out.emissive_texture.is_some() {
                        out.base_color_texture = out.emissive_texture;
                    }
                    let has_visible_texture =
                        out.base_color_texture.is_some() || out.emissive_texture.is_some();
                    let factor_is_black = out.base_color_factor[0].abs() <= 0.001
                        && out.base_color_factor[1].abs() <= 0.001
                        && out.base_color_factor[2].abs() <= 0.001;
                    if has_visible_texture && factor_is_black {
                        // Sketchfab-style toon/anime exports often keep the visible
                        // texture in both baseColor/emissive slots while leaving
                        // baseColorFactor black. Engines like Godot still show the
                        // texture; multiplying by black would turn the whole model
                        // into a silhouette.
                        out.base_color_factor[0] = emissive_factor[0].max(1.0).clamp(0.0, 1.0);
                        out.base_color_factor[1] = emissive_factor[1].max(1.0).clamp(0.0, 1.0);
                        out.base_color_factor[2] = emissive_factor[2].max(1.0).clamp(0.0, 1.0);
                    }
                    out
                })
                .collect()
        })
        .unwrap_or_default()
}

fn read_positions(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 3]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.component_type != 5126 || accessor.item_type != "VEC3" {
        return invalid(path, "POSITION accessor must be float VEC3");
    }
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view.byte_stride.unwrap_or(12).max(12);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, 12, chunks.bin.len())?;
        out.push([
            read_f32(&chunks.bin, offset).unwrap_or(0.0),
            read_f32(&chunks.bin, offset + 4).unwrap_or(0.0),
            read_f32(&chunks.bin, offset + 8).unwrap_or(0.0),
        ]);
    }
    Ok(out)
}

fn read_normals(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 3]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.component_type != 5126 || accessor.item_type != "VEC3" {
        return Ok(Vec::new());
    }
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view.byte_stride.unwrap_or(12).max(12);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, 12, chunks.bin.len())?;
        let normal = normalize3([
            read_f32(&chunks.bin, offset).unwrap_or(0.0),
            read_f32(&chunks.bin, offset + 4).unwrap_or(0.0),
            read_f32(&chunks.bin, offset + 8).unwrap_or(1.0),
        ]);
        out.push(normal);
    }
    Ok(out)
}

fn read_texcoords(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 2]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.component_type != 5126 || accessor.item_type != "VEC2" {
        return Ok(Vec::new());
    }
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view.byte_stride.unwrap_or(8).max(8);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, 8, chunks.bin.len())?;
        out.push([
            read_f32(&chunks.bin, offset).unwrap_or(0.0),
            read_f32(&chunks.bin, offset + 4).unwrap_or(0.0),
        ]);
    }
    Ok(out)
}

fn read_colors(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 4]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    let component_count = match accessor.item_type.as_str() {
        "VEC3" => 3,
        "VEC4" => 4,
        _ => return Ok(Vec::new()),
    };
    let component_size = match accessor.component_type {
        5121 => 1,
        5123 => 2,
        5126 => 4,
        _ => return Ok(Vec::new()),
    };
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let tight = component_size * component_count;
    let stride = view.byte_stride.unwrap_or(tight).max(tight);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, tight, chunks.bin.len())?;
        let mut color = [1.0f32; 4];
        for (slot, component) in color.iter_mut().enumerate().take(component_count) {
            let component_offset = offset + slot * component_size;
            *component = match accessor.component_type {
                5121 => chunks.bin[component_offset] as f32 / 255.0,
                5123 => read_u16(&chunks.bin, component_offset).unwrap_or(0) as f32 / 65535.0,
                5126 => read_f32(&chunks.bin, component_offset).unwrap_or(1.0),
                _ => unreachable!(),
            };
        }
        out.push(color);
    }
    Ok(out)
}

fn read_indices(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<u32>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.item_type != "SCALAR" {
        return invalid(path, "index accessor must be SCALAR");
    }
    let component_size = match accessor.component_type {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        other => {
            return invalid(
                path,
                &format!("unsupported index component type {other}; expected u8/u16/u32"),
            );
        }
    };
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view
        .byte_stride
        .unwrap_or(component_size)
        .max(component_size);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, component_size, chunks.bin.len())?;
        let index = match accessor.component_type {
            5121 => chunks.bin[offset] as u32,
            5123 => read_u16(&chunks.bin, offset).unwrap_or(0) as u32,
            5125 => read_u32(&chunks.bin, offset).unwrap_or(0),
            _ => unreachable!(),
        };
        out.push(index);
    }
    Ok(out)
}

fn read_nodes(chunks: &GlbChunks) -> Vec<GlbNodeData> {
    let Some(nodes) = chunks.json.get("nodes").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| GlbNodeData {
            index,
            name: node
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            parent: None,
            children: node
                .get("children")
                .and_then(Value::as_array)
                .map(|children| {
                    children
                        .iter()
                        .filter_map(Value::as_u64)
                        .map(|index| index as usize)
                        .collect()
                })
                .unwrap_or_default(),
            mesh: json_usize(node, "mesh"),
            skin: json_usize(node, "skin"),
            translation: json_f32_array_3(node.get("translation")).unwrap_or([0.0, 0.0, 0.0]),
            rotation: json_f32_array_4(node.get("rotation")).unwrap_or([0.0, 0.0, 0.0, 1.0]),
            scale: json_f32_array_3(node.get("scale")).unwrap_or([1.0, 1.0, 1.0]),
            matrix: json_f32_array_16(node.get("matrix")),
        })
        .collect::<Vec<_>>();

    let child_links = out
        .iter()
        .enumerate()
        .flat_map(|(parent, node)| {
            node.children
                .iter()
                .copied()
                .map(move |child| (parent, child))
        })
        .collect::<Vec<_>>();
    for (parent, child) in child_links {
        if let Some(node) = out.get_mut(child) {
            node.parent = Some(parent);
        }
    }
    out
}

fn read_skin(
    chunks: &GlbChunks,
    path: &Path,
    nodes: &[GlbNodeData],
) -> Result<Option<GlbSkinData>, GlbLoadError> {
    let Some(skins) = chunks.json.get("skins").and_then(Value::as_array) else {
        return Ok(None);
    };
    let Some(skin) = skins.first() else {
        return Ok(None);
    };
    let Some(joint_indices) = skin.get("joints").and_then(Value::as_array) else {
        return invalid(path, "skin missing joints array");
    };
    let inverse_bind_matrices = skin
        .get("inverseBindMatrices")
        .and_then(Value::as_u64)
        .map(|index| read_mat4s(chunks, index as usize, path))
        .transpose()?
        .unwrap_or_default();

    let mut joints = Vec::with_capacity(joint_indices.len());
    for (joint_slot, joint) in joint_indices.iter().enumerate() {
        let Some(node_index) = joint.as_u64().map(|value| value as usize) else {
            return invalid(path, "skin joint entry must be a node index");
        };
        let inverse_bind_matrix = inverse_bind_matrices
            .get(joint_slot)
            .copied()
            .unwrap_or_else(identity_mat4);
        let name = nodes.get(node_index).and_then(|node| node.name.clone());
        joints.push(GlbSkinJointData {
            node_index,
            name,
            inverse_bind_matrix,
        });
    }

    Ok(Some(GlbSkinData {
        skeleton: json_usize(skin, "skeleton"),
        joints,
    }))
}

fn read_animations(chunks: &GlbChunks, path: &Path) -> Result<Vec<GlbAnimationData>, GlbLoadError> {
    let Some(animations) = chunks.json.get("animations").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(animations.len());
    for animation in animations {
        let samplers = animation
            .get("samplers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut channels_out = Vec::new();
        let mut duration = 0.0f32;
        for channel in animation
            .get("channels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(sampler_index) = json_usize(channel, "sampler") else {
                continue;
            };
            let Some(sampler) = samplers.get(sampler_index) else {
                continue;
            };
            let Some(node_index) = channel
                .get("target")
                .and_then(|target| json_usize(target, "node"))
            else {
                continue;
            };
            let property = match channel
                .get("target")
                .and_then(|target| target.get("path"))
                .and_then(Value::as_str)
            {
                Some("translation") => GlbAnimationProperty::Translation,
                Some("rotation") => GlbAnimationProperty::Rotation,
                Some("scale") => GlbAnimationProperty::Scale,
                _ => continue,
            };
            let Some(input) = json_usize(sampler, "input") else {
                continue;
            };
            let Some(output) = json_usize(sampler, "output") else {
                continue;
            };
            let interpolation = match sampler
                .get("interpolation")
                .and_then(Value::as_str)
                .unwrap_or("LINEAR")
            {
                "STEP" => GlbAnimationInterpolation::Step,
                "CUBICSPLINE" => GlbAnimationInterpolation::CubicSpline,
                _ => GlbAnimationInterpolation::Linear,
            };
            let times = read_float_scalars(chunks, input, path)?;
            if times.is_empty() {
                continue;
            }
            duration = duration.max(times.last().copied().unwrap_or(0.0));
            let cubic = interpolation == GlbAnimationInterpolation::CubicSpline;
            let values = match property {
                GlbAnimationProperty::Rotation => {
                    let values = read_float_vec4(chunks, output, path)?;
                    GlbAnimationValues::Quat(cubic_values(values, times.len(), cubic))
                }
                GlbAnimationProperty::Translation | GlbAnimationProperty::Scale => {
                    let values = read_positions(chunks, output, path)?;
                    GlbAnimationValues::Vec3(cubic_values(values, times.len(), cubic))
                }
            };
            channels_out.push(GlbAnimationChannelData {
                node_index,
                property,
                interpolation,
                times,
                values,
            });
        }
        out.push(GlbAnimationData {
            name: animation
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            duration,
            channels: channels_out,
        });
    }
    Ok(out)
}

fn cubic_values<T: Copy>(values: Vec<T>, key_count: usize, cubic: bool) -> Vec<T> {
    if !cubic || values.len() < key_count.saturating_mul(3) {
        return values;
    }
    (0..key_count)
        .filter_map(|index| values.get(index * 3 + 1).copied())
        .collect()
}

fn read_float_scalars(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<f32>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.component_type != 5126 || accessor.item_type != "SCALAR" {
        return invalid(path, "animation input accessor must be float SCALAR");
    }
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view.byte_stride.unwrap_or(4);
    let base = view.byte_offset + accessor.byte_offset;
    (0..accessor.count)
        .map(|index| {
            read_f32(&chunks.bin, base + index * stride)
                .ok_or_else(|| invalid_err(path, "truncated animation input accessor"))
        })
        .collect()
}

fn read_float_vec4(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 4]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.component_type != 5126 || accessor.item_type != "VEC4" {
        return invalid(path, "animation rotation accessor must be float VEC4");
    }
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view.byte_stride.unwrap_or(16);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for index in 0..accessor.count {
        let offset = base + index * stride;
        out.push([
            read_f32(&chunks.bin, offset)
                .ok_or_else(|| invalid_err(path, "truncated animation VEC4 accessor"))?,
            read_f32(&chunks.bin, offset + 4)
                .ok_or_else(|| invalid_err(path, "truncated animation VEC4 accessor"))?,
            read_f32(&chunks.bin, offset + 8)
                .ok_or_else(|| invalid_err(path, "truncated animation VEC4 accessor"))?,
            read_f32(&chunks.bin, offset + 12)
                .ok_or_else(|| invalid_err(path, "truncated animation VEC4 accessor"))?,
        ]);
    }
    Ok(out)
}

fn read_mat4s(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 16]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.component_type != 5126 || accessor.item_type != "MAT4" {
        return invalid(path, "MAT4 accessor must be float MAT4");
    }
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view.byte_stride.unwrap_or(64).max(64);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, 64, chunks.bin.len())?;
        let mut matrix = [0.0; 16];
        for (slot, value) in matrix.iter_mut().enumerate() {
            *value = read_f32(&chunks.bin, offset + slot * 4).unwrap_or(0.0);
        }
        out.push(matrix);
    }
    Ok(out)
}

fn read_joints(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[u16; 4]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.item_type != "VEC4" {
        return invalid(path, "JOINTS_0 accessor must be VEC4");
    }
    let component_size = match accessor.component_type {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        other => {
            return invalid(
                path,
                &format!("unsupported JOINTS_0 component type {other}; expected u8/u16/u32"),
            );
        }
    };
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view
        .byte_stride
        .unwrap_or(component_size * 4)
        .max(component_size * 4);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, component_size * 4, chunks.bin.len())?;
        let mut joints = [0u16; 4];
        for (slot, joint) in joints.iter_mut().enumerate() {
            let component_offset = offset + slot * component_size;
            *joint = match accessor.component_type {
                5121 => chunks.bin[component_offset] as u16,
                5123 => read_u16(&chunks.bin, component_offset).unwrap_or(0),
                5125 => read_u32(&chunks.bin, component_offset)
                    .unwrap_or(0)
                    .min(u16::MAX as u32) as u16,
                _ => unreachable!(),
            };
        }
        out.push(joints);
    }
    Ok(out)
}

fn read_weights(
    chunks: &GlbChunks,
    accessor_index: usize,
    path: &Path,
) -> Result<Vec<[f32; 4]>, GlbLoadError> {
    let accessor = parse_accessor(chunks, accessor_index, path)?;
    if accessor.item_type != "VEC4" {
        return invalid(path, "WEIGHTS_0 accessor must be VEC4");
    }
    let component_size = match accessor.component_type {
        5121 => 1,
        5123 => 2,
        5126 => 4,
        other => {
            return invalid(
                path,
                &format!("unsupported WEIGHTS_0 component type {other}; expected u8/u16/f32"),
            );
        }
    };
    let view = parse_buffer_view(chunks, accessor.buffer_view, path)?;
    let stride = view
        .byte_stride
        .unwrap_or(component_size * 4)
        .max(component_size * 4);
    let base = view.byte_offset + accessor.byte_offset;
    let mut out = Vec::with_capacity(accessor.count);
    for item in 0..accessor.count {
        let offset = base + item * stride;
        ensure_range(path, offset, component_size * 4, chunks.bin.len())?;
        let mut weights = [0.0f32; 4];
        for (slot, weight) in weights.iter_mut().enumerate() {
            let component_offset = offset + slot * component_size;
            *weight = match accessor.component_type {
                5121 => chunks.bin[component_offset] as f32 / 255.0,
                5123 => read_u16(&chunks.bin, component_offset).unwrap_or(0) as f32 / 65535.0,
                5126 => read_f32(&chunks.bin, component_offset).unwrap_or(0.0),
                _ => unreachable!(),
            };
        }
        out.push(weights);
    }
    Ok(out)
}

fn parse_accessor(
    chunks: &GlbChunks,
    index: usize,
    path: &Path,
) -> Result<AccessorInfo, GlbLoadError> {
    let accessor = chunks
        .json
        .get("accessors")
        .and_then(Value::as_array)
        .and_then(|items| items.get(index))
        .ok_or_else(|| invalid_err(path, &format!("accessor {index} not found")))?;
    let item_type = accessor
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_err(path, &format!("accessor {index} missing type")))?;
    Ok(AccessorInfo {
        buffer_view: json_usize(accessor, "bufferView")
            .ok_or_else(|| invalid_err(path, &format!("accessor {index} missing bufferView")))?,
        byte_offset: json_usize(accessor, "byteOffset").unwrap_or(0),
        component_type: json_usize(accessor, "componentType")
            .ok_or_else(|| invalid_err(path, &format!("accessor {index} missing componentType")))?
            as u32,
        count: json_usize(accessor, "count")
            .ok_or_else(|| invalid_err(path, &format!("accessor {index} missing count")))?,
        item_type: item_type.to_string(),
    })
}

fn parse_buffer_view(
    chunks: &GlbChunks,
    index: usize,
    path: &Path,
) -> Result<BufferViewInfo, GlbLoadError> {
    let view = chunks
        .json
        .get("bufferViews")
        .and_then(Value::as_array)
        .and_then(|items| items.get(index))
        .ok_or_else(|| invalid_err(path, &format!("bufferView {index} not found")))?;
    let byte_offset = json_usize(view, "byteOffset").unwrap_or(0);
    let byte_length = json_usize(view, "byteLength")
        .ok_or_else(|| invalid_err(path, &format!("bufferView {index} missing byteLength")))?;
    ensure_range(path, byte_offset, byte_length, chunks.bin.len())?;
    Ok(BufferViewInfo {
        byte_offset,
        byte_stride: json_usize(view, "byteStride"),
    })
}

fn compute_bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    (min, max)
}

fn normalize3(value: [f32; 3]) -> [f32; 3] {
    let len = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if len <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        [value[0] / len, value[1] / len, value[2] / len]
    }
}

fn ensure_range(path: &Path, offset: usize, len: usize, total: usize) -> Result<(), GlbLoadError> {
    if offset.saturating_add(len) <= total {
        Ok(())
    } else {
        invalid(path, "buffer range exceeds GLB BIN chunk")
    }
}

fn json_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
}

fn json_f32_array_3(value: Option<&Value>) -> Option<[f32; 3]> {
    let items = value?.as_array()?;
    Some([
        items.first()?.as_f64()? as f32,
        items.get(1)?.as_f64()? as f32,
        items.get(2)?.as_f64()? as f32,
    ])
}

fn json_f32_array_4(value: Option<&Value>) -> Option<[f32; 4]> {
    let items = value?.as_array()?;
    Some([
        items.first()?.as_f64()? as f32,
        items.get(1)?.as_f64()? as f32,
        items.get(2)?.as_f64()? as f32,
        items.get(3)?.as_f64()? as f32,
    ])
}

fn json_f32_array_16(value: Option<&Value>) -> Option<[f32; 16]> {
    let items = value?.as_array()?;
    let mut out = [0.0f32; 16];
    for (idx, slot) in out.iter_mut().enumerate() {
        *slot = items.get(idx)?.as_f64()? as f32;
    }
    Some(out)
}

fn identity_mat4() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let slice = bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_i16(bytes: &[u8], offset: usize) -> Option<i16> {
    let slice = bytes.get(offset..offset + 2)?;
    Some(i16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset + 4)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_f32(bytes: &[u8], offset: usize) -> Option<f32> {
    read_u32(bytes, offset).map(f32::from_bits)
}

fn array_len(value: Option<&Value>) -> usize {
    value.and_then(Value::as_array).map_or(0, Vec::len)
}

fn string_names(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("name").and_then(Value::as_str))
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn joint_names(json: &Value) -> Vec<String> {
    let Some(nodes) = json.get("nodes").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(skins) = json.get("skins").and_then(Value::as_array) {
        for skin in skins {
            if let Some(joints) = skin.get("joints").and_then(Value::as_array) {
                for joint in joints {
                    let Some(index) = joint.as_u64().map(|index| index as usize) else {
                        continue;
                    };
                    let Some(name) = nodes
                        .get(index)
                        .and_then(|node| node.get("name"))
                        .and_then(Value::as_str)
                    else {
                        continue;
                    };
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

fn invalid<T>(path: &Path, message: &str) -> Result<T, GlbLoadError> {
    Err(invalid_err(path, message))
}

fn invalid_err(path: &Path, message: &str) -> GlbLoadError {
    GlbLoadError::Invalid {
        path: path.to_path_buf(),
        message: message.to_string(),
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
    use serde_json::json;

    use super::{
        GlbChunks, load_glb_mesh_data, load_glb_metadata, parse_glb_animation_data,
        parse_glb_mesh_data, read_materials, texture_source_index,
    };

    #[test]
    fn animation_only_gltf_is_accepted_without_weakening_model_loading() {
        let source = br#"{
          "asset":{"version":"2.0"},
          "scene":0,
          "scenes":[{"nodes":[0]}],
          "nodes":[{"name":"source:Hips"}],
          "skins":[{"joints":[0]}],
          "animations":[{"name":"Walk","channels":[],"samplers":[]}]
        }"#;
        let path = std::path::Path::new("motion-only.gltf");

        let animation =
            parse_glb_animation_data(path, source).expect("animation-only glTF should load");
        assert!(animation.positions.is_empty());
        assert_eq!(animation.nodes.len(), 1);
        assert_eq!(animation.animations.len(), 1);
        assert_eq!(animation.animations[0].name.as_deref(), Some("Walk"));

        let model_error = parse_glb_mesh_data(path, source)
            .expect_err("ModelAsset loading must still require renderable geometry");
        assert!(
            model_error
                .to_string()
                .contains("no triangle POSITION/index data")
        );
    }

    #[test]
    fn resolves_core_webp_and_basisu_texture_sources() {
        assert_eq!(texture_source_index(&json!({ "source": 3 })), Some(3));
        assert_eq!(
            texture_source_index(&json!({
                "extensions": { "EXT_texture_webp": { "source": 7 } }
            })),
            Some(7)
        );
        assert_eq!(
            texture_source_index(&json!({
                "extensions": { "KHR_texture_basisu": { "source": 9 } }
            })),
            Some(9)
        );
    }

    #[test]
    fn loads_example_glb_metadata_when_present() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/motionloom/sample_assets/glb/mammuthus_primigenius_blumbach.glb");
        if !path.exists() {
            return;
        }
        let metadata = load_glb_metadata(&path).expect("example GLB metadata");
        assert!(metadata.mesh_count > 0);
        assert!(metadata.node_count > 0);
        assert!(metadata.material_count > 0);
    }

    #[test]
    fn loads_example_glb_mesh_data_when_present() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/motionloom/sample_assets/glb/mammuthus_primigenius_blumbach.glb");
        if !path.exists() {
            return;
        }
        let mesh = load_glb_mesh_data(&path).expect("example GLB mesh data");
        assert!(!mesh.positions.is_empty());
        assert!(mesh.indices.len() >= 3);
        assert_eq!(mesh.positions.len(), mesh.texcoords.len());
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.positions.len(), mesh.joints.len());
        assert_eq!(mesh.positions.len(), mesh.weights.len());
        assert!(!mesh.materials.is_empty());
        assert!(mesh.bounds_max[1] > mesh.bounds_min[1]);
    }

    #[test]
    fn loads_draco_iphone_glb_mesh_data_when_present() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../motionloom-example/assets/sample_assets/glb/iphone.glb");
        if !path.exists() {
            return;
        }
        let mesh = load_glb_mesh_data(&path).expect("Draco iPhone GLB mesh data");
        assert!(!mesh.positions.is_empty());
        assert!(mesh.indices.len() >= 3);
        assert_eq!(mesh.positions.len(), mesh.texcoords.len());
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert!(!mesh.materials.is_empty());
        assert!(mesh.bounds_max[1] > mesh.bounds_min[1]);
    }

    #[test]
    fn loads_self_contained_json_gltf_mesh_data() {
        let mut buffer = Vec::new();
        for value in [
            -1.0f32, -1.0, 0.0, //
            1.0, -1.0, 0.0, //
            0.0, 1.0, 0.0,
        ] {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0u16, 1, 2] {
            buffer.extend_from_slice(&index.to_le_bytes());
        }
        let document = json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }],
            "buffers": [{
                "byteLength": buffer.len(),
                "uri": format!(
                    "data:application/octet-stream;base64,{}",
                    BASE64_STANDARD.encode(&buffer)
                )
            }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3"
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": 3,
                    "type": "SCALAR"
                }
            ]
        });
        let bytes = serde_json::to_vec(&document).expect("serialize embedded glTF");
        let mesh =
            parse_glb_mesh_data(std::path::Path::new("triangle.gltf"), &bytes).expect("JSON glTF");
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.bounds_min, [-1.0, -1.0, 0.0]);
        assert_eq!(mesh.bounds_max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn parses_specular_glossiness_diffuse_material() {
        let chunks = GlbChunks {
            version: 2,
            json_len: 0,
            bin_len: 0,
            json: json!({
                "materials": [{
                    "extensions": {
                        "KHR_materials_pbrSpecularGlossiness": {
                            "diffuseFactor": [0.7, 0.6, 0.5, 0.9],
                            "diffuseTexture": { "index": 3 }
                        }
                    }
                }]
            }),
            bin: Vec::new(),
        };
        let materials = read_materials(&chunks);
        assert_eq!(materials.len(), 1);
        assert!(materials[0].specular_glossiness);
        assert_eq!(materials[0].base_color_texture, Some(3));
        assert_eq!(materials[0].base_color_factor, [0.7, 0.6, 0.5, 0.9]);
    }

    #[test]
    fn parses_metallic_roughness_normal_emissive_and_specular_material() {
        let chunks = GlbChunks {
            version: 2,
            json_len: 0,
            bin_len: 0,
            json: json!({
                "materials": [{
                    "name": "black_glass",
                    "pbrMetallicRoughness": {
                        "metallicFactor": 0.72,
                        "roughnessFactor": 0.18,
                        "metallicRoughnessTexture": { "index": 4 }
                    },
                    "normalTexture": { "index": 5, "scale": 0.65 },
                    "emissiveTexture": { "index": 6 },
                    "emissiveFactor": [0.1, 0.2, 0.3],
                    "extensions": {
                        "KHR_materials_emissive_strength": { "emissiveStrength": 2.5 },
                        "KHR_materials_specular": {
                            "specularFactor": 0.8,
                            "specularColorFactor": [0.9, 0.85, 0.75]
                        },
                        "KHR_materials_transmission": { "transmissionFactor": 0.94 },
                        "KHR_materials_ior": { "ior": 1.52 },
                        "KHR_materials_volume": {
                            "thicknessFactor": 0.012,
                            "attenuationColor": [0.72, 0.87, 0.89],
                            "attenuationDistance": 6.0
                        }
                    }
                }]
            }),
            bin: Vec::new(),
        };
        let materials = read_materials(&chunks);
        assert_eq!(materials.len(), 1);
        let material = &materials[0];
        assert_eq!(material.name.as_deref(), Some("black_glass"));
        assert_eq!(material.metallic_roughness_texture, Some(4));
        assert_eq!(material.normal_texture, Some(5));
        assert_eq!(material.emissive_texture, Some(6));
        assert_eq!(material.metallic_factor, 0.72);
        assert_eq!(material.roughness_factor, 0.18);
        assert_eq!(material.normal_scale, 0.65);
        assert_eq!(material.emissive_factor, [0.1, 0.2, 0.3]);
        assert_eq!(material.emissive_strength, 2.5);
        assert_eq!(material.specular_factor, 0.8);
        assert_eq!(material.specular_color_factor, [0.9, 0.85, 0.75]);
        assert_eq!(material.transmission_factor, 0.94);
        assert_eq!(material.ior, 1.52);
        assert_eq!(material.thickness_factor, 0.012);
        assert_eq!(material.attenuation_color, [0.72, 0.87, 0.89]);
        assert_eq!(material.attenuation_distance, 6.0);
    }
}
