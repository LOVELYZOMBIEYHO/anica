// =========================================
// =========================================
// crates/motionloom/src/dsl.rs

use std::collections::HashMap;

pub use crate::error::GraphParseError;
pub use crate::process::model::{
    AlphaMode, BlendMode, BufferElemType, BufferNode, BufferUsage, ColorSpace, EffectNode,
    GraphApplyScope, InputNode, InputType, LayerNode, LoadOp, OutputNode, OutputTarget, PassCache,
    PassKind, PassNode, PassParam, PassRole, PassTransitionClips, PassTransitionEasing,
    PassTransitionFallback, PassTransitionMode, PresentNode, PresentTarget, Quality, ResourceRef,
    SampleAddress, SampleConfig, SampleFilter, StoreOp, TexNode, TexUsage, TextureFormat,
};
pub use crate::scene::dsl::{
    ActionBoneNode, ActionContactNode, ActionNode, ActionPoseNode, ApplyActionNode, BackgroundNode,
    ImageNode, ModelProfileBoneAxisMapNode, ModelProfileBoneAxisNode, ModelProfileNode,
    ModelProfileRetargetMapNode, ModelProfileRetargetNode, SkeletonBoneNode,
    SkeletonConstraintNode, SkeletonControlNode, SkeletonGuideNode, SkeletonLandmarkNode,
    SkeletonMeasureNode, SkeletonNode, SkeletonRatioNode, SkeletonRegionNode, SvgNode,
};
use crate::scene::dsl::{
    BrushParseContext, lower_parametric_component_uses, parse_action_block,
    parse_apply_action_node, parse_background_node, parse_camera_block, parse_camera_node,
    parse_character_block, parse_circle_node, parse_defs_block, parse_face_jaw_node,
    parse_group_block, parse_image_node, parse_layout_block, parse_line_node, parse_mask_any,
    parse_mesh_topology_block, parse_model_profile_block, parse_part_block, parse_path_node,
    parse_pin_node, parse_pixel_grid_block, parse_polyline_node, parse_precompose_block,
    parse_puppet_block, parse_rect_node, parse_repeat_block, parse_scene_root_block,
    parse_shadow_node, parse_skeleton_block, parse_svg_node, parse_text_node,
    resolve_lowered_puppet_targets, validate_scene_camera_structure,
    validate_scene_model_profile_refs,
};
use crate::scene::model::{SceneNode, SceneRootNode};
pub use crate::scene::text::TextNode;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GraphScript {
    #[serde(default)]
    pub audio_clips: Vec<crate::audio::AudioClipNode>,
    #[serde(default)]
    pub audio_targets: Vec<crate::audio::AudioTargetNode>,
    /// Scene visual resources; absence preserves the legacy renderer.
    #[serde(default)]
    pub render_styles: Vec<crate::render_style::RenderStyleNode>,
    #[serde(skip)]
    pub raw_script: Option<String>,
    pub id: Option<String>,
    pub version: Option<String>,
    pub fps: f32,
    #[serde(default)]
    pub apply: GraphApplyScope,
    pub duration_ms: u64,
    #[serde(default)]
    pub duration_explicit: bool,
    pub size: (u32, u32),
    #[serde(default)]
    pub render_size: Option<(u32, u32)>,
    /// External files shared by every Scene and Process in the Graph.
    #[serde(default)]
    pub assets: Vec<GraphAssetNode>,
    /// Reusable physically based materials referenced by typed geometry assets.
    #[serde(default)]
    pub material_assets: Vec<MaterialAssetNode>,
    pub inputs: Vec<InputNode>,
    pub textures: Vec<TexNode>,
    pub buffers: Vec<BufferNode>,
    #[serde(default)]
    pub backgrounds: Vec<BackgroundNode>,
    #[serde(default)]
    pub texts: Vec<TextNode>,
    #[serde(default)]
    pub images: Vec<ImageNode>,
    #[serde(default)]
    pub svgs: Vec<SvgNode>,
    #[serde(default)]
    pub scenes: Vec<SceneRootNode>,
    #[serde(default)]
    pub scene_nodes: Vec<SceneNode>,
    #[serde(default)]
    pub model_profiles: Vec<ModelProfileNode>,
    #[serde(default)]
    pub skeletons: Vec<SkeletonNode>,
    #[serde(default)]
    pub actions: Vec<ActionNode>,
    /// Selective, namespaced imports from standalone MotionLoom action libraries.
    #[serde(default)]
    pub action_libraries: Vec<ActionLibraryNode>,
    #[serde(default)]
    pub apply_actions: Vec<ApplyActionNode>,
    /// Runtime bindings that align one model-local socket with another model's bone.
    #[serde(default)]
    pub attachments: Vec<AttachmentNode>,
    /// Reusable semantic planes used by Action contact correction.
    #[serde(default)]
    pub contact_surfaces: Vec<ContactSurfaceNode>,
    /// Time-bounded relationships between 3D model endpoints.
    #[serde(default)]
    pub scene_constraints: Vec<SceneConstraintNode>,
    #[serde(default)]
    pub animation_targets: Vec<AnimationTargetNode>,
    #[serde(default)]
    pub layers: Vec<LayerNode>,
    /// Process boundary metadata. The original flattened Process resources and
    /// passes remain untouched; this index lets Scene effect scopes safely
    /// reference an existing Process by id.
    #[serde(default)]
    pub processes: Vec<ProcessDefinitionNode>,
    #[serde(default)]
    pub world_sources: Vec<WorldSourceNode>,
    pub passes: Vec<PassNode>,
    pub outputs: Vec<OutputNode>,
    pub present: PresentNode,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionLibraryNode {
    /// Namespace used by ApplyAction, for example `cinematic.formal_bow`.
    pub id: String,
    /// Relative path, URL, or WASM resolver key for the library document.
    pub src: String,
    /// Action ids selectively imported from the external document.
    pub actions: Vec<String>,
}

/// A complete object-socket to target-bone binding evaluated after animation.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentNode {
    pub id: String,
    pub object: String,
    pub sockets: Vec<AttachmentSocketNode>,
    pub attaches: Vec<AttachmentTargetNode>,
}

/// Object-local frame that is aligned with the target bone.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentSocketNode {
    pub id: String,
    pub position: [String; 3],
    pub rotation: [String; 3],
}

/// Target and blend controls for one attachment.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentTargetNode {
    pub socket: String,
    pub target: String,
    pub target_model: String,
    pub target_bone: String,
    pub mode: String,
    pub drive: String,
    pub position_weight: String,
    pub rotation_weight: String,
    pub max_stretch: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphAssetNode {
    pub id: String,
    pub kind: GraphAssetKind,
    pub source: GraphAssetSource,
    #[serde(default)]
    pub decoder: Option<String>,
    #[serde(default)]
    pub color_space: Option<String>,
    /// Source skeleton convention used by imported animation-only assets.
    #[serde(default)]
    pub profile: Option<String>,
    /// Optional named clip selected from a multi-animation GLB.
    #[serde(default)]
    pub clip: Option<String>,
}

/// A Graph asset is either externally resolved data or typed engine geometry.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", tag = "type")]
#[expect(
    clippy::large_enum_variant,
    reason = "Keep public asset constructors source-compatible; boxing changes the Rust API."
)]
pub enum GraphAssetSource {
    External { src: String },
    Primitive(PrimitiveAssetNode),
    Terrain(TerrainAssetNode),
    Vegetation(VegetationAssetNode),
    Compound(CompoundAssetNode),
}

impl GraphAssetNode {
    pub fn external_src(&self) -> Option<&str> {
        match &self.source {
            GraphAssetSource::External { src } => Some(src),
            GraphAssetSource::Primitive(_)
            | GraphAssetSource::Terrain(_)
            | GraphAssetSource::Vegetation(_)
            | GraphAssetSource::Compound(_) => None,
        }
    }

    pub fn primitive(&self) -> Option<&PrimitiveAssetNode> {
        match &self.source {
            GraphAssetSource::Primitive(asset) => Some(asset),
            GraphAssetSource::External { .. }
            | GraphAssetSource::Terrain(_)
            | GraphAssetSource::Vegetation(_)
            | GraphAssetSource::Compound(_) => None,
        }
    }

    pub fn compound(&self) -> Option<&CompoundAssetNode> {
        match &self.source {
            GraphAssetSource::Compound(asset) => Some(asset),
            GraphAssetSource::External { .. }
            | GraphAssetSource::Primitive(_)
            | GraphAssetSource::Terrain(_)
            | GraphAssetSource::Vegetation(_) => None,
        }
    }

    pub fn terrain(&self) -> Option<&TerrainAssetNode> {
        match &self.source {
            GraphAssetSource::Terrain(asset) => Some(asset),
            GraphAssetSource::External { .. }
            | GraphAssetSource::Primitive(_)
            | GraphAssetSource::Vegetation(_)
            | GraphAssetSource::Compound(_) => None,
        }
    }

    pub fn vegetation(&self) -> Option<&VegetationAssetNode> {
        match &self.source {
            GraphAssetSource::Vegetation(asset) => Some(asset),
            GraphAssetSource::External { .. }
            | GraphAssetSource::Primitive(_)
            | GraphAssetSource::Terrain(_)
            | GraphAssetSource::Compound(_) => None,
        }
    }
}

/// A heightfield terrain remains a typed model asset while reusing the shared
/// PBR mesh renderer on native and WASM targets.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerrainAssetNode {
    pub id: String,
    pub height_map: String,
    #[serde(default)]
    pub height_map_src: Option<String>,
    pub size: [f32; 2],
    pub height_scale: f32,
    pub height_offset: f32,
    pub material: Option<String>,
    #[serde(default)]
    pub material_definition: Option<MaterialAssetNode>,
    /// Up to four PBR materials blended by the RGBA channels of blend_map.
    #[serde(default)]
    pub layers: Vec<String>,
    #[serde(default)]
    pub layer_definitions: Vec<MaterialAssetNode>,
    #[serde(default)]
    pub blend_map: Option<String>,
    #[serde(default)]
    pub blend_map_src: Option<String>,
    pub chunks: [u32; 2],
    pub lod: String,
    pub collision: PrimitiveCollisionMode,
}

/// One reusable procedural plant or plant cluster. Scene-wide distribution is
/// deliberately outside this asset so vegetation generation stays bounded.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VegetationAssetNode {
    pub id: String,
    pub kind: VegetationKind,
    pub height: f32,
    pub material: Option<String>,
    #[serde(default)]
    pub material_definition: Option<MaterialAssetNode>,
    pub stem_material: Option<String>,
    #[serde(default)]
    pub stem_material_definition: Option<MaterialAssetNode>,
    pub trunk_material: Option<String>,
    #[serde(default)]
    pub trunk_material_definition: Option<MaterialAssetNode>,
    pub foliage_material: Option<String>,
    #[serde(default)]
    pub foliage_material_definition: Option<MaterialAssetNode>,
    pub density: u32,
    pub branch_levels: u32,
    pub seed: u64,
    pub lod: VegetationLod,
    pub wind: bool,
    pub collision: PrimitiveCollisionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VegetationKind {
    Tree,
    Shrub,
    Grass,
    Flower,
    Fern,
    Deadwood,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VegetationLod {
    Auto,
    Full,
    Half,
    Quarter,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompoundAssetNode {
    pub id: String,
    #[serde(default)]
    pub rig: Option<String>,
    #[serde(default)]
    pub material_seed: Option<u64>,
    /// Optional native skin settings preserve the existing rigid compound path.
    #[serde(default)]
    pub skin_binding: Option<NativeSkinBindingNode>,
    pub instances: Vec<CompoundAssetInstanceNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeSkinBindingNode {
    pub mode: String,
    pub max_influences: u32,
    pub falloff: f32,
    pub normalize: bool,
    #[serde(default)]
    pub weight_regions: Vec<NativeWeightRegionNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeWeightRegionNode {
    pub instance: String,
    pub bone: String,
    pub center: [f32; 3],
    pub radius: f32,
    pub strength: f32,
    pub operation: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeSkinMode {
    #[default]
    Rigid,
    Smooth,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompoundAssetInstanceNode {
    pub id: String,
    pub asset: String,
    #[serde(default)]
    pub bone: Option<String>,
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: f32,
    #[serde(default)]
    pub material_seed: Option<u64>,
    /// Smooth is opt-in; omitted instances remain byte-compatible rigid parts.
    #[serde(default)]
    pub skin: NativeSkinMode,
    #[serde(default)]
    pub influences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrimitiveAssetNode {
    pub id: String,
    pub geometry: PrimitiveGeometry,
    pub color: [f32; 4],
    /// Optional stable id plus its resolved definition keep materials reusable
    /// while allowing the world renderer to consume a self-contained asset.
    #[serde(default)]
    pub material: Option<String>,
    #[serde(default)]
    pub material_definition: Option<MaterialAssetNode>,
    #[serde(default)]
    pub bevel_radius: f32,
    #[serde(default)]
    pub bevel_segments: u32,
    #[serde(default)]
    pub material_seed: Option<u64>,
    #[serde(default)]
    pub collision: PrimitiveCollisionNode,
    /// Advanced procedural controls are optional so every existing primitive
    /// keeps its original geometry and runtime behavior.
    #[serde(default)]
    pub modifiers: Vec<PrimitiveModifierNode>,
    #[serde(default)]
    pub mesh_build: PrimitiveMeshBuildNode,
    #[serde(default)]
    pub lod: PrimitiveLodNode,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum PrimitiveModifierNode {
    Transform {
        translate: [f32; 3],
        rotate: [f32; 3],
        scale: [f32; 3],
    },
    Taper {
        axis: PrimitiveAxis,
        start: f32,
        end: f32,
    },
    Bend {
        axis: PrimitiveAxis,
        angle: f32,
        pivot: [f32; 3],
    },
    Twist {
        axis: PrimitiveAxis,
        angle: f32,
    },
    Subdivision {
        levels: u32,
    },
    Smooth {
        angle: f32,
    },
    WeightedNormals {
        strength: f32,
        keep_sharp_edges: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PrimitiveAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PrimitiveMeshBuildNode {
    pub topology: String,
    pub triangulation: String,
    pub quality: String,
    pub max_triangles: Option<u32>,
}

impl Default for PrimitiveMeshBuildNode {
    fn default() -> Self {
        Self {
            topology: "auto".to_string(),
            triangulation: "auto".to_string(),
            quality: "standard".to_string(),
            max_triangles: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PrimitiveLodNode {
    pub mode: String,
    pub levels: u32,
    pub preserve_silhouette: bool,
}

impl Default for PrimitiveLodNode {
    fn default() -> Self {
        Self {
            mode: "none".to_string(),
            levels: 1,
            preserve_silhouette: true,
        }
    }
}

/// A first-class PBR material reuses the glTF renderer without pretending that
/// a screen-space Scene texture is a physical 3D surface.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterialAssetNode {
    pub id: String,
    pub shading: String,
    pub base_color: [f32; 4],
    pub base_color_texture: Option<String>,
    pub metallic_roughness_texture: Option<String>,
    pub normal_texture: Option<String>,
    pub occlusion_texture: Option<String>,
    pub emissive_texture: Option<String>,
    /// Resolved image sources are intentionally retained beside public ids so
    /// generated primitives remain renderable after CompoundAsset expansion.
    #[serde(default)]
    pub base_color_texture_src: Option<String>,
    #[serde(default)]
    pub metallic_roughness_texture_src: Option<String>,
    #[serde(default)]
    pub normal_texture_src: Option<String>,
    #[serde(default)]
    pub occlusion_texture_src: Option<String>,
    #[serde(default)]
    pub emissive_texture_src: Option<String>,
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub specular: f32,
    pub double_sided: bool,
    pub alpha_mode: String,
    pub alpha_cutoff: f32,
    /// Transmission models a solid surface that passes light; it is distinct
    /// from alpha coverage used by decals, smoke, and fades.
    #[serde(default)]
    pub transmission: f32,
    #[serde(default = "default_material_ior")]
    pub ior: f32,
    #[serde(default)]
    pub thickness: f32,
    #[serde(default = "default_material_attenuation_color")]
    pub attenuation_color: [f32; 3],
    #[serde(default = "default_material_attenuation_distance")]
    pub attenuation_distance: f32,
    #[serde(default = "default_material_depth_write")]
    pub depth_write: String,
    #[serde(default)]
    pub sort_priority: i32,
    pub mapping: String,
    pub texture_scale: [f32; 2],
    pub texture_offset: [f32; 2],
    pub texture_rotation: f32,
    pub variation_amount: [f32; 2],
}

fn default_material_ior() -> f32 {
    1.5
}

fn default_material_attenuation_color() -> [f32; 3] {
    [1.0; 3]
}

fn default_material_attenuation_distance() -> f32 {
    1_000_000.0
}

fn default_material_depth_write() -> String {
    "auto".to_string()
}

/// Asset-owned collision data stays reusable across every primitive instance.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PrimitiveCollisionNode {
    pub mode: PrimitiveCollisionMode,
    pub collider: PrimitiveColliderShape,
    pub size: Option<Vec<f32>>,
    pub radius: Option<f32>,
    pub height: Option<f32>,
    pub scale: [f32; 3],
    pub offset: [f32; 3],
    pub rotation: [f32; 3],
    pub margin: f32,
    pub friction: f32,
    pub restitution: f32,
    pub density: f32,
    pub group: u32,
    pub mask: u32,
}

impl Default for PrimitiveCollisionNode {
    fn default() -> Self {
        Self {
            mode: PrimitiveCollisionMode::None,
            collider: PrimitiveColliderShape::Auto,
            size: None,
            radius: None,
            height: None,
            scale: [1.0; 3],
            offset: [0.0; 3],
            rotation: [0.0; 3],
            margin: 0.0,
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            group: 1,
            mask: u32::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveCollisionMode {
    None,
    Solid,
    Sensor,
}

impl PrimitiveCollisionMode {
    pub fn participates(self) -> bool {
        self != Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrimitiveColliderShape {
    Auto,
    Box,
    Sphere,
    Capsule,
    Plane,
    Cylinder,
    Cone,
    Convex,
    Mesh,
}

/// Explicit polygon control cage shared by subdivision surfaces and semantic assets.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ControlCageNode {
    pub positions: Vec<[f32; 3]>,
    #[serde(default)]
    pub uvs: Vec<[f32; 2]>,
    pub pinned: Vec<bool>,
    pub faces: Vec<Vec<u32>>,
    pub subdivision: u32,
}

/// One measured horizontal section of a semantic head profile.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadSectionNode {
    pub id: String,
    pub at: f32,
    pub width: f32,
    pub front_depth: f32,
    pub back_depth: f32,
}

/// Analytic crown parameters keep the generated pole smooth and deterministic.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadDomeNode {
    pub start: f32,
    pub top: f32,
    pub center_depth: f32,
    pub front_radius: f32,
    pub back_radius: f32,
    pub samples: u32,
}

/// Versioned settings for the reusable section/patch facial-cage generator.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FacialCageNode {
    pub generator_version: u32,
    pub segments: u32,
    pub profile_segments: u32,
    pub samples_per_section: u32,
    pub subdivision: u32,
    pub orbital_rings: u32,
    pub mouth_rings: u32,
    pub preserve_profile: bool,
    pub uv_mode: String,
}

#[path = "dsl_control_cage.rs"]
mod control_cage_parser;

// Keep the public serialized geometry model direct and stable across all variants.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", tag = "shape")]
pub enum PrimitiveGeometry {
    /// Arbitrary explicit triangle/quad mesh with optional Catmull-Clark subdivision.
    Mesh {
        cage: ControlCageNode,
    },
    Box {
        size: [f32; 3],
    },
    Sphere {
        radius: f32,
        segments: u32,
        rings: u32,
    },
    Capsule {
        radius: f32,
        height: f32,
        segments: u32,
        rings: u32,
    },
    Plane {
        size: [f32; 2],
        segments: u32,
    },
    Cylinder {
        radius: f32,
        height: f32,
        segments: u32,
    },
    Cone {
        radius: f32,
        height: f32,
        segments: u32,
    },
    Wedge {
        size: [f32; 3],
    },
    Ellipsoid {
        radii: [f32; 3],
        segments: u32,
        rings: u32,
    },
    Frustum {
        top_size: [f32; 2],
        bottom_size: [f32; 2],
        height: f32,
    },
    RoundedBox {
        size: [f32; 3],
        radius: f32,
        segments: u32,
    },
    Loft {
        segments: u32,
        closed: bool,
        cap_start: bool,
        cap_end: bool,
        sections: Vec<PrimitiveLoftSectionNode>,
    },
    Ribbon {
        width: f32,
        thickness: f32,
        cap_start: bool,
        cap_end: bool,
        points: Vec<PrimitiveRibbonPointNode>,
    },
    /// Guide-authored hair remains representation-neutral even though V1
    /// compiles it to retained card geometry.
    HairCards {
        representation_id: String,
        bind_bone: Option<String>,
        space: String,
        length_segments: u32,
        width_segments: u32,
        thickness: f32,
        cross_section: String,
        tip_shape: String,
        guides: Vec<HairGuideNode>,
    },
    /// Head surfaces use a species-neutral volume plus optional semantic
    /// feature fields. Human face helpers are data, not a fixed topology.
    HeadSurface {
        archetype: String,
        variant: Option<String>,
        bind_bone: Option<String>,
        symmetry: String,
        topology: String,
        segments: u32,
        rings: u32,
        head_shape: HeadShapeNode,
        face_layout: Option<FaceLayoutNode>,
        facial_cage: Option<FacialCageNode>,
        head_profile: Vec<HeadSectionNode>,
        head_dome: Option<HeadDomeNode>,
        explicit_cage: Option<ControlCageNode>,
        features: Vec<HeadFeatureNode>,
        morph: HeadMorphNode,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrimitiveLoftSectionNode {
    pub at: f32,
    pub width: f32,
    pub depth: f32,
    pub profile: String,
    pub offset: [f32; 2],
    pub rotation: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrimitiveRibbonPointNode {
    pub position: [f32; 3],
    pub width: Option<f32>,
    pub thickness: Option<f32>,
    pub roll: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HairGuideNode {
    pub id: String,
    pub group: String,
    pub role: String,
    /// Optional outward root direction, projected onto the root tangent plane.
    #[serde(default)]
    pub normal: Option<[f32; 3]>,
    pub points: Vec<HairPointNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HairPointNode {
    pub position: [f32; 3],
    pub width: f32,
    pub radius: f32,
    pub camber: f32,
    pub roll: f32,
    pub stiffness: f32,
}

/// Species-neutral cranium and jaw proportions. These values do not assume a
/// human face and can also describe mammal, reptile, or invented head volumes.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadShapeNode {
    pub size: [f32; 3],
    pub forehead: f32,
    pub cheek_width: f32,
    pub jaw_width: f32,
    pub chin_length: f32,
    pub chin_roundness: f32,
}

/// Explicit facial components with independent IDs and arbitrary component counts.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceLayoutNode {
    pub eyebrows: Vec<EyebrowNode>,
    pub eyes: Vec<EyeNode>,
    pub noses: Vec<NoseNode>,
    pub mouths: Vec<MouthNode>,
    pub ears: Vec<EarNode>,
}

/// A surface-attached eyebrow ribbon generated in head-local coordinates.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyebrowNode {
    pub id: String,
    pub position: [f32; 3],
    pub width: f32,
    pub thickness: f32,
    pub arch: f32,
    pub tilt: f32,
    pub texture: Option<FaceTextureNode>,
}

/// Image binding in the owning component's UV space.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceTextureNode {
    pub asset: String,
    /// Resolved through the same asset resolver as material textures.
    pub source: Option<String>,
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyeNode {
    pub id: String,
    pub position: [f32; 3],
    pub width: f32,
    pub opening: f32,
    pub tilt: f32,
    pub socket_width: f32,
    pub socket_height: f32,
    pub socket_depth: f32,
    pub texture: Option<FaceTextureNode>,
    pub iris: Option<IrisNode>,
    pub eyeliners: Vec<EyelinerNode>,
}

/// Optional iris disc attached to an eye's generated eyeball.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IrisNode {
    pub id: String,
    /// Eye-local offset; X/Y move the iris across the eyeball and Z lifts it.
    pub position: [f32; 3],
    pub shape: String,
    /// Geometry scale applied to radius before the nested Texture transform.
    pub scale: [f32; 2],
    pub radius: f32,
    pub pupil_radius: f32,
    pub texture: Option<FaceTextureNode>,
}

/// One independently authored ribbon attached to an upper or lower eyelid edge.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EyelinerNode {
    pub id: String,
    pub edge: String,
    pub thickness: f32,
    pub span: [f32; 2],
    pub taper: [f32; 2],
    pub extension: [f32; 2],
    pub tip_lift: [f32; 2],
    pub texture: Option<FaceTextureNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NoseNode {
    pub id: String,
    pub position: [f32; 3],
    pub length: f32,
    pub width: f32,
    pub projection: f32,
    pub texture: Option<FaceTextureNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MouthNode {
    pub id: String,
    pub position: [f32; 3],
    pub width: f32,
    pub opening: f32,
    pub upper_lip: f32,
    pub lower_lip: f32,
    pub muzzle_length: f32,
    pub muzzle_width: f32,
    pub texture: Option<FaceTextureNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EarNode {
    pub id: String,
    pub position: [f32; 3],
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub texture: Option<FaceTextureNode>,
}

#[path = "dsl_face_layout.rs"]
mod face_layout_parser;

/// One continuous deformation field on the base head surface. The open string
/// kind allows future anatomy without changing the serialized representation.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadFeatureNode {
    pub id: String,
    pub kind: String,
    pub center: [f32; 3],
    pub size: [f32; 3],
    pub amount: f32,
    pub offset: [f32; 3],
    pub falloff: String,
    pub mirror_x: bool,
}

/// Non-destructive proportions are kept separate from authored anatomy so one
/// head definition can be varied without moving every feature by hand.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HeadMorphNode {
    pub head_width: f32,
    pub head_height: f32,
    pub head_depth: f32,
    pub face_width: f32,
    pub face_height: f32,
    pub jaw_width: f32,
    pub muzzle_length: f32,
    pub feature_scale: f32,
}

impl PrimitiveGeometry {
    pub fn shape_name(&self) -> &'static str {
        match self {
            Self::Mesh { .. } => "mesh",
            Self::Box { .. } => "box",
            Self::Sphere { .. } => "sphere",
            Self::Capsule { .. } => "capsule",
            Self::Plane { .. } => "plane",
            Self::Cylinder { .. } => "cylinder",
            Self::Cone { .. } => "cone",
            Self::Wedge { .. } => "wedge",
            Self::Ellipsoid { .. } => "ellipsoid",
            Self::Frustum { .. } => "frustum",
            Self::RoundedBox { .. } => "roundedBox",
            Self::Loft { .. } => "loft",
            Self::Ribbon { .. } => "ribbon",
            Self::HairCards { .. } => "hairCards",
            Self::HeadSurface { .. } => "headSurface",
        }
    }

    pub fn triangle_count(&self) -> usize {
        match self {
            Self::Mesh { cage } => {
                if cage.subdivision == 0 {
                    cage.faces.iter().map(|f| f.len() - 2).sum()
                } else {
                    cage.faces.iter().map(Vec::len).sum::<usize>()
                        * 2
                        * 4usize.pow(cage.subdivision - 1)
                }
            }
            Self::Box { .. } => 12,
            Self::Sphere {
                segments, rings, ..
            } => (*segments * *rings * 2) as usize,
            Self::Capsule {
                segments, rings, ..
            } => {
                let hemisphere_rings = ((*rings).max(4) / 2).max(2);
                (*segments * (hemisphere_rings * 2 + 1) * 2) as usize
            }
            Self::Plane { segments, .. } => (*segments * *segments * 2) as usize,
            Self::Cylinder { segments, .. } => (*segments * 4) as usize,
            Self::Cone { segments, .. } => (*segments * 2) as usize,
            Self::Wedge { .. } => 8,
            Self::Ellipsoid {
                segments, rings, ..
            } => (*segments * *rings * 2) as usize,
            Self::Frustum { .. } => 12,
            Self::RoundedBox { segments, .. } => {
                let samples = segments * 2 + 1;
                (samples * samples * 12) as usize
            }
            Self::Loft {
                segments,
                sections,
                cap_start,
                cap_end,
                ..
            } => {
                let sides = sections.len().saturating_sub(1) * *segments as usize * 2;
                sides
                    + usize::from(*cap_start) * *segments as usize
                    + usize::from(*cap_end) * *segments as usize
            }
            Self::Ribbon { points, .. } => points.len().saturating_sub(1) * 8,
            Self::HairCards {
                length_segments,
                width_segments,
                guides,
                ..
            } => {
                guides.len()
                    * (*length_segments as usize * *width_segments as usize * 4
                        + *length_segments as usize * 4
                        + *width_segments as usize * 4)
            }
            Self::HeadSurface {
                segments, rings, ..
            } => (*segments * *rings * 2) as usize,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum GraphAssetKind {
    Video,
    Image,
    Model,
    Audio,
    Animation,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneConstraintNode {
    pub constraint_type: String,
    pub source: String,
    pub target: String,
    pub at_ms: u64,
    pub duration_ms: u64,
    pub solver: String,
    #[serde(default)]
    pub weight: String,
}

/// A lightweight semantic contact plane. It complements collision geometry
/// with author intent such as a seat, bed, wall, or work surface.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContactSurfaceNode {
    pub id: String,
    pub source: String,
    pub kind: String,
    pub plane: String,
    #[serde(default)]
    pub position: Option<[f32; 3]>,
    pub normal: [f32; 3],
    pub forward: [f32; 3],
    pub bounds: [f32; 2],
    pub margin: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessDefinitionNode {
    pub id: String,
    pub output: String,
    #[serde(default)]
    pub input_ids: Vec<String>,
    #[serde(default)]
    pub texture_ids: Vec<String>,
    #[serde(default)]
    pub pass_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationTargetNode {
    pub node: String,
    pub property: String,
    pub keys: Vec<AnimationKeyNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimationKeyNode {
    pub frame: u32,
    #[serde(default)]
    pub time: Option<String>,
    #[serde(default)]
    pub seconds: f32,
    pub value: String,
    #[serde(default = "default_animation_key_ease")]
    pub ease: String,
}

fn default_animation_key_ease() -> String {
    "linear".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldSourceNode {
    pub id: String,
}

impl GraphScript {
    pub fn skeleton_validation_reports(
        &self,
    ) -> Vec<crate::scene::domain::SkeletonValidationReport> {
        self.skeletons
            .iter()
            .map(crate::scene::domain::validate_skeleton)
            .collect()
    }

    pub fn summary(&self) -> String {
        format!(
            "Graph parsed: fps={:.2}, apply={:?}, duration={}ms, size={}x{}, asset={}, input={}, tex={}, buffer={}, scene={}, scene_node={}, model_profile={}, skeleton={}, action={}, apply_action={}, animation_target={}, layer={}, process={}, world={}, pass={}, output={}, present={}",
            self.fps,
            self.apply,
            self.duration_ms,
            self.size.0,
            self.size.1,
            self.assets.len(),
            self.inputs.len(),
            self.textures.len(),
            self.buffers.len(),
            self.scenes.len(),
            self.scene_nodes.len(),
            self.model_profiles.len(),
            self.skeletons.len(),
            self.actions.len(),
            self.apply_actions.len(),
            self.animation_targets.len(),
            self.layers.len(),
            self.processes.len(),
            self.world_sources.len(),
            self.passes.len(),
            self.outputs.len(),
            self.present.from
        )
    }

    pub fn resource_size(&self, id: &str) -> Option<(u32, u32)> {
        if id == "scene" && self.has_scene_nodes() {
            return Some(self.size);
        }
        if let Some(scene_id) = id.strip_prefix("scene:") {
            return self
                .scenes
                .iter()
                .find(|scene| scene.id == scene_id)
                .map(|scene| scene.size.unwrap_or(self.size));
        }
        if let Some(scene) = self.scenes.iter().find(|scene| scene.id == id) {
            return Some(scene.size.unwrap_or(self.size));
        }
        if self.world_sources.iter().any(|world| world.id == id) {
            return Some(self.render_size.unwrap_or(self.size));
        }
        self.outputs
            .iter()
            .find(|o| o.id == id)
            .and_then(|o| o.size)
            .or_else(|| {
                self.textures
                    .iter()
                    .find(|t| t.id == id)
                    .and_then(|t| t.size)
            })
            .or_else(|| self.inputs.iter().find(|i| i.id == id).and_then(|i| i.size))
    }

    pub fn has_scene_nodes(&self) -> bool {
        !self.backgrounds.is_empty()
            || !self.texts.is_empty()
            || !self.images.is_empty()
            || !self.svgs.is_empty()
            || !self.scenes.is_empty()
            || !self.scene_nodes.is_empty()
    }
}

pub fn is_graph_script(input: &str) -> bool {
    graph_root_start(input).is_ok()
}

pub(crate) fn graph_root_start(input: &str) -> Result<usize, GraphParseError> {
    let Some(start) = first_non_ws_or_comment(input, 0, input.len()) else {
        return Err(GraphParseError {
            line: 1,
            message: "Missing <Graph ...> root tag.".to_string(),
        });
    };
    if input[start..].starts_with("<!--") {
        return Err(GraphParseError {
            line: line_of_byte(input, start),
            message: "Unclosed XML comment.".to_string(),
        });
    }
    let Some(graph_start) = find_open_tag_byte(input, "Graph", start) else {
        return Err(GraphParseError {
            line: line_of_byte(input, start),
            message: "Missing <Graph ...> root tag.".to_string(),
        });
    };
    if graph_start != start {
        return Err(GraphParseError {
            line: line_of_byte(input, start),
            message: "Only whitespace and XML comments may appear before <Graph ...>.".to_string(),
        });
    }
    Ok(graph_start)
}

pub(crate) fn validate_graph_present_placement(input: &str) -> Result<(), GraphParseError> {
    let normalized = input.replace('＝', "=");
    let graph_start = graph_root_start(&normalized)?;
    let graph_open_end =
        find_tag_end_byte(&normalized, graph_start).ok_or_else(|| GraphParseError {
            line: line_of_byte(&normalized, graph_start),
            message: "Unclosed <Graph ...> opening tag.".to_string(),
        })?;
    let graph_close = normalized[graph_open_end + 1..]
        .rfind("</Graph>")
        .map(|offset| graph_open_end + 1 + offset)
        .ok_or_else(|| GraphParseError {
            line: line_of_byte(&normalized, graph_start),
            message: "Missing </Graph> closing tag.".to_string(),
        })?;

    let mut present_count = 0usize;
    let mut stack = Vec::<String>::new();
    let mut cursor = graph_open_end + 1;
    while cursor < graph_close {
        let Some(rel_tag_start) = normalized[cursor..graph_close].find('<') else {
            break;
        };
        let tag_start = cursor + rel_tag_start;
        if normalized[tag_start..].starts_with("<!--") {
            let Some(rel_end) = normalized[tag_start + 4..graph_close].find("-->") else {
                return Err(GraphParseError {
                    line: line_of_byte(&normalized, tag_start),
                    message: "Unclosed XML comment.".to_string(),
                });
            };
            cursor = tag_start + 4 + rel_end + 3;
            continue;
        }
        let Some(tag_end) = find_tag_end_byte(&normalized, tag_start) else {
            return Err(GraphParseError {
                line: line_of_byte(&normalized, tag_start),
                message: "Tag block is not closed.".to_string(),
            });
        };
        if tag_end >= graph_close {
            break;
        }
        let tag = &normalized[tag_start..=tag_end];
        if tag.starts_with("</") {
            if let Some(name) = closing_tag_name(tag) {
                if stack.last().is_some_and(|last| last == name) {
                    stack.pop();
                } else if let Some(pos) = stack.iter().rposition(|open| open == name) {
                    stack.truncate(pos);
                }
            }
            cursor = tag_end + 1;
            continue;
        }

        let Some(name) = opening_tag_name(tag) else {
            cursor = tag_end + 1;
            continue;
        };
        if name == "Present" {
            present_count += 1;
            if present_count > 1 {
                return Err(GraphParseError {
                    line: line_of_byte(&normalized, tag_start),
                    message: "Only one <Present ... /> node is supported.".to_string(),
                });
            }
            if let Some(parent) = stack.last() {
                return Err(GraphParseError {
                    line: line_of_byte(&normalized, tag_start),
                    message: format!(
                        "<Present> must be a direct child of <Graph>; it cannot be inside <{parent}>."
                    ),
                });
            }
            if !is_raw_self_closing_tag(tag) {
                return Err(GraphParseError {
                    line: line_of_byte(&normalized, tag_start),
                    message: "<Present> must be self-closing: <Present from=\"...\" />."
                        .to_string(),
                });
            }
            if let Some(non_comment_ix) =
                first_non_ws_or_comment(&normalized, tag_end + 1, graph_close)
            {
                return Err(GraphParseError {
                    line: line_of_byte(&normalized, non_comment_ix),
                    message:
                        "<Present ... /> must be the final node in <Graph>, immediately before </Graph>."
                            .to_string(),
                });
            }
            cursor = tag_end + 1;
            continue;
        }

        if !is_raw_self_closing_tag(tag) {
            stack.push(name.to_string());
        }
        cursor = tag_end + 1;
    }

    if present_count == 0 {
        return Err(GraphParseError {
            line: line_of_byte(&normalized, graph_start),
            message: "Missing <Present from=\"...\" /> node.".to_string(),
        });
    }

    Ok(())
}

fn find_open_tag_byte(input: &str, tag_name: &str, start: usize) -> Option<usize> {
    let pattern = format!("<{tag_name}");
    let mut cursor = start.min(input.len());
    while let Some(offset) = input[cursor..].find(&pattern) {
        let ix = cursor + offset;
        let next_ix = ix + pattern.len();
        let next = input[next_ix..].chars().next();
        if matches!(next, Some(ch) if ch.is_whitespace() || ch == '>' || ch == '/') {
            return Some(ix);
        }
        cursor = next_ix;
    }
    None
}

fn find_tag_end_byte(input: &str, start: usize) -> Option<usize> {
    let mut in_double_quote = false;
    let mut in_single_quote = false;
    let mut brace_depth = 0usize;
    for (offset, ch) in input[start..].char_indices() {
        match ch {
            '"' if !in_single_quote && brace_depth == 0 => in_double_quote = !in_double_quote,
            '\'' if !in_double_quote && brace_depth == 0 => in_single_quote = !in_single_quote,
            '{' if !in_double_quote && !in_single_quote => brace_depth += 1,
            '}' if !in_double_quote && !in_single_quote => {
                brace_depth = brace_depth.saturating_sub(1)
            }
            '>' if !in_double_quote && !in_single_quote && brace_depth == 0 => {
                return Some(start + offset);
            }
            _ => {}
        }
    }
    None
}

fn opening_tag_name(tag: &str) -> Option<&str> {
    let rest = tag.strip_prefix('<')?.trim_start();
    if rest.starts_with('/') || rest.starts_with('!') || rest.starts_with('?') {
        return None;
    }
    let end = rest
        .find(|ch: char| ch.is_whitespace() || ch == '>' || ch == '/')
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

fn closing_tag_name(tag: &str) -> Option<&str> {
    let rest = tag.strip_prefix("</")?.trim_start();
    let end = rest
        .find(|ch: char| ch.is_whitespace() || ch == '>')
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

fn is_raw_self_closing_tag(tag: &str) -> bool {
    tag.trim_end()
        .strip_suffix('>')
        .is_some_and(|body| body.trim_end().ends_with('/'))
}

fn first_non_ws_or_comment(input: &str, mut start: usize, end: usize) -> Option<usize> {
    while start < end {
        let rest = &input[start..end];
        let trimmed = rest.trim_start();
        start += rest.len() - trimmed.len();
        if start >= end {
            return None;
        }
        if input[start..end].starts_with("<!--") {
            let comment_start = start;
            let Some(rel_comment_end) = input[start + 4..end].find("-->") else {
                return Some(comment_start);
            };
            start = start + 4 + rel_comment_end + 3;
            continue;
        }
        return Some(start);
    }
    None
}

fn line_of_byte(input: &str, byte_ix: usize) -> usize {
    input[..byte_ix.min(input.len())]
        .bytes()
        .filter(|b| *b == b'\n')
        .count()
        + 1
}

pub fn parse_graph_script(input: &str) -> Result<GraphScript, GraphParseError> {
    const DEFAULT_GRAPH_DURATION_MS: u64 = 2_000;
    let normalized = input.replace('＝', "=");
    validate_graph_present_placement(&normalized)?;
    let lines: Vec<&str> = normalized.lines().collect();
    let Some(graph_start_ix) = lines
        .iter()
        .position(|line| line.trim_start().starts_with("<Graph"))
    else {
        return Err(GraphParseError {
            line: 0,
            message: "Missing <Graph ...> root tag.".to_string(),
        });
    };

    let (graph_open, graph_open_end_ix) = collect_tag_block(&lines, graph_start_ix, '>', false)?;
    let id = attr_value(&graph_open, "id").map(|v| strip_wrappers(&v).to_string());
    let version = attr_value(&graph_open, "version").map(|v| strip_wrappers(&v).to_string());
    if attr_value(&graph_open, "scope").is_some() {
        return Err(GraphParseError {
            line: graph_start_ix + 1,
            message: "Graph scope has been removed. Use unified <Graph fps={...} duration=\"...\" size={[w,h]}> syntax.".to_string(),
        });
    }
    let fps = parse_fps(&graph_open, graph_start_ix + 1)?;
    let apply = attr_value(&graph_open, "apply")
        .as_deref()
        .map(|v| parse_graph_apply_scope(v, graph_start_ix + 1, "apply"))
        .transpose()?
        .unwrap_or(GraphApplyScope::Clip);
    let duration_explicit = attr_value(&graph_open, "duration").is_some();
    let duration_ms =
        parse_duration_ms(&graph_open, graph_start_ix + 1, DEFAULT_GRAPH_DURATION_MS)?;
    let size = parse_size(
        &required_attr_value(&graph_open, "size", graph_start_ix + 1)?,
        graph_start_ix + 1,
        "size",
    )?;
    let render_size = attr_value(&graph_open, "renderSize")
        .as_deref()
        .map(|value| parse_size(value, graph_start_ix + 1, "renderSize"))
        .transpose()?;
    if let Some((0, _)) | Some((_, 0)) = render_size {
        return Err(GraphParseError {
            line: graph_start_ix + 1,
            message: "renderSize width and height must be greater than zero.".to_string(),
        });
    }

    let Some(graph_close_ix) = lines
        .iter()
        .enumerate()
        .skip(graph_open_end_ix + 1)
        .find(|(_, line)| line.trim_start().starts_with("</Graph>"))
        .map(|(ix, _)| ix)
    else {
        return Err(GraphParseError {
            line: graph_start_ix + 1,
            message: "Missing </Graph> closing tag.".to_string(),
        });
    };

    let mut inputs = Vec::<InputNode>::new();
    let mut assets = Vec::<GraphAssetNode>::new();
    let mut material_assets = Vec::<MaterialAssetNode>::new();
    let mut textures = Vec::<TexNode>::new();
    let mut buffers = Vec::<BufferNode>::new();
    let mut backgrounds = Vec::<BackgroundNode>::new();
    let mut texts = Vec::<TextNode>::new();
    let mut images = Vec::<ImageNode>::new();
    let mut svgs = Vec::<SvgNode>::new();
    let mut scenes = Vec::<SceneRootNode>::new();
    let mut render_styles = Vec::new();
    let mut scene_nodes = Vec::<SceneNode>::new();
    let mut model_profiles = Vec::<ModelProfileNode>::new();
    let mut skeletons = Vec::<SkeletonNode>::new();
    let mut actions = Vec::<ActionNode>::new();
    let mut action_libraries = Vec::<ActionLibraryNode>::new();
    let mut apply_actions = Vec::<ApplyActionNode>::new();
    let mut attachments = Vec::<AttachmentNode>::new();
    let mut contact_surfaces = Vec::<ContactSurfaceNode>::new();
    let mut scene_constraints = Vec::<SceneConstraintNode>::new();
    let mut audio_clips = Vec::new();
    let mut audio_targets = Vec::new();
    let mut animation_targets = Vec::<AnimationTargetNode>::new();
    let mut layers = Vec::<LayerNode>::new();
    let mut processes = Vec::<ProcessDefinitionNode>::new();
    let world_sources = Vec::<WorldSourceNode>::new();
    let mut outputs = Vec::<OutputNode>::new();
    let mut process_outputs = Vec::<OutputNode>::new();
    let mut passes = Vec::<PassNode>::new();
    let mut present: Option<PresentNode> = None;
    let mut brush_ctx = BrushParseContext::default();
    let mut i = graph_open_end_ix + 1;

    while i < graph_close_ix {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }

        if line.starts_with("<Input") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            inputs.push(parse_input_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Assets") {
            let (mut parsed_assets, mut parsed_materials, end_ix) = parse_assets_block(&lines, i)?;
            assets.append(&mut parsed_assets);
            material_assets.append(&mut parsed_materials);
            i = end_ix + 1;
            continue;
        }

        if line.starts_with("<Clip") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            inputs.push(parse_clip_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Defs") {
            let (defs, end_ix) = parse_defs_block(&lines, i, &mut brush_ctx)?;
            scene_nodes.push(SceneNode::Defs(defs));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "RenderQuality") {
            return Err(GraphParseError {
                line: i + 1,
                message: "RenderQuality is not supported".into(),
            });
        }

        if starts_open_tag(line, "RenderStyle") {
            let (value, end_ix) = crate::render_style::parse_resource(&lines, i)?;
            render_styles.push(serde_json::from_value(value).map_err(|e| GraphParseError {
                line: i + 1,
                message: e.to_string(),
            })?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Scene") {
            let (scene, end_ix) = parse_scene_root_block(&lines, i, &brush_ctx)?;
            scenes.push(scene);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "process") {
            return Err(GraphParseError {
                line: i + 1,
                message: "Use <Process> with an uppercase P. MotionLoom DSL tag names are case-sensitive.".to_string(),
            });
        }

        if starts_open_tag(line, "Process") {
            let (process_output, process_definition, process_body_start_ix) =
                parse_process_resource_alias(&lines, i)?;
            process_outputs.push(process_output);
            processes.push(process_definition);
            i = process_body_start_ix;
            continue;
        }

        if starts_open_tag(line, "World") {
            return Err(GraphParseError {
                line: i + 1,
                message: "<World> has been removed from MotionLoom DSL. Put 3D content in <Scene><Timeline><Track><Sequence><CompositeGroup space=\"3d\">...</CompositeGroup> and keep space=\"world\" only as a coordinate-space value.".to_string(),
            });
        }

        if starts_open_tag(line, "ModelProfile") {
            let (profile, end_ix) = parse_model_profile_block(&lines, i)?;
            model_profiles.push(profile);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "ActionLibrary") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            action_libraries.push(parse_action_library_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Action") {
            let (action, end_ix) = parse_action_block(&lines, i)?;
            actions.push(action);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Skeleton") {
            let (skeleton, end_ix) = parse_skeleton_block(&lines, i)?;
            skeletons.push(skeleton);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "ApplyAction") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            apply_actions.push(parse_apply_action_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Attachment") {
            let (attachment, end_ix) = parse_attachment_block(&lines, i)?;
            attachments.push(attachment);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "ContactSurface") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            contact_surfaces.push(parse_contact_surface_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Constraint") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_constraints.push(parse_scene_constraint_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        // Keep audio separate from visual animation property lowering.
        if starts_open_tag(line, "AudioClip") {
            let (tag, end) = collect_self_closing_block(&lines, i)?;
            audio_clips.push(audio_parser::clip(&tag, i + 1)?);
            i = end + 1;
            continue;
        }
        if starts_open_tag(line, "AudioTarget") {
            let (target, end) = audio_parser::target(&lines, i, fps)?;
            audio_targets.push(target);
            i = end + 1;
            continue;
        }
        if starts_open_tag(line, "AnimationTarget") {
            let (target, end_ix) = parse_animation_target_block(&lines, i, fps)?;
            animation_targets.push(target);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Solid") {
            return Err(GraphParseError {
                line: i + 1,
                message:
                    "<Solid> has been removed. Use top-level <Background color=\"...\" /> instead."
                        .to_string(),
            });
        }

        if starts_open_tag(line, "Background") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            backgrounds.push(parse_background_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "PixelGrid") {
            let (grid, end_ix) = parse_pixel_grid_block(&lines, i)?;
            scene_nodes.push(SceneNode::PixelGrid(grid));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Text") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            let node = parse_text_node(&tag, i + 1, None, Vec::new())?;
            scene_nodes.push(SceneNode::Text(Box::new(node.clone())));
            texts.push(node);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Image") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            let node = parse_image_node(&tag, i + 1)?;
            scene_nodes.push(SceneNode::Image(node.clone()));
            images.push(node);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Svg") || starts_open_tag(line, "SVG") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            let node = parse_svg_node(&tag, i + 1)?;
            scene_nodes.push(SceneNode::Svg(node.clone()));
            svgs.push(node);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Rect") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Rect(parse_rect_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Circle") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Circle(parse_circle_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Line") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Line(parse_line_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Polyline") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Polyline(parse_polyline_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Path") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Path(parse_path_node(&tag, i + 1, &brush_ctx)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "FaceJaw") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::FaceJaw(parse_face_jaw_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Shadow") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Shadow(parse_shadow_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Group") {
            let (group, end_ix) = parse_group_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Group(group));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Layout") {
            let (layout, end_ix) = parse_layout_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Group(layout));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Puppet") || starts_open_tag(line, "PuppetWarp") {
            let (puppet, end_ix) = parse_puppet_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Puppet(puppet));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Pin") || starts_open_tag(line, "PuppetPin") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            scene_nodes.push(SceneNode::Pin(parse_pin_node(&tag, i + 1)?));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "MeshTopology") {
            let (topology, end_ix) = parse_mesh_topology_block(&lines, i)?;
            scene_nodes.push(SceneNode::MeshTopology(topology));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Part") {
            let (part, end_ix) = parse_part_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Part(part));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Repeat") {
            let (repeat, end_ix) = parse_repeat_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(repeat);
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Mask") {
            let (mask, end_ix) = parse_mask_any(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Mask(mask));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Precompose") {
            let (precompose, end_ix) = parse_precompose_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Precompose(precompose));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Character") {
            let (character, end_ix) = parse_character_block(&lines, i, &brush_ctx)?;
            scene_nodes.push(SceneNode::Character(character));
            i = end_ix + 1;
            continue;
        }

        if starts_open_tag(line, "Camera") {
            let (tag, tag_end_ix) = collect_tag_block(&lines, i, '>', false)?;
            if is_self_closing_tag(&tag) {
                scene_nodes.push(SceneNode::Camera(parse_camera_node(
                    &tag,
                    i + 1,
                    Vec::new(),
                )?));
                i = tag_end_ix + 1;
            } else {
                let (camera, end_ix) = parse_camera_block(&lines, i, &brush_ctx)?;
                scene_nodes.push(SceneNode::Camera(camera));
                i = end_ix + 1;
            }
            continue;
        }

        if starts_open_tag(line, "Layer") {
            let (layer, end_ix) = parse_layer_block(&lines, i)?;
            layers.push(layer);
            i = end_ix + 1;
            continue;
        }

        if line.starts_with("<Tex") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            textures.push(parse_tex_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if line.starts_with("<Buffer") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            buffers.push(parse_buffer_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if line.starts_with("<Output") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            outputs.push(parse_output_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if line.starts_with("<Pass") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            passes.push(parse_pass_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        if line.starts_with("<Present") {
            let (tag, end_ix) = collect_self_closing_block(&lines, i)?;
            if present.is_some() {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "Only one <Present ... /> node is supported.".to_string(),
                });
            }
            present = Some(parse_present_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }

        i += 1;
    }
    outputs.extend(process_outputs);

    let present = present.ok_or_else(|| GraphParseError {
        line: graph_start_ix + 1,
        message: "Missing <Present from=\"...\" /> node.".to_string(),
    })?;

    lower_parametric_component_uses(&mut scene_nodes, &mut scenes)?;
    resolve_lowered_puppet_targets(&mut scene_nodes, &mut scenes)?;
    resolve_primitive_material_assets(&mut assets, &material_assets, graph_start_ix + 1)?;

    validate_graph(
        fps,
        duration_ms,
        size,
        &assets,
        &inputs,
        &textures,
        &buffers,
        &backgrounds,
        &texts,
        &images,
        &svgs,
        &scenes,
        &scene_nodes,
        &model_profiles,
        &skeletons,
        &actions,
        &action_libraries,
        &apply_actions,
        &attachments,
        &contact_surfaces,
        &scene_constraints,
        &layers,
        &world_sources,
        &outputs,
        &passes,
        &present,
        graph_start_ix + 1,
    )?;

    let mut graph = GraphScript {
        audio_clips,
        audio_targets,
        render_styles,
        raw_script: Some(input.to_string()),
        id: id.clone(),
        version,
        fps,
        apply,
        duration_ms,
        duration_explicit,
        size,
        render_size,
        assets,
        material_assets,
        inputs,
        textures,
        buffers,
        backgrounds,
        texts,
        images,
        svgs,
        scenes,
        scene_nodes,
        model_profiles,
        skeletons,
        actions,
        action_libraries,
        apply_actions,
        attachments,
        contact_surfaces,
        scene_constraints,
        animation_targets,
        layers,
        processes,
        world_sources,
        passes,
        outputs,
        present,
    };
    crate::audio::compile_audio_plan(&graph)?;
    crate::render_style::lower(&mut graph)?;
    crate::render_graph::compile_render_pass_dag(&graph)?;
    Ok(graph)
}

fn parse_contact_surface_node(
    block: &str,
    line: usize,
) -> Result<ContactSurfaceNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?)
        .trim()
        .to_string();
    let source = strip_wrappers(&required_attr_value(block, "source", line)?)
        .trim()
        .to_string();
    let kind = attr_value(block, "kind")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "surface".to_string());
    let plane = attr_value(block, "plane")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "top".to_string());
    let position = parse_optional_primitive_vec::<3>(block, "position", &id, line, false)?;
    let normal = attr_value(block, "normal")
        .map(|value| parse_primitive_vec_value::<3>(&value, "normal", &id, line, false))
        .transpose()?
        .unwrap_or([0.0, 1.0, 0.0]);
    let forward = attr_value(block, "forward")
        .map(|value| parse_primitive_vec_value::<3>(&value, "forward", &id, line, false))
        .transpose()?
        .unwrap_or([0.0, 0.0, 1.0]);
    let bounds = attr_value(block, "bounds")
        .map(|value| parse_primitive_vec_value::<2>(&value, "bounds", &id, line, true))
        .transpose()?
        .unwrap_or([1.0, 1.0]);
    let margin = attr_value(block, "margin")
        .map(|value| {
            strip_wrappers(&value)
                .parse::<f32>()
                .map_err(|_| GraphParseError {
                    line,
                    message: format!("ContactSurface '{id}' margin must be a finite number."),
                })
        })
        .transpose()?
        .unwrap_or(0.0);
    Ok(ContactSurfaceNode {
        id,
        source,
        kind,
        plane,
        position,
        normal,
        forward,
        bounds,
        margin,
    })
}

/// Parse one complete inline socket-to-bone binding without changing asset definitions.
fn parse_attachment_block(
    lines: &[&str],
    start: usize,
) -> Result<(AttachmentNode, usize), GraphParseError> {
    let (open, open_end) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open) {
        return Err(GraphParseError {
            line: start + 1,
            message: "Attachment must contain at least one Socket and one Attach child."
                .to_string(),
        });
    }
    let close = find_matching_close_tag(lines, open_end + 1, "Attachment")?;
    let id = strip_wrappers(&required_attr_value(&open, "id", start + 1)?)
        .trim()
        .to_string();
    let object = strip_wrappers(&required_attr_value(&open, "object", start + 1)?)
        .trim()
        .to_string();
    let mut sockets = Vec::new();
    let mut attaches = Vec::new();
    let mut index = open_end + 1;
    while index < close {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "Socket") {
            let (tag, end) = collect_self_closing_block(lines, index)?;
            sockets.push(parse_attachment_socket(&tag, index + 1)?);
            index = end + 1;
            continue;
        }
        if starts_open_tag(line, "Attach") {
            let (tag, end) = collect_self_closing_block(lines, index)?;
            attaches.push(parse_attachment_target(&tag, index + 1)?);
            index = end + 1;
            continue;
        }
        return Err(GraphParseError {
            line: index + 1,
            message: format!("Unsupported Attachment child in '{id}': {line}"),
        });
    }
    let default_socket = sockets
        .first()
        .map(|socket| socket.id.clone())
        .unwrap_or_default();
    for (index, attach) in attaches.iter_mut().enumerate() {
        if attach.socket.is_empty() {
            attach.socket.clone_from(&default_socket);
        }
        if attach.drive.is_empty() {
            attach.drive = if index == 0 { "object" } else { "target" }.to_string();
        }
    }
    Ok((
        AttachmentNode {
            id: id.clone(),
            object,
            sockets: (!sockets.is_empty())
                .then_some(sockets)
                .ok_or_else(|| GraphParseError {
                    line: start + 1,
                    message: format!("Attachment '{id}' is missing Socket."),
                })?,
            attaches: (!attaches.is_empty()).then_some(attaches).ok_or_else(|| {
                GraphParseError {
                    line: start + 1,
                    message: format!("Attachment '{id}' is missing Attach."),
                }
            })?,
        },
        close,
    ))
}

fn parse_attachment_vec3(
    block: &str,
    attr: &str,
    line: usize,
) -> Result<[String; 3], GraphParseError> {
    let raw = attr_value(block, attr)
        .map(|value| strip_wrappers(&value).trim().to_string())
        .unwrap_or_else(|| "[0,0,0]".to_string());
    let inner = raw
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{attr} must be a three-value array."),
        })?;
    let parts = inner.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err(GraphParseError {
            line,
            message: format!("{attr} must contain exactly three values."),
        });
    }
    Ok(std::array::from_fn(|axis| parts[axis].to_string()))
}

fn parse_attachment_socket(
    block: &str,
    line: usize,
) -> Result<AttachmentSocketNode, GraphParseError> {
    Ok(AttachmentSocketNode {
        id: strip_wrappers(&required_attr_value(block, "id", line)?)
            .trim()
            .to_string(),
        position: parse_attachment_vec3(block, "position", line)?,
        rotation: parse_attachment_vec3(block, "rotation", line)?,
    })
}

fn parse_attachment_target(
    block: &str,
    line: usize,
) -> Result<AttachmentTargetNode, GraphParseError> {
    let target = strip_wrappers(&required_attr_value(block, "target", line)?)
        .trim()
        .to_string();
    let Some((target_model, target_bone)) = target.split_once('.') else {
        return Err(GraphParseError {
            line,
            message: "Attach.target must use model-id.canonical-bone syntax.".to_string(),
        });
    };
    if target_model.is_empty() || target_bone.is_empty() || target_bone.contains('.') {
        return Err(GraphParseError {
            line,
            message: "Attach.target must use model-id.canonical-bone syntax.".to_string(),
        });
    }
    Ok(AttachmentTargetNode {
        socket: attr_value(block, "socket")
            .map(|value| strip_wrappers(&value).to_string())
            .unwrap_or_default(),
        target: target.clone(),
        target_model: target_model.to_string(),
        target_bone: target_bone.to_string(),
        mode: attr_value(block, "mode")
            .map(|value| strip_wrappers(&value).to_ascii_lowercase())
            .unwrap_or_else(|| "snap".to_string()),
        drive: attr_value(block, "drive")
            .map(|value| strip_wrappers(&value).to_ascii_lowercase())
            .unwrap_or_default(),
        position_weight: attr_value(block, "positionWeight")
            .map(|value| strip_wrappers(&value).to_string())
            .unwrap_or_else(|| "1".to_string()),
        rotation_weight: attr_value(block, "rotationWeight")
            .map(|value| strip_wrappers(&value).to_string())
            .unwrap_or_else(|| "1".to_string()),
        max_stretch: attr_value(block, "maxStretch")
            .map(|value| strip_wrappers(&value).to_string())
            .unwrap_or_else(|| "1.05".to_string()),
    })
}

fn parse_action_library_node(
    block: &str,
    line: usize,
) -> Result<ActionLibraryNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?)
        .trim()
        .to_string();
    let src = strip_wrappers(&required_attr_value(block, "src", line)?)
        .trim()
        .to_string();
    let actions = parse_string_array(
        &required_attr_value(block, "actions", line)?,
        line,
        "ActionLibrary.actions",
    )?;
    if id.is_empty() || id.contains('.') {
        return Err(GraphParseError {
            line,
            message: "ActionLibrary.id must be non-empty and cannot contain '.'.".to_string(),
        });
    }
    if src.is_empty() {
        return Err(GraphParseError {
            line,
            message: "ActionLibrary.src cannot be empty.".to_string(),
        });
    }
    let mut unique = HashSet::new();
    for action in &actions {
        if action.is_empty() || action.contains('.') {
            return Err(GraphParseError {
                line,
                message: "ActionLibrary action ids must be non-empty and cannot contain '.'."
                    .to_string(),
            });
        }
        if !unique.insert(action.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate ActionLibrary action selection: {action}"),
            });
        }
    }
    Ok(ActionLibraryNode { id, src, actions })
}

/// Parse a standalone `<ActionLibrary>` document produced by the Action Editor.
pub fn parse_action_library_document(input: &str) -> Result<Vec<ActionNode>, GraphParseError> {
    let normalized = input.replace('＝', "=");
    let lines = normalized.lines().collect::<Vec<_>>();
    let Some(start) = lines
        .iter()
        .position(|line| starts_open_tag(line.trim(), "ActionLibrary"))
    else {
        return Err(GraphParseError {
            line: 0,
            message: "Missing <ActionLibrary ...> root tag.".to_string(),
        });
    };
    let (_, open_end) = collect_tag_block(&lines, start, '>', false)?;
    let close = find_matching_close_tag(&lines, open_end + 1, "ActionLibrary")?;
    let mut actions = Vec::new();
    let mut ids = HashSet::new();
    let mut index = open_end + 1;
    while index < close {
        let line = lines[index].trim();
        if line.starts_with("<!--") {
            while index < close && !lines[index].contains("-->") {
                index += 1;
            }
            index += 1;
            continue;
        }
        if line.is_empty() || line.starts_with("//") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "Action") {
            let (action, end) = parse_action_block(&lines, index)?;
            if action.source.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "ActionLibrary Action {} must be authored inline; external AnimationAsset-backed Actions are not supported in ActionLibrary v1.",
                        action.id
                    ),
                });
            }
            if action.poses.is_empty() && action.iks.is_empty() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "ActionLibrary Action {} must contain at least one <Pose> or <IK />.",
                        action.id
                    ),
                });
            }
            if !ids.insert(action.id.clone()) {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("Duplicate Action id in ActionLibrary: {}", action.id),
                });
            }
            actions.push(action);
            index = end + 1;
            continue;
        }
        return Err(GraphParseError {
            line: index + 1,
            message: "ActionLibrary documents may only contain <Action> children.".to_string(),
        });
    }
    if actions.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: "ActionLibrary must contain at least one <Action>.".to_string(),
        });
    }
    Ok(actions)
}

#[path = "audio/dsl.rs"]
mod audio_parser;

fn parse_animation_target_block(
    lines: &[&str],
    start: usize,
    fps: f32,
) -> Result<(AnimationTargetNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "AnimationTarget")?;
    let node = strip_wrappers(&required_attr_value(&open_tag, "node", start + 1)?).to_string();
    let property =
        strip_wrappers(&required_attr_value(&open_tag, "property", start + 1)?).to_string();
    validate_animation_target_property(&property, start + 1)?;

    let mut keys = Vec::<AnimationKeyNode>::new();
    let mut i = open_end_ix + 1;
    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Key") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            keys.push(parse_animation_key_node(&tag, i + 1, fps)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<AnimationTarget> only accepts <Key /> children, got: {line}"),
        });
    }

    if keys.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: "<AnimationTarget> requires at least one <Key /> child.".to_string(),
        });
    }
    let descriptor = crate::scene::animation::animation_property_descriptor(&property)
        .expect("AnimationTarget property was validated before its keys");
    for key in &keys {
        crate::scene::animation::validate_animation_key_value(descriptor, &key.value).map_err(
            |reason| GraphParseError {
                line: start + 1,
                message: format!(
                    "AnimationTarget node={node:?} property={property:?} has invalid key value {:?}: {reason}.",
                    key.value
                ),
            },
        )?;
    }
    keys.sort_by(|a, b| {
        a.seconds
            .partial_cmp(&b.seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.frame.cmp(&b.frame))
    });
    Ok((
        AnimationTargetNode {
            node,
            property,
            keys,
        },
        close_ix,
    ))
}

fn parse_animation_key_node(
    block: &str,
    line: usize,
    fps: f32,
) -> Result<AnimationKeyNode, GraphParseError> {
    let frame_attr = attr_value(block, "frame");
    let time_attr = attr_value(block, "time");
    if frame_attr.is_some() && time_attr.is_some() {
        return Err(GraphParseError {
            line,
            message: "Key must use either frame or time, not both.".to_string(),
        });
    }
    let (frame, time, seconds) = if let Some(frame_attr) = frame_attr {
        let frame_raw = strip_wrappers(&frame_attr);
        let frame = frame_raw.parse::<u32>().map_err(|_| GraphParseError {
            line,
            message: format!("Key.frame must be a non-negative integer, got {frame_raw}."),
        })?;
        (frame, None, frame as f32 / fps.max(1.0))
    } else if let Some(time_attr) = time_attr {
        let time_raw = strip_wrappers(&time_attr).to_string();
        let seconds = parse_time_seconds(&time_raw, line, "Key.time")?;
        let frame = (seconds * fps.max(1.0)).round().max(0.0) as u32;
        (frame, Some(time_raw), seconds)
    } else {
        return Err(GraphParseError {
            line,
            message: "Key requires either frame=\"...\" or time=\"...\".".to_string(),
        });
    };
    let value = strip_wrappers(&required_attr_value(block, "value", line)?).to_string();
    let ease = attr_value(block, "ease")
        .as_deref()
        .map(strip_wrappers)
        .unwrap_or("linear")
        .to_string();
    Ok(AnimationKeyNode {
        frame,
        time,
        seconds,
        value,
        ease,
    })
}

fn validate_animation_target_property(property: &str, line: usize) -> Result<(), GraphParseError> {
    if crate::scene::animation::animation_property_descriptor(property).is_some() {
        Ok(())
    } else {
        Err(GraphParseError {
            line,
            message: format!("AnimationTarget.property is not registered: {property}."),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_graph(
    fps: f32,
    duration_ms: u64,
    size: (u32, u32),
    assets: &[GraphAssetNode],
    inputs: &[InputNode],
    textures: &[TexNode],
    buffers: &[BufferNode],
    backgrounds: &[BackgroundNode],
    texts: &[TextNode],
    images: &[ImageNode],
    svgs: &[SvgNode],
    scenes: &[SceneRootNode],
    scene_nodes: &[SceneNode],
    model_profiles: &[ModelProfileNode],
    skeletons: &[SkeletonNode],
    actions: &[ActionNode],
    action_libraries: &[ActionLibraryNode],
    apply_actions: &[ApplyActionNode],
    attachments: &[AttachmentNode],
    contact_surfaces: &[ContactSurfaceNode],
    scene_constraints: &[SceneConstraintNode],
    layers: &[LayerNode],
    world_sources: &[WorldSourceNode],
    outputs: &[OutputNode],
    passes: &[PassNode],
    present: &PresentNode,
    line: usize,
) -> Result<(), GraphParseError> {
    if !fps.is_finite() || fps <= 0.0 {
        return Err(GraphParseError {
            line,
            message: "fps must be a positive number.".to_string(),
        });
    }
    if duration_ms == 0 {
        return Err(GraphParseError {
            line,
            message: "duration must be greater than zero.".to_string(),
        });
    }
    if size.0 == 0 || size.1 == 0 {
        return Err(GraphParseError {
            line,
            message: "size width and height must be greater than zero.".to_string(),
        });
    }
    let has_scene_nodes = !backgrounds.is_empty()
        || !texts.is_empty()
        || !images.is_empty()
        || !svgs.is_empty()
        || !scenes.is_empty()
        || !scene_nodes.is_empty();
    if passes.is_empty()
        && !has_scene_nodes
        && skeletons.is_empty()
        && actions.is_empty()
        && apply_actions.is_empty()
        && world_sources.is_empty()
    {
        return Err(GraphParseError {
            line,
            message: "Graph requires at least one renderable node or <Pass ... /> node."
                .to_string(),
        });
    }

    let mut resource_ids = HashSet::<String>::new();
    if has_scene_nodes {
        resource_ids.insert("scene".to_string());
    }
    for scene in scenes {
        if !resource_ids.insert(scene.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", scene.id),
            });
        }
        let prefixed = format!("scene:{}", scene.id);
        if !resource_ids.insert(prefixed.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", prefixed),
            });
        }
    }
    let mut model_profile_ids = HashSet::<String>::new();
    for profile in model_profiles {
        if !matches!(profile.kind.as_str(), "2d" | "3d") {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ModelProfile {} kind must be \"2d\" or \"3d\", got: {}",
                    profile.id, profile.kind
                ),
            });
        }
        if !model_profile_ids.insert(profile.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate model profile id: {}", profile.id),
            });
        }
    }
    validate_scene_model_profile_refs(scenes, scene_nodes, &model_profile_ids, line)?;
    validate_scene_camera_structure(scenes, scene_nodes, line)?;
    let mut dynamic_rigid_body_targets = HashSet::new();
    for scene in scenes {
        collect_dynamic_rigid_body_targets(&scene.children, &mut dynamic_rigid_body_targets);
    }
    collect_dynamic_rigid_body_targets(scene_nodes, &mut dynamic_rigid_body_targets);
    validate_attachments(
        attachments,
        scenes,
        scene_nodes,
        &dynamic_rigid_body_targets,
        line,
    )?;

    let mut skeleton_ids = HashSet::<String>::new();
    for skeleton in skeletons {
        if !skeleton_ids.insert(skeleton.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate skeleton id: {}", skeleton.id),
            });
        }
        let mut bone_ids = HashSet::<String>::new();
        for bone in &skeleton.bones {
            if !bone_ids.insert(bone.id.clone()) {
                return Err(GraphParseError {
                    line,
                    message: format!("Duplicate bone id in skeleton {}: {}", skeleton.id, bone.id),
                });
            }
        }
        for bone in &skeleton.bones {
            if let Some(parent) = bone.parent.as_deref()
                && !bone_ids.contains(parent)
            {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Bone {} parent not found in skeleton {}: {}",
                        bone.id, skeleton.id, parent
                    ),
                });
            }
        }
        let report = crate::scene::domain::validate_skeleton(skeleton);
        if skeleton.validation.eq_ignore_ascii_case("strict") && report.has_errors() {
            let message = report
                .diagnostics
                .iter()
                .filter(|item| {
                    item.severity == crate::scene::domain::SkeletonDiagnosticSeverity::Error
                })
                .map(|item| item.message.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            return Err(GraphParseError {
                line,
                message: format!("Skeleton {} validation failed: {message}", skeleton.id),
            });
        }
    }

    for compound in assets.iter().filter_map(GraphAssetNode::compound) {
        let Some(rig_id) = compound.rig.as_deref() else {
            if compound.instances.iter().any(|instance| {
                instance.bone.is_some()
                    || instance.skin == NativeSkinMode::Smooth
                    || !instance.influences.is_empty()
            }) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "CompoundAsset {} has bone-bound Instances but does not declare rig.",
                        compound.id
                    ),
                });
            }
            continue;
        };
        let Some(skeleton) = skeletons.iter().find(|skeleton| skeleton.id == rig_id) else {
            return Err(GraphParseError {
                line,
                message: format!(
                    "CompoundAsset {} references unknown Skeleton rig: {rig_id}",
                    compound.id
                ),
            });
        };
        if skeleton.space != "3d" {
            return Err(GraphParseError {
                line,
                message: format!(
                    "CompoundAsset {} rig {} must use Skeleton space=\"3d\".",
                    compound.id, rig_id
                ),
            });
        }
        for instance in &compound.instances {
            if let Some(bone) = instance.bone.as_deref()
                && !skeleton.bones.iter().any(|candidate| candidate.id == bone)
            {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "CompoundAsset {} Instance {} references unknown bone {} in rig {}.",
                        compound.id, instance.id, bone, rig_id
                    ),
                });
            }
            if instance.skin == NativeSkinMode::Smooth && compound.skin_binding.is_none() {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "CompoundAsset {} Instance {} uses skin=\"smooth\" but the asset has no SkinBinding.",
                        compound.id, instance.id
                    ),
                });
            }
            if instance.skin == NativeSkinMode::Smooth && instance.bone.is_none() {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "CompoundAsset {} Instance {} uses skin=\"smooth\" but has no primary bone.",
                        compound.id, instance.id
                    ),
                });
            }
            if !instance.influences.is_empty() && instance.skin != NativeSkinMode::Smooth {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "CompoundAsset {} Instance {} declares influences but does not use skin=\"smooth\".",
                        compound.id, instance.id
                    ),
                });
            }
            for influence in &instance.influences {
                if !skeleton
                    .bones
                    .iter()
                    .any(|candidate| candidate.id == *influence)
                {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "CompoundAsset {} Instance {} references unknown influence bone {} in rig {}.",
                            compound.id, instance.id, influence, rig_id
                        ),
                    });
                }
            }
        }
        if let Some(binding) = &compound.skin_binding {
            for region in &binding.weight_regions {
                let Some(instance) = compound
                    .instances
                    .iter()
                    .find(|instance| instance.id == region.instance)
                else {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "CompoundAsset {} WeightRegion references unknown Instance {}.",
                            compound.id, region.instance
                        ),
                    });
                };
                if instance.skin != NativeSkinMode::Smooth {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "CompoundAsset {} WeightRegion targets rigid Instance {}.",
                            compound.id, region.instance
                        ),
                    });
                }
                if !skeleton
                    .bones
                    .iter()
                    .any(|candidate| candidate.id == region.bone)
                {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "CompoundAsset {} WeightRegion references unknown bone {} in rig {}.",
                            compound.id, region.bone, rig_id
                        ),
                    });
                }
            }
        }
    }

    let mut action_ids = HashSet::<String>::new();
    let animation_assets = assets
        .iter()
        .filter(|asset| asset.kind == GraphAssetKind::Animation)
        .map(|asset| (asset.id.as_str(), asset))
        .collect::<std::collections::HashMap<_, _>>();
    for action in actions {
        if !action_ids.insert(action.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate action id: {}", action.id),
            });
        }
        if let Some(source) = action.source.as_deref() {
            if !action.poses.is_empty() || !action.iks.is_empty() {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "External Action {} cannot contain <Pose> or <IK> children.",
                        action.id
                    ),
                });
            }
            if !animation_assets.contains_key(source) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Action {} references missing AnimationAsset source: {source}",
                        action.id
                    ),
                });
            }
        } else if action.poses.is_empty() && action.iks.is_empty() {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Action {} must contain at least one <Pose> or <IK />.",
                    action.id
                ),
            });
        }
        let mut contact_ids = HashSet::<String>::new();
        for contact in &action.contacts {
            if !contact_ids.insert(contact.id.clone()) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Duplicate Contact id '{}' in Action {}.",
                        contact.id, action.id
                    ),
                });
            }
            if !matches!(
                contact.effector.as_str(),
                "knee_l"
                    | "knee_r"
                    | "foot_l"
                    | "foot_r"
                    | "hand_l"
                    | "hand_r"
                    | "elbow_l"
                    | "elbow_r"
                    | "pelvis"
                    | "back"
            ) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Action {} Contact '{}' uses unsupported canonical effector '{}'. Use a canonical limb effector, pelvis, or back.",
                        action.id, contact.id, contact.effector
                    ),
                });
            }
            if !matches!(
                contact.target.as_str(),
                "ground" | "seat" | "support" | "wall" | "rail"
            ) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Action {} Contact '{}' uses unsupported semantic target: {}.",
                        action.id, contact.id, contact.target
                    ),
                });
            }
            if !matches!(contact.mode.as_str(), "lock" | "surface") {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Action {} Contact '{}' mode must be lock or surface, got: {}.",
                        action.id, contact.id, contact.mode
                    ),
                });
            }
            let weight = contact.weight.parse::<f32>().map_err(|_| GraphParseError {
                line,
                message: format!(
                    "Action {} Contact '{}' weight must be a number in 0..1, got: {}.",
                    action.id, contact.id, contact.weight
                ),
            })?;
            if !(0.0..=1.0).contains(&weight) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Action {} Contact '{}' weight must be inside 0..1, got: {}.",
                        action.id, contact.id, contact.weight
                    ),
                });
            }
        }
    }
    let mut contact_surface_ids = HashSet::new();
    for surface in contact_surfaces {
        if surface.id.is_empty() || !contact_surface_ids.insert(surface.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate or empty ContactSurface id: {}", surface.id),
            });
        }
        if !matches!(
            surface.kind.as_str(),
            "seat" | "support" | "wall" | "rail" | "surface"
        ) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ContactSurface {} has unsupported kind: {}",
                    surface.id, surface.kind
                ),
            });
        }
        if !matches!(surface.plane.as_str(), "top" | "explicit") {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ContactSurface {} plane must be top or explicit, got: {}",
                    surface.id, surface.plane
                ),
            });
        }
        if surface.plane == "explicit" && surface.position.is_none() {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ContactSurface {} plane=explicit requires position.",
                    surface.id
                ),
            });
        }
        let normal_length = surface
            .normal
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !normal_length.is_finite() || normal_length <= 0.0001 {
            return Err(GraphParseError {
                line,
                message: format!("ContactSurface {} normal must be non-zero.", surface.id),
            });
        }
    }
    let mut action_library_ids = HashSet::<String>::new();
    for library in action_libraries {
        if !action_library_ids.insert(library.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate ActionLibrary id: {}", library.id),
            });
        }
        for selected in &library.actions {
            let namespaced = format!("{}.{}", library.id, selected);
            if !action_ids.insert(namespaced.clone()) {
                return Err(GraphParseError {
                    line,
                    message: format!("Duplicate action id: {namespaced}"),
                });
            }
        }
    }
    for apply_action in apply_actions {
        if dynamic_rigid_body_targets.contains(&apply_action.target) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ApplyAction target '{}' is controlled by a dynamic RigidBody. Use type=\"kinematic\" for authored action motion, or remove ApplyAction so physics owns the transform.",
                    apply_action.target
                ),
            });
        }
        if !action_ids.contains(&apply_action.action) {
            if animation_assets.contains_key(apply_action.action.as_str()) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "ApplyAction.action '{}' references a raw AnimationAsset. Wrap it with <Action id=\"...\" source=\"{}\" sourceProfile=\"...\" clip=\"...\" /> and reference that Action id instead.",
                        apply_action.action, apply_action.action
                    ),
                });
            }
            return Err(GraphParseError {
                line,
                message: format!(
                    "ApplyAction target action not found: {}",
                    apply_action.action
                ),
            });
        }
        if let Some(root_motion) = apply_action.root_motion.as_deref()
            && !matches!(root_motion, "none" | "clip" | "in_place" | "match_target")
        {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ApplyAction rootMotion must be none, clip, in_place, or match_target, got: {root_motion}"
                ),
            });
        }
        if let Some(contact_correction) = apply_action.contact_correction.as_deref()
            && !matches!(contact_correction, "none" | "auto")
        {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ApplyAction contactCorrection must be none or auto, got: {contact_correction}"
                ),
            });
        }
        if apply_action.contact_correction.as_deref() == Some("auto")
            && apply_action.ground.is_none()
            && apply_action.contact_targets.is_empty()
        {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ApplyAction target '{}' uses contactCorrection=auto but has no ground or contactTargets binding.",
                    apply_action.target
                ),
            });
        }
        for (slot, surface_id) in &apply_action.contact_targets {
            if slot == "ground" {
                continue;
            }
            if !contact_surface_ids.contains(surface_id) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "ApplyAction target '{}' contactTargets.{slot} references unknown ContactSurface: {surface_id}",
                        apply_action.target
                    ),
                });
            }
        }
        if apply_action.contact_correction.as_deref() == Some("auto")
            && actions
                .iter()
                .find(|action| action.id == apply_action.action)
                .is_some_and(|action| action.contacts.is_empty())
        {
            return Err(GraphParseError {
                line,
                message: format!(
                    "ApplyAction target '{}' uses contactCorrection=auto, but Action '{}' has no <Contact /> declarations.",
                    apply_action.target, apply_action.action
                ),
            });
        }
    }
    for apply_action in apply_actions {
        let Some(group) = apply_action.sync_group.as_deref() else {
            continue;
        };
        for peer in apply_actions.iter().filter(|peer| {
            peer.sync_group.as_deref() == Some(group) && peer.target != apply_action.target
        }) {
            if peer.at_ms != apply_action.at_ms {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "ApplyAction syncGroup '{group}' members must use the same at time."
                    ),
                });
            }
            if peer.sync_marker != apply_action.sync_marker {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "ApplyAction syncGroup '{group}' members must use the same syncMarker."
                    ),
                });
            }
        }
    }
    for constraint in scene_constraints {
        if constraint.constraint_type != "position" {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Scene Constraint type must currently be position, got: {}",
                    constraint.constraint_type
                ),
            });
        }
        if constraint.solver != "two_bone_ik" {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Scene Constraint solver must currently be two_bone_ik, got: {}",
                    constraint.solver
                ),
            });
        }
        let Some((_, source_bone)) = constraint.source.rsplit_once('.') else {
            return Err(GraphParseError {
                line,
                message: "Scene Constraint source must use model-id.canonical-bone syntax."
                    .to_string(),
            });
        };
        if !matches!(
            source_bone,
            "head"
                | "forearm_l"
                | "forearm_r"
                | "hand_l"
                | "hand_r"
                | "lower_leg_l"
                | "lower_leg_r"
                | "foot_l"
                | "foot_r"
        ) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Scene Constraint two_bone_ik source must end in a supported humanoid head, elbow, hand, knee, or foot effector, got: {source_bone}"
                ),
            });
        }
        if constraint.target.rsplit_once('.').is_none() {
            return Err(GraphParseError {
                line,
                message: "Scene Constraint target must use model-id.canonical-bone syntax."
                    .to_string(),
            });
        }
    }
    for input in inputs {
        if !resource_ids.insert(input.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", input.id),
            });
        }
    }
    for tex in textures {
        if !resource_ids.insert(tex.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", tex.id),
            });
        }
    }
    for buf in buffers {
        if !resource_ids.insert(buf.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", buf.id),
            });
        }
    }
    for layer in layers {
        if !resource_ids.insert(layer.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", layer.id),
            });
        }
    }
    for world in world_sources {
        if !resource_ids.insert(world.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", world.id),
            });
        }
    }
    for output in outputs {
        if !resource_ids.insert(output.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate resource id: {}", output.id),
            });
        }
        if let Some(src) = &output.from
            && !resource_ids.contains(src)
        {
            return Err(GraphParseError {
                line,
                message: format!("Output {} source not found: {}", output.id, src),
            });
        }
    }

    let mut pass_ids = HashSet::<String>::new();
    for pass in passes {
        if !pass_ids.insert(pass.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate pass id: {}", pass.id),
            });
        }
        if pass.inputs.is_empty() {
            return Err(GraphParseError {
                line,
                message: format!("Pass {} must declare at least one input.", pass.id),
            });
        }
        if pass.outputs.is_empty() {
            return Err(GraphParseError {
                line,
                message: format!("Pass {} must declare at least one output.", pass.id),
            });
        }
        for tex_in in &pass.inputs {
            if !resource_ids.contains(tex_in.resource_id()) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Pass {} input resource not found: {}",
                        pass.id,
                        tex_in.resource_id()
                    ),
                });
            }
        }
        for tex_out in &pass.outputs {
            if !resource_ids.contains(tex_out.resource_id()) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Pass {} output resource not found: {}",
                        pass.id,
                        tex_out.resource_id()
                    ),
                });
            }
        }
    }

    if !resource_ids.contains(&present.from) {
        return Err(GraphParseError {
            line,
            message: format!("Present source resource not found: {}", present.from),
        });
    }

    Ok(())
}

fn collect_dynamic_rigid_body_targets(nodes: &[SceneNode], targets: &mut HashSet<String>) {
    for node in nodes {
        match node {
            SceneNode::Simulation(crate::simulation::model::SimulationBindingNode::RigidBody(
                body,
            )) if body.body_type == crate::simulation::model::RigidBodyType::Dynamic => {
                targets.insert(body.target.clone());
            }
            SceneNode::Group(group) => {
                if let Some(composite) = &group.composite {
                    for node in &composite.nodes_3d {
                        if let crate::scene::model::Scene3DNode::RigidBody(body) = node
                            && body.body_type == crate::simulation::model::RigidBodyType::Dynamic
                        {
                            targets.insert(body.target.clone());
                        }
                    }
                }
                collect_dynamic_rigid_body_targets(&group.children, targets);
            }
            SceneNode::Timeline(node) => {
                collect_dynamic_rigid_body_targets(&node.children, targets)
            }
            SceneNode::Track(node) => collect_dynamic_rigid_body_targets(&node.children, targets),
            SceneNode::Sequence(node) => {
                collect_dynamic_rigid_body_targets(&node.children, targets)
            }
            SceneNode::Layer(node) => collect_dynamic_rigid_body_targets(&node.children, targets),
            SceneNode::Part(node) => collect_dynamic_rigid_body_targets(&node.children, targets),
            _ => {}
        }
    }
}

/// Validate attachment ownership and dependency order before any renderer sees the graph.
fn validate_attachments(
    attachments: &[AttachmentNode],
    scenes: &[SceneRootNode],
    scene_nodes: &[SceneNode],
    dynamic_targets: &HashSet<String>,
    line: usize,
) -> Result<(), GraphParseError> {
    let mut model_ids = HashSet::new();
    for scene in scenes {
        collect_scene_model_ids(&scene.children, &mut model_ids);
    }
    collect_scene_model_ids(scene_nodes, &mut model_ids);
    let mut attachment_ids = HashSet::new();
    let mut owners = HashSet::new();
    let mut dependencies = HashMap::<String, String>::new();
    for attachment in attachments {
        if attachment.id.is_empty() || !attachment_ids.insert(attachment.id.clone()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate or empty Attachment id: {}", attachment.id),
            });
        }
        if !model_ids.contains(&attachment.object) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Attachment '{}' object Model not found: {}",
                    attachment.id, attachment.object
                ),
            });
        }
        if !owners.insert(attachment.object.clone()) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Model '{}' has more than one Attachment transform owner.",
                    attachment.object
                ),
            });
        }
        if dynamic_targets.contains(&attachment.object) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Attachment object '{}' is controlled by a dynamic RigidBody.",
                    attachment.object
                ),
            });
        }
        let socket_ids = attachment
            .sockets
            .iter()
            .map(|socket| socket.id.as_str())
            .collect::<HashSet<_>>();
        if socket_ids.len() != attachment.sockets.len() || socket_ids.contains("") {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Attachment '{}' has duplicate or empty Socket ids.",
                    attachment.id
                ),
            });
        }
        let primary = attachment
            .attaches
            .iter()
            .filter(|attach| attach.drive == "object")
            .collect::<Vec<_>>();
        if primary.len() != 1 || primary[0].mode != "snap" {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Attachment '{}' requires exactly one drive=object mode=snap Attach.",
                    attachment.id
                ),
            });
        }
        for attach in &attachment.attaches {
            if !socket_ids.contains(attach.socket.as_str()) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' references unknown Socket '{}'.",
                        attachment.id, attach.socket
                    ),
                });
            }
            if !model_ids.contains(&attach.target_model) || attachment.object == attach.target_model
            {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' has an invalid target Model '{}'.",
                        attachment.id, attach.target_model
                    ),
                });
            }
            if !matches!(attach.target_bone.as_str(), "hand_l" | "hand_r") {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' target must currently be hand_l or hand_r.",
                        attachment.id
                    ),
                });
            }
            if attach.drive == "target" && attach.mode != "ik" {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' drive=target requires mode=ik.",
                        attachment.id
                    ),
                });
            }
            if !matches!(attach.drive.as_str(), "object" | "target") {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' drive must be object or target.",
                        attachment.id
                    ),
                });
            }
            for (name, value) in [
                ("positionWeight", &attach.position_weight),
                ("rotationWeight", &attach.rotation_weight),
            ] {
                let parsed = value.parse::<f32>().map_err(|_| GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' {name} must be a number in 0..1.",
                        attachment.id
                    ),
                })?;
                if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "Attachment '{}' {name} must be inside 0..1.",
                            attachment.id
                        ),
                    });
                }
            }
            let max_stretch = attach
                .max_stretch
                .parse::<f32>()
                .map_err(|_| GraphParseError {
                    line,
                    message: format!(
                        "Attachment '{}' maxStretch must be a finite number >= 1.",
                        attachment.id
                    ),
                })?;
            if !max_stretch.is_finite() || max_stretch < 1.0 {
                return Err(GraphParseError {
                    line,
                    message: format!("Attachment '{}' maxStretch must be >= 1.", attachment.id),
                });
            }
        }
        for socket in &attachment.sockets {
            for values in [&socket.position, &socket.rotation] {
                if values
                    .iter()
                    .any(|value| !value.parse::<f32>().is_ok_and(f32::is_finite))
                {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "Attachment '{}' Socket values must contain finite numbers.",
                            attachment.id
                        ),
                    });
                }
            }
        }
        dependencies.insert(attachment.object.clone(), primary[0].target_model.clone());
    }
    for object in dependencies.keys() {
        let mut visited = HashSet::new();
        let mut current = object.as_str();
        while let Some(next) = dependencies.get(current) {
            if !visited.insert(current.to_string()) {
                return Err(GraphParseError {
                    line,
                    message: format!("Attachment cycle detected at Model '{current}'."),
                });
            }
            current = next;
        }
    }
    Ok(())
}

fn collect_scene_model_ids(nodes: &[SceneNode], ids: &mut HashSet<String>) {
    for node in nodes {
        match node {
            SceneNode::Group(group) => {
                if let Some(composite) = &group.composite {
                    for node in &composite.nodes_3d {
                        match node {
                            crate::scene::model::Scene3DNode::Model(model) => {
                                if let Some(id) = &model.id {
                                    ids.insert(id.clone());
                                }
                            }
                            crate::scene::model::Scene3DNode::VolumeRepeat(repeat) => {
                                if let Some(id) = &repeat.template.id {
                                    ids.insert(id.clone());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                collect_scene_model_ids(&group.children, ids);
            }
            SceneNode::Timeline(node) => collect_scene_model_ids(&node.children, ids),
            SceneNode::Track(node) => collect_scene_model_ids(&node.children, ids),
            SceneNode::Sequence(node) => collect_scene_model_ids(&node.children, ids),
            SceneNode::Layer(node) => collect_scene_model_ids(&node.children, ids),
            SceneNode::Part(node) => collect_scene_model_ids(&node.children, ids),
            _ => {}
        }
    }
}

fn parse_process_resource_alias(
    lines: &[&str],
    start: usize,
) -> Result<(OutputNode, ProcessDefinitionNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: "<Process> must contain process nodes.".to_string(),
        });
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Process")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let body = lines[open_end_ix + 1..close_ix].join("\n");
    let from = infer_process_output_resource(&open_tag, &body, start + 1)?;
    let definition = ProcessDefinitionNode {
        id: id.clone(),
        output: from.clone(),
        input_ids: collect_tag_attr_values(&body, "Input", "id")?,
        texture_ids: collect_tag_attr_values(&body, "Tex", "id")?,
        pass_ids: collect_tag_attr_values(&body, "Pass", "id")?,
    };
    Ok((
        OutputNode {
            id,
            from: Some(from),
            to: OutputTarget::Host,
            fmt: None,
            size: None,
            color_space: None,
            alpha: None,
            is_process_implicit: true,
        },
        definition,
        open_end_ix + 1,
    ))
}

fn parse_assets_block(
    lines: &[&str],
    start: usize,
) -> Result<(Vec<GraphAssetNode>, Vec<MaterialAssetNode>, usize), GraphParseError> {
    let (_open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Assets")?;
    let mut assets = Vec::new();
    let mut materials = Vec::new();
    let mut i = open_end_ix + 1;
    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with('{')
            || line.starts_with("<!--")
        {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "MaterialAsset") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            materials.push(parse_material_asset(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        let (kind, tag_name) = if starts_open_tag(line, "VideoAsset") {
            (GraphAssetKind::Video, "VideoAsset")
        } else if starts_open_tag(line, "ImageAsset") {
            (GraphAssetKind::Image, "ImageAsset")
        } else if starts_open_tag(line, "ModelAsset") {
            (GraphAssetKind::Model, "ModelAsset")
        } else if starts_open_tag(line, "AudioAsset") {
            (GraphAssetKind::Audio, "AudioAsset")
        } else if starts_open_tag(line, "AnimationAsset") {
            (GraphAssetKind::Animation, "AnimationAsset")
        } else if starts_open_tag(line, "PrimitiveAsset") {
            (GraphAssetKind::Model, "PrimitiveAsset")
        } else if starts_open_tag(line, "HairAsset") {
            (GraphAssetKind::Model, "HairAsset")
        } else if starts_open_tag(line, "MeshAsset") {
            (GraphAssetKind::Model, "MeshAsset")
        } else if starts_open_tag(line, "HeadAsset") {
            (GraphAssetKind::Model, "HeadAsset")
        } else if starts_open_tag(line, "TerrainAsset") {
            (GraphAssetKind::Model, "TerrainAsset")
        } else if starts_open_tag(line, "VegetationAsset") {
            (GraphAssetKind::Model, "VegetationAsset")
        } else if starts_open_tag(line, "CompoundAsset") {
            (GraphAssetKind::Model, "CompoundAsset")
        } else {
            return Err(GraphParseError {
                line: i + 1,
                message: format!(
                    "<Assets> only accepts <VideoAsset>, <ImageAsset>, <ModelAsset>, <PrimitiveAsset>, <HairAsset>, <HeadAsset>, <MeshAsset>, <TerrainAsset>, <VegetationAsset>, <CompoundAsset>, <MaterialAsset>, <AudioAsset>, or <AnimationAsset>, got: {line}"
                ),
            });
        };
        if tag_name == "CompoundAsset" {
            let (compound, end_ix) = parse_compound_asset(lines, i)?;
            assets.push(GraphAssetNode {
                id: compound.id.clone(),
                kind,
                source: GraphAssetSource::Compound(compound),
                decoder: None,
                color_space: None,
                profile: None,
                clip: None,
            });
            i = end_ix + 1;
            continue;
        }
        if tag_name == "PrimitiveAsset" {
            let (primitive, end_ix) = parse_primitive_asset_block(lines, i)?;
            assets.push(GraphAssetNode {
                id: primitive.id.clone(),
                kind,
                source: GraphAssetSource::Primitive(primitive),
                decoder: None,
                color_space: None,
                profile: None,
                clip: None,
            });
            i = end_ix + 1;
            continue;
        }
        if tag_name == "HairAsset" {
            let (hair, end_ix) = parse_hair_asset_block(lines, i)?;
            assets.push(GraphAssetNode {
                id: hair.id.clone(),
                kind,
                source: GraphAssetSource::Primitive(hair),
                decoder: None,
                color_space: None,
                profile: None,
                clip: None,
            });
            i = end_ix + 1;
            continue;
        }
        if tag_name == "HeadAsset" || tag_name == "MeshAsset" {
            let (head, end_ix) = if tag_name == "MeshAsset" {
                control_cage_parser::parse_mesh_asset(lines, i)?
            } else {
                parse_head_asset_block(lines, i)?
            };
            assets.push(GraphAssetNode {
                id: head.id.clone(),
                kind,
                source: GraphAssetSource::Primitive(head),
                decoder: None,
                color_space: None,
                profile: None,
                clip: None,
            });
            i = end_ix + 1;
            continue;
        }
        let (tag, end_ix) = collect_self_closing_block(lines, i)?;
        if !starts_open_tag(tag.trim(), tag_name) {
            return Err(GraphParseError {
                line: i + 1,
                message: format!("Invalid <{tag_name}> asset tag."),
            });
        }
        if kind == GraphAssetKind::Animation
            && (attr_value(&tag, "profile").is_some() || attr_value(&tag, "clip").is_some())
        {
            return Err(GraphParseError {
                line: i + 1,
                message: "AnimationAsset is a raw clip container and only owns id/src. Move profile/clip to an executable <Action source=\"...\" sourceProfile=\"...\" clip=\"...\" />."
                    .to_string(),
            });
        }
        let id = strip_wrappers(&required_attr_value(&tag, "id", i + 1)?).to_string();
        let source = if tag_name == "TerrainAsset" {
            GraphAssetSource::Terrain(parse_terrain_asset(&tag, &id, i + 1)?)
        } else if tag_name == "VegetationAsset" {
            GraphAssetSource::Vegetation(parse_vegetation_asset(&tag, &id, i + 1)?)
        } else {
            let src = strip_wrappers(&required_attr_value(&tag, "src", i + 1)?).to_string();
            if src.starts_with("motionloom:box:") {
                return Err(GraphParseError {
                    line: i + 1,
                    message: "motionloom:box shorthand has been removed. Declare <PrimitiveAsset shape=\"box\" size={...} color=\"...\" />.".to_string(),
                });
            }
            GraphAssetSource::External { src }
        };
        assets.push(GraphAssetNode {
            id,
            kind,
            source,
            decoder: attr_value(&tag, "decoder").map(|v| strip_wrappers(&v).to_string()),
            color_space: attr_value(&tag, "colorSpace")
                .or_else(|| attr_value(&tag, "color_space"))
                .map(|v| strip_wrappers(&v).to_string()),
            profile: attr_value(&tag, "profile").map(|v| strip_wrappers(&v).to_string()),
            clip: attr_value(&tag, "clip").map(|v| strip_wrappers(&v).to_string()),
        });
        i = end_ix + 1;
    }
    let mut ids = HashSet::new();
    for asset in &assets {
        if asset.id.trim().is_empty()
            || asset
                .external_src()
                .is_some_and(|src| src.trim().is_empty())
        {
            return Err(GraphParseError {
                line: start + 1,
                message: "Asset id and src must not be empty.".to_string(),
            });
        }
        if !ids.insert(asset.id.clone()) {
            return Err(GraphParseError {
                line: start + 1,
                message: format!("Duplicate Asset id: {}", asset.id),
            });
        }
    }
    for asset in &assets {
        let Some(compound) = asset.compound() else {
            continue;
        };
        for instance in &compound.instances {
            let Some(referenced) = assets.iter().find(|asset| asset.id == instance.asset) else {
                return Err(GraphParseError {
                    line: start + 1,
                    message: format!(
                        "CompoundAsset \"{}\" Instance \"{}\" references unknown asset \"{}\".",
                        compound.id, instance.id, instance.asset
                    ),
                });
            };
            if referenced.primitive().is_none() {
                return Err(GraphParseError {
                    line: start + 1,
                    message: format!(
                        "CompoundAsset \"{}\" Instance \"{}\" must reference a PrimitiveAsset in V1.",
                        compound.id, instance.id
                    ),
                });
            }
        }
    }
    let mut material_ids = HashSet::new();
    if let Some(duplicate) = materials
        .iter()
        .find(|material| !material_ids.insert(material.id.clone()))
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("Duplicate MaterialAsset id: {}", duplicate.id),
        });
    }
    Ok((assets, materials, close_ix))
}

fn parse_primitive_asset_block(
    lines: &[&str],
    start: usize,
) -> Result<(PrimitiveAssetNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let mut primitive = parse_primitive_asset(&open_tag, &id, start + 1)?;
    if is_self_closing_tag(&open_tag) {
        if matches!(
            primitive.geometry,
            PrimitiveGeometry::Loft { .. } | PrimitiveGeometry::Ribbon { .. }
        ) {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "PrimitiveAsset \"{id}\" requires a nested Loft or Ribbon geometry block."
                ),
            });
        }
        validate_primitive_build_budget(&primitive, start + 1)?;
        return Ok((primitive, open_end_ix));
    }

    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "PrimitiveAsset")?;
    let mut index = open_end_ix + 1;
    let mut saw_modifiers = false;
    let mut saw_mesh_build = false;
    let mut saw_lod = false;
    let mut saw_geometry_block = false;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "Modifiers") {
            if saw_modifiers {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "PrimitiveAsset \"{id}\" may declare only one <Modifiers> block."
                    ),
                });
            }
            let (modifiers, end_ix) = parse_primitive_modifiers(lines, index, &id)?;
            primitive.modifiers = modifiers;
            saw_modifiers = true;
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "MeshBuild") {
            if saw_mesh_build {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("PrimitiveAsset \"{id}\" may declare only one <MeshBuild />."),
                });
            }
            let (tag, end_ix) = collect_self_closing_block(lines, index)?;
            primitive.mesh_build = parse_primitive_mesh_build(&tag, &id, index + 1)?;
            saw_mesh_build = true;
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "LOD") {
            if saw_lod {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("PrimitiveAsset \"{id}\" may declare only one <LOD />."),
                });
            }
            let (tag, end_ix) = collect_self_closing_block(lines, index)?;
            primitive.lod = parse_primitive_lod(&tag, &id, index + 1)?;
            saw_lod = true;
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Loft") {
            if saw_geometry_block || !matches!(primitive.geometry, PrimitiveGeometry::Loft { .. }) {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "PrimitiveAsset \"{id}\" accepts one <Loft> block only when shape=\"loft\"."
                    ),
                });
            }
            let (geometry, end_ix) = parse_primitive_loft(lines, index, &id)?;
            primitive.geometry = geometry;
            saw_geometry_block = true;
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Ribbon") {
            if saw_geometry_block || !matches!(primitive.geometry, PrimitiveGeometry::Ribbon { .. })
            {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "PrimitiveAsset \"{id}\" accepts one <Ribbon> block only when shape=\"ribbon\"."
                    ),
                });
            }
            let (geometry, end_ix) = parse_primitive_ribbon(lines, index, &id)?;
            primitive.geometry = geometry;
            saw_geometry_block = true;
            index = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: index + 1,
            message: format!(
                "PrimitiveAsset \"{id}\" accepts its Loft/Ribbon geometry block, <Modifiers>, <MeshBuild />, or <LOD /> children, got: {line}"
            ),
        });
    }
    if matches!(
        primitive.geometry,
        PrimitiveGeometry::Loft { .. } | PrimitiveGeometry::Ribbon { .. }
    ) && !saw_geometry_block
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{id}\" requires a matching Loft or Ribbon geometry block."
            ),
        });
    }
    validate_primitive_build_budget(&primitive, start + 1)?;
    Ok((primitive, close_ix))
}

/// Parse a procedural head as a generic anatomy surface. FaceLayout is an
/// optional convenience block; custom and creature heads can use only fields.
fn parse_head_asset_block(
    lines: &[&str],
    start: usize,
) -> Result<(PrimitiveAssetNode, usize), GraphParseError> {
    let (tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: "HeadAsset requires a HeadShape child.".to_string(),
        });
    }
    validate_head_attributes(
        &tag,
        &[
            "id",
            "material",
            "archetype",
            "variant",
            "bindBone",
            "symmetry",
            "topology",
            "segments",
            "rings",
            "seed",
        ],
        "HeadAsset",
        start + 1,
    )?;
    let id = strip_wrappers(&required_attr_value(&tag, "id", start + 1)?).to_string();
    let material = strip_wrappers(&required_attr_value(&tag, "material", start + 1)?).to_string();
    let archetype = strip_wrappers(&required_attr_value(&tag, "archetype", start + 1)?)
        .trim()
        .to_ascii_lowercase();
    if archetype.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HeadAsset \"{id}\" archetype must not be empty."),
        });
    }
    let variant = attr_value(&tag, "variant").map(|raw| strip_wrappers(&raw).to_string());
    let bind_bone = attr_value(&tag, "bindBone").map(|raw| strip_wrappers(&raw).to_string());
    let symmetry = primitive_string_attribute(&tag, "symmetry", "x");
    if !matches!(symmetry.as_str(), "x" | "none") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HeadAsset \"{id}\" symmetry must be x or none."),
        });
    }
    let topology = primitive_string_attribute(&tag, "topology", "procedural");
    if !matches!(topology.as_str(), "procedural" | "facialcage" | "explicit") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "HeadAsset \"{id}\" topology must be procedural, facialCage, or explicit."
            ),
        });
    }
    if topology != "procedural"
        && (attr_value(&tag, "segments").is_some() || attr_value(&tag, "rings").is_some())
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "HeadAsset \"{id}\" segments/rings apply only to procedural topology; use FacialCage segments for facialCage topology."
            ),
        });
    }
    let segments = parse_optional_primitive_u32(&tag, "segments", &id, start + 1)?.unwrap_or(48);
    let rings = parse_optional_primitive_u32(&tag, "rings", &id, start + 1)?.unwrap_or(32);
    let maximum_resolution = 128;
    if !(12..=maximum_resolution).contains(&segments) || !(8..=maximum_resolution).contains(&rings)
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "HeadAsset \"{id}\" requires segments 12..{maximum_resolution} and rings 8..{maximum_resolution}."
            ),
        });
    }
    let material_seed = parse_optional_primitive_u64(&tag, "seed", &id, start + 1)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "HeadAsset")?;
    let mut shape = None;
    let mut face_layout = None;
    let mut morph = None;
    let mut facial_cage = None;
    let mut head_profile = Vec::new();
    let mut head_dome = None;
    let mut explicit_cage = None;
    let mut features = Vec::new();
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "HeadProfile") {
            if !head_profile.is_empty() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one HeadProfile."),
                });
            }
            let (profile, end_ix) = parse_head_profile(lines, index, &id)?;
            head_profile = profile;
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "HeadCage") {
            if explicit_cage.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one HeadCage."),
                });
            }
            let (cage, end_ix) = control_cage_parser::parse_head_cage(lines, index, &id)?;
            explicit_cage = Some(cage);
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "FaceLayout") {
            if face_layout.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one FaceLayout."),
                });
            }
            let (layout, end) = face_layout_parser::parse(lines, index, &id)?;
            face_layout = Some(layout);
            index = end + 1;
            continue;
        }
        let (child, end_ix) = collect_self_closing_block(lines, index)?;
        if starts_open_tag(line, "HeadShape") {
            if shape.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one HeadShape."),
                });
            }
            shape = Some(parse_head_shape(&child, &id, index + 1)?);
        } else if starts_open_tag(line, "FacialCage") {
            if facial_cage.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one FacialCage."),
                });
            }
            facial_cage = Some(parse_facial_cage(&child, &id, index + 1)?);
        } else if starts_open_tag(line, "HeadDome") {
            if head_dome.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one HeadDome."),
                });
            }
            head_dome = Some(parse_head_dome(&child, &id, index + 1)?);
        } else if starts_open_tag(line, "HeadMorph") {
            if morph.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HeadAsset \"{id}\" may contain one HeadMorph."),
                });
            }
            morph = Some(parse_head_morph(&child, &id, index + 1)?);
        } else if starts_open_tag(line, "HeadFeature") {
            features.push(parse_head_feature(&child, &id, index + 1)?);
        } else {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "HeadAsset \"{id}\" accepts HeadShape, FaceLayout, FacialCage, HeadProfile, HeadDome, HeadCage, HeadMorph, and HeadFeature children, got: {line}"
                ),
            });
        }
        index = end_ix + 1;
    }
    let shape = shape.ok_or_else(|| GraphParseError {
        line: start + 1,
        message: format!("HeadAsset \"{id}\" requires HeadShape."),
    })?;
    match topology.as_str() {
        "facialcage"
            if facial_cage.is_none() || head_profile.len() < 2 || face_layout.is_none() =>
        {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "HeadAsset \"{id}\" facialCage topology requires FacialCage, FaceLayout, and at least two HeadSection children."
                ),
            });
        }
        "explicit" if explicit_cage.is_none() => {
            return Err(GraphParseError {
                line: start + 1,
                message: format!("HeadAsset \"{id}\" explicit topology requires HeadCage."),
            });
        }
        "procedural" if facial_cage.is_some() || explicit_cage.is_some() => {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "HeadAsset \"{id}\" procedural topology cannot contain FacialCage or HeadCage."
                ),
            });
        }
        _ => {}
    }
    if topology == "facialcage" {
        let face = face_layout.as_ref().expect("validated facial cage layout");
        crate::world::primitive::validate_facial_layout(
            facial_cage.as_ref().expect("validated settings"),
            &head_profile,
            head_dome.as_ref(),
            face,
        )
        .map_err(|error| GraphParseError {
            line: start + 1,
            message: format!("HeadAsset \"{id}\": {error}"),
        })?;
        let min_y = head_profile.first().expect("validated profile").at;
        let max_y = head_dome
            .as_ref()
            .map(|dome| dome.top)
            .unwrap_or_else(|| head_profile.last().expect("validated profile").at);
        let max_half_width = head_profile
            .iter()
            .map(|section| section.width * 0.5)
            .fold(0.0_f32, f32::max);
        let positive = true;
        let inside = face.eyes.iter().all(|eye| {
            eye.position[0].abs() + eye.socket_width * 0.5 < max_half_width
                && eye.position[1] - eye.socket_height * 0.5 > min_y
                && eye.position[1] + eye.socket_height * 0.5 < max_y
        }) && face.mouths.iter().all(|mouth| {
            mouth.position[0].abs() + (mouth.width * 0.5 * 3.488_372).max(0.08) < max_half_width
                && mouth.position[1] - (mouth.opening * 0.5 * 14.285_714).max(0.04) > min_y
                && mouth.position[1] + (mouth.opening * 0.5 * 14.285_714).max(0.04) < max_y
        });
        if !positive || !inside {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "HeadAsset \"{id}\" facial features must have positive dimensions and fit inside HeadProfile/HeadDome bounds."
                ),
            });
        }
    }
    let mut feature_ids = HashSet::new();
    if let Some(duplicate) = features
        .iter()
        .find(|feature| !feature_ids.insert(feature.id.clone()))
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "HeadAsset \"{id}\" has duplicate HeadFeature id \"{}\".",
                duplicate.id
            ),
        });
    }
    let primitive = PrimitiveAssetNode {
        id,
        geometry: PrimitiveGeometry::HeadSurface {
            archetype,
            variant,
            bind_bone,
            symmetry,
            topology,
            segments,
            rings,
            head_shape: shape,
            face_layout,
            facial_cage,
            head_profile,
            head_dome,
            explicit_cage,
            features,
            morph: morph.unwrap_or_else(default_head_morph),
        },
        color: [1.0; 4],
        material: Some(material),
        material_definition: None,
        bevel_radius: 0.0,
        bevel_segments: 0,
        material_seed,
        collision: PrimitiveCollisionNode::default(),
        modifiers: Vec::new(),
        mesh_build: PrimitiveMeshBuildNode::default(),
        lod: PrimitiveLodNode::default(),
    };
    Ok((primitive, close_ix))
}

fn parse_facial_cage(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<FacialCageNode, GraphParseError> {
    validate_head_attributes(
        tag,
        &[
            "generatorVersion",
            "segments",
            "profileSegments",
            "samplesPerSection",
            "subdivision",
            "orbitalRings",
            "mouthRings",
            "preserveProfile",
            "uvMode",
        ],
        "FacialCage",
        line,
    )?;
    let positive = |name: &str, default: u32| {
        parse_optional_primitive_u32(tag, name, asset_id, line).map(|v| v.unwrap_or(default))
    };
    let preserve_profile = primitive_string_attribute(tag, "preserveProfile", "true");
    if !matches!(preserve_profile.as_str(), "true" | "false") {
        return Err(GraphParseError {
            line,
            message: format!(
                "HeadAsset \"{asset_id}\" FacialCage preserveProfile must be true or false."
            ),
        });
    }
    let result = FacialCageNode {
        generator_version: positive("generatorVersion", 1)?,
        segments: positive("segments", 96)?,
        profile_segments: positive("profileSegments", 96)?,
        samples_per_section: positive("samplesPerSection", 6)?,
        subdivision: positive("subdivision", 1)?,
        orbital_rings: positive("orbitalRings", 10)?,
        mouth_rings: positive("mouthRings", 8)?,
        preserve_profile: preserve_profile == "true",
        uv_mode: primitive_string_attribute(tag, "uvMode", "frontBack"),
    };
    if result.generator_version != 1
        || !(12..=256).contains(&result.segments)
        || !(12..=256).contains(&result.profile_segments)
        || !(1..=32).contains(&result.samples_per_section)
        || result.subdivision > 2
        || !(2..=32).contains(&result.orbital_rings)
        || !(2..=32).contains(&result.mouth_rings)
        || !matches!(result.uv_mode.as_str(), "fallbackxy" | "frontback")
    {
        return Err(GraphParseError {
            line,
            message: format!("HeadAsset \"{asset_id}\" has invalid FacialCage limits."),
        });
    }
    Ok(result)
}

fn parse_head_dome(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<HeadDomeNode, GraphParseError> {
    validate_head_attributes(
        tag,
        &[
            "start",
            "top",
            "centerDepth",
            "frontRadius",
            "backRadius",
            "samples",
        ],
        "HeadDome",
        line,
    )?;
    let result = HeadDomeNode {
        start: parse_finite_primitive_number(tag, "start", asset_id, line)?,
        top: parse_finite_primitive_number(tag, "top", asset_id, line)?,
        center_depth: parse_finite_primitive_number(tag, "centerDepth", asset_id, line)?,
        front_radius: parse_positive_primitive_number(tag, "frontRadius", asset_id, line)?,
        back_radius: parse_positive_primitive_number(tag, "backRadius", asset_id, line)?,
        samples: parse_optional_primitive_u32(tag, "samples", asset_id, line)?.unwrap_or(80),
    };
    if result.top <= result.start || !(4..=256).contains(&result.samples) {
        return Err(GraphParseError {
            line,
            message: format!(
                "HeadAsset \"{asset_id}\" HeadDome requires top > start and samples 4..256."
            ),
        });
    }
    Ok(result)
}

fn parse_head_profile(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(Vec<HeadSectionNode>, usize), GraphParseError> {
    let (tag, open) = collect_tag_block(lines, start, '>', false)?;
    validate_head_attributes(&tag, &[], "HeadProfile", start + 1)?;
    let end = find_matching_close_tag(lines, open + 1, "HeadProfile")?;
    let mut sections = Vec::new();
    let mut index = open + 1;
    while index < end {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("<!--") || line.starts_with("//") {
            index += 1;
            continue;
        }
        let (child, child_end) = collect_self_closing_block(lines, index)?;
        if !starts_open_tag(line, "HeadSection") {
            return Err(GraphParseError {
                line: index + 1,
                message: "HeadProfile accepts HeadSection children only.".into(),
            });
        }
        validate_head_attributes(
            &child,
            &["id", "at", "width", "frontDepth", "backDepth"],
            "HeadSection",
            index + 1,
        )?;
        sections.push(HeadSectionNode {
            id: strip_wrappers(&required_attr_value(&child, "id", index + 1)?).into(),
            at: parse_finite_primitive_number(&child, "at", asset_id, index + 1)?,
            width: parse_positive_primitive_number(&child, "width", asset_id, index + 1)?,
            front_depth: parse_finite_primitive_number(&child, "frontDepth", asset_id, index + 1)?,
            back_depth: parse_finite_primitive_number(&child, "backDepth", asset_id, index + 1)?,
        });
        index = child_end + 1;
    }
    if sections.windows(2).any(|pair| pair[0].at >= pair[1].at)
        || sections
            .iter()
            .any(|section| section.front_depth <= section.back_depth)
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "HeadAsset \"{asset_id}\" HeadSections must ascend and keep frontDepth > backDepth."
            ),
        });
    }
    Ok((sections, end))
}

fn parse_head_shape(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<HeadShapeNode, GraphParseError> {
    validate_head_attributes(
        tag,
        &[
            "size",
            "forehead",
            "cheekWidth",
            "jawWidth",
            "chinLength",
            "chinRoundness",
        ],
        "HeadShape",
        line,
    )?;
    Ok(HeadShapeNode {
        size: parse_primitive_vec::<3>(tag, "size", asset_id, line)?,
        forehead: parse_optional_positive_primitive_number(tag, "forehead", asset_id, line)?
            .unwrap_or(1.0),
        cheek_width: parse_optional_positive_primitive_number(tag, "cheekWidth", asset_id, line)?
            .unwrap_or(1.0),
        jaw_width: parse_optional_positive_primitive_number(tag, "jawWidth", asset_id, line)?
            .unwrap_or(0.78),
        chin_length: parse_optional_nonnegative_primitive_number(
            tag,
            "chinLength",
            asset_id,
            line,
        )?
        .unwrap_or(0.08),
        chin_roundness: parse_unit_head_number(tag, "chinRoundness", asset_id, line, 0.7)?,
    })
}

fn parse_head_feature(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<HeadFeatureNode, GraphParseError> {
    validate_head_attributes(
        tag,
        &[
            "id", "kind", "center", "size", "amount", "offset", "falloff", "mirror",
        ],
        "HeadFeature",
        line,
    )?;
    let id = strip_wrappers(&required_attr_value(tag, "id", line)?).to_string();
    let kind = strip_wrappers(&required_attr_value(tag, "kind", line)?)
        .trim()
        .to_ascii_lowercase();
    if kind.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("HeadAsset \"{asset_id}\" HeadFeature kind must not be empty."),
        });
    }
    let falloff = primitive_string_attribute(tag, "falloff", "smooth");
    if !matches!(falloff.as_str(), "smooth" | "linear" | "sharp") {
        return Err(GraphParseError {
            line,
            message: format!(
                "HeadAsset \"{asset_id}\" HeadFeature falloff must be smooth, linear, or sharp."
            ),
        });
    }
    let mirror = primitive_string_attribute(tag, "mirror", "none");
    if !matches!(mirror.as_str(), "none" | "x") {
        return Err(GraphParseError {
            line,
            message: format!("HeadAsset \"{asset_id}\" HeadFeature mirror must be none or x."),
        });
    }
    Ok(HeadFeatureNode {
        id,
        kind,
        center: parse_optional_primitive_vec::<3>(tag, "center", asset_id, line, false)?
            .ok_or_else(|| GraphParseError {
                line,
                message: format!("HeadAsset \"{asset_id}\" HeadFeature requires center."),
            })?,
        size: parse_primitive_vec::<3>(tag, "size", asset_id, line)?,
        amount: parse_head_number(tag, "amount", asset_id, line, 0.0)?,
        offset: parse_optional_primitive_vec::<3>(tag, "offset", asset_id, line, false)?
            .unwrap_or([0.0; 3]),
        falloff,
        mirror_x: mirror == "x",
    })
}

fn parse_head_morph(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<HeadMorphNode, GraphParseError> {
    validate_head_attributes(
        tag,
        &[
            "headWidth",
            "headHeight",
            "headDepth",
            "faceWidth",
            "faceHeight",
            "jawWidth",
            "muzzleLength",
            "featureScale",
        ],
        "HeadMorph",
        line,
    )?;
    let mut value = default_head_morph();
    for (attribute, target) in [
        ("headWidth", &mut value.head_width),
        ("headHeight", &mut value.head_height),
        ("headDepth", &mut value.head_depth),
        ("faceWidth", &mut value.face_width),
        ("faceHeight", &mut value.face_height),
        ("jawWidth", &mut value.jaw_width),
        ("muzzleLength", &mut value.muzzle_length),
        ("featureScale", &mut value.feature_scale),
    ] {
        if let Some(parsed) =
            parse_optional_positive_primitive_number(tag, attribute, asset_id, line)?
        {
            *target = parsed;
        }
    }
    Ok(value)
}

fn default_head_morph() -> HeadMorphNode {
    HeadMorphNode {
        head_width: 1.0,
        head_height: 1.0,
        head_depth: 1.0,
        face_width: 1.0,
        face_height: 1.0,
        jaw_width: 1.0,
        muzzle_length: 1.0,
        feature_scale: 1.0,
    }
}

fn parse_head_number(
    tag: &str,
    attribute: &str,
    asset_id: &str,
    line: usize,
    default: f32,
) -> Result<f32, GraphParseError> {
    attr_value(tag, attribute)
        .map(|_| parse_finite_primitive_number(tag, attribute, asset_id, line))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_unit_head_number(
    tag: &str,
    attribute: &str,
    asset_id: &str,
    line: usize,
    default: f32,
) -> Result<f32, GraphParseError> {
    let value = parse_head_number(tag, attribute, asset_id, line, default)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(GraphParseError {
            line,
            message: format!("HeadAsset \"{asset_id}\" {attribute} must be from zero through one."),
        });
    }
    Ok(value)
}

fn validate_head_attributes(
    tag: &str,
    allowed: &[&str],
    tag_name: &str,
    line: usize,
) -> Result<(), GraphParseError> {
    if let Some(attribute) = tag_attribute_names(tag)
        .into_iter()
        .find(|attribute| !allowed.contains(&attribute.as_str()))
    {
        return Err(GraphParseError {
            line,
            message: format!("<{tag_name}> does not support attribute \"{attribute}\"."),
        });
    }
    Ok(())
}

#[derive(Debug)]
struct HairCardParseSettings {
    id: String,
    length_segments: u32,
    width_segments: u32,
    thickness: f32,
    cross_section: String,
    tip_shape: String,
}

/// Parse guide-authored hair separately from PrimitiveAsset so future strand
/// representations can reuse the same groom without changing authored guides.
fn parse_hair_asset_block(
    lines: &[&str],
    start: usize,
) -> Result<(PrimitiveAssetNode, usize), GraphParseError> {
    let (tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: "HairAsset requires HairGroom and HairRepresentations children.".to_string(),
        });
    }
    validate_hair_attributes(
        &tag,
        &[
            "id",
            "material",
            "bindBone",
            "space",
            "defaultRepresentation",
            "seed",
        ],
        "HairAsset",
        start + 1,
    )?;
    let id = strip_wrappers(&required_attr_value(&tag, "id", start + 1)?).to_string();
    let material = strip_wrappers(&required_attr_value(&tag, "material", start + 1)?).to_string();
    let bind_bone = attr_value(&tag, "bindBone").map(|raw| strip_wrappers(&raw).to_string());
    let space = attr_value(&tag, "space")
        .map(|raw| strip_wrappers(&raw).to_ascii_lowercase())
        .unwrap_or_else(|| "bone_local".to_string());
    if !matches!(space.as_str(), "bone_local" | "asset_local") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HairAsset \"{id}\" space must be bone_local or asset_local."),
        });
    }
    let default_representation =
        attr_value(&tag, "defaultRepresentation").map(|raw| strip_wrappers(&raw).to_string());
    let material_seed = parse_optional_primitive_u64(&tag, "seed", &id, start + 1)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "HairAsset")?;
    let mut guides = None;
    let mut cards = None;
    let mut lod_representation = None;
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "HairGroom") {
            if guides.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HairAsset \"{id}\" may contain one HairGroom."),
                });
            }
            let (parsed, end_ix) = parse_hair_groom(lines, index, &id)?;
            guides = Some(parsed);
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "HairRepresentations") {
            if cards.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("HairAsset \"{id}\" may contain one HairRepresentations."),
                });
            }
            let (parsed, end_ix) = parse_hair_representations(lines, index, &id)?;
            cards = Some(parsed);
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "HairLOD") {
            let (lod_tag, end_ix) = collect_self_closing_block(lines, index)?;
            validate_hair_attributes(&lod_tag, &["representation"], "HairLOD", index + 1)?;
            lod_representation = Some(
                strip_wrappers(&required_attr_value(&lod_tag, "representation", index + 1)?)
                    .to_string(),
            );
            index = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: index + 1,
            message: format!(
                "HairAsset \"{id}\" accepts HairGroom, HairRepresentations, and HairLOD children, got: {line}"
            ),
        });
    }
    let guides = guides.ok_or_else(|| GraphParseError {
        line: start + 1,
        message: format!("HairAsset \"{id}\" requires HairGroom."),
    })?;
    let cards = cards.ok_or_else(|| GraphParseError {
        line: start + 1,
        message: format!("HairAsset \"{id}\" requires a HairCards representation."),
    })?;
    for selected in [default_representation.as_ref(), lod_representation.as_ref()]
        .into_iter()
        .flatten()
    {
        if selected != &cards.id {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "HairAsset \"{id}\" selects unknown representation \"{selected}\"; V1 provides \"{}\".",
                    cards.id
                ),
            });
        }
    }
    let primitive = PrimitiveAssetNode {
        id,
        geometry: PrimitiveGeometry::HairCards {
            representation_id: cards.id,
            bind_bone,
            space,
            length_segments: cards.length_segments,
            width_segments: cards.width_segments,
            thickness: cards.thickness,
            cross_section: cards.cross_section,
            tip_shape: cards.tip_shape,
            guides,
        },
        color: [1.0; 4],
        material: Some(material),
        material_definition: None,
        bevel_radius: 0.0,
        bevel_segments: 0,
        material_seed,
        collision: PrimitiveCollisionNode::default(),
        modifiers: Vec::new(),
        mesh_build: PrimitiveMeshBuildNode::default(),
        lod: PrimitiveLodNode::default(),
    };
    Ok((primitive, close_ix))
}

fn parse_hair_groom(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(Vec<HairGuideNode>, usize), GraphParseError> {
    let (tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    validate_hair_attributes(&tag, &[], "HairGroom", start + 1)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "HairGroom")?;
    let mut guides = Vec::new();
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if !starts_open_tag(line, "HairGroup") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("HairAsset \"{asset_id}\" HairGroom accepts HairGroup children."),
            });
        }
        let (mut group_guides, end_ix) = parse_hair_group(lines, index, asset_id)?;
        guides.append(&mut group_guides);
        index = end_ix + 1;
    }
    if guides.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HairAsset \"{asset_id}\" HairGroom requires at least one HairGuide."),
        });
    }
    let mut ids = HashSet::new();
    if let Some(duplicate) = guides.iter().find(|guide| !ids.insert(guide.id.as_str())) {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "HairAsset \"{asset_id}\" has duplicate HairGuide id \"{}\".",
                duplicate.id
            ),
        });
    }
    Ok((guides, close_ix))
}

fn parse_hair_group(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(Vec<HairGuideNode>, usize), GraphParseError> {
    let (tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    validate_hair_attributes(&tag, &["id", "role"], "HairGroup", start + 1)?;
    let group = strip_wrappers(&required_attr_value(&tag, "id", start + 1)?).to_string();
    let role = attr_value(&tag, "role")
        .map(|raw| strip_wrappers(&raw).to_string())
        .unwrap_or_else(|| "detail".to_string());
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "HairGroup")?;
    let mut guides = Vec::new();
    let mut defaults = HairPointNode {
        position: [0.0; 3],
        width: 0.1,
        radius: 0.003,
        camber: 0.0,
        roll: 0.0,
        stiffness: 1.0,
    };
    let mut has_defaults = false;
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "HairDefaults") {
            if has_defaults || !guides.is_empty() {
                return Err(GraphParseError {
                    line: index + 1,
                    message:
                        "HairDefaults must appear once before guides or mirrors in its HairGroup."
                            .into(),
                });
            }
            let (tag, end) = collect_self_closing_block(lines, index)?;
            validate_hair_attributes(
                &tag,
                &["width", "radius", "camber", "roll", "stiffness"],
                "HairDefaults",
                index + 1,
            )?;
            defaults = parse_hair_point_settings(&tag, &defaults, asset_id, index + 1)?;
            has_defaults = true;
            index = end + 1;
            continue;
        }
        if starts_open_tag(line, "HairMirror") {
            let (tag, end) = collect_self_closing_block(lines, index)?;
            validate_hair_attributes(&tag, &["id", "source", "axis"], "HairMirror", index + 1)?;
            let id = strip_wrappers(&required_attr_value(&tag, "id", index + 1)?).to_string();
            let source =
                strip_wrappers(&required_attr_value(&tag, "source", index + 1)?).to_string();
            let axis = required_attr_value(&tag, "axis", index + 1)?;
            let axis = match strip_wrappers(&axis) {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => {
                    return Err(GraphParseError {
                        line: index + 1,
                        message: "HairMirror axis must be x, y, or z.".into(),
                    });
                }
            };
            let original: &HairGuideNode = guides.iter().find(|g: &&HairGuideNode| g.id == source)
                .ok_or_else(|| GraphParseError { line: index + 1, message: format!("HairMirror {id:?} source {source:?} must name an earlier guide or mirror in the same HairGroup.") })?;
            let mut mirrored = original.clone();
            mirrored.id = id;
            // Reflect a resolved root frame as well as positions; negating roll
            // compensates for reflection reversing handedness.
            let a = original.points[0].position;
            let b = original.points[1].position;
            let t = hair_unit(std::array::from_fn(|i| b[i] - a[i]));
            let reference = if t[1].abs() < 0.92 {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let mut normal = original
                .normal
                .unwrap_or_else(|| hair_unit(hair_cross(t, hair_unit(hair_cross(reference, t)))));
            normal[axis] = -normal[axis];
            mirrored.normal = Some(normal);
            for point in &mut mirrored.points {
                point.position[axis] = -point.position[axis];
                point.roll = -point.roll;
            }
            guides.push(mirrored);
            index = end + 1;
            continue;
        }
        if !starts_open_tag(line, "HairGuide") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "HairGroup \"{group}\" accepts HairDefaults, HairGuide, and HairMirror children."
                ),
            });
        }
        let (guide, end_ix) = parse_hair_guide(lines, index, asset_id, &group, &role, &defaults)?;
        guides.push(guide);
        index = end_ix + 1;
    }
    if guides.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HairGroup \"{group}\" requires at least one HairGuide."),
        });
    }
    Ok((guides, close_ix))
}

// Small local vector helpers keep authoring independent of a rendering math backend.
fn hair_unit(v: [f32; 3]) -> [f32; 3] {
    let length = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0e-20);
    v.map(|x| x / length)
}
fn hair_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Resolve shared values before parsing points, so expanded guides are self-contained.
fn parse_hair_point_settings(
    tag: &str,
    inherited: &HairPointNode,
    asset_id: &str,
    line: usize,
) -> Result<HairPointNode, GraphParseError> {
    let mut result = inherited.clone();
    for (name, slot) in [
        ("width", &mut result.width),
        ("radius", &mut result.radius),
        ("camber", &mut result.camber),
        ("roll", &mut result.roll),
        ("stiffness", &mut result.stiffness),
    ] {
        if attr_value(tag, name).is_some() {
            *slot = parse_finite_primitive_number(tag, name, asset_id, line)?;
        }
        let valid = match name {
            "width" => *slot > 0.0,
            "radius" => *slot >= 0.0,
            "camber" | "stiffness" => (0.0..=1.0).contains(slot),
            _ => true,
        };
        if !valid {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Hair setting {name} is outside its allowed range (width > 0, radius >= 0, camber/stiffness 0..1)."
                ),
            });
        }
    }
    Ok(result)
}

fn parse_hair_guide(
    lines: &[&str],
    start: usize,
    asset_id: &str,
    group: &str,
    role: &str,
    defaults: &HairPointNode,
) -> Result<(HairGuideNode, usize), GraphParseError> {
    let (tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    validate_hair_attributes(
        &tag,
        &[
            "id",
            "normal",
            "width",
            "radius",
            "camber",
            "roll",
            "stiffness",
        ],
        "HairGuide",
        start + 1,
    )?;
    let id = strip_wrappers(&required_attr_value(&tag, "id", start + 1)?).to_string();
    let defaults = parse_hair_point_settings(&tag, defaults, asset_id, start + 1)?;
    let normal = parse_optional_primitive_vec::<3>(&tag, "normal", asset_id, start + 1, false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "HairGuide")?;
    let mut points = Vec::new();
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if !starts_open_tag(line, "HairPoint") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("HairGuide \"{id}\" accepts HairPoint children."),
            });
        }
        let (point_tag, end_ix) = collect_self_closing_block(lines, index)?;
        validate_hair_attributes(
            &point_tag,
            &["position", "width", "radius", "camber", "roll", "stiffness"],
            "HairPoint",
            index + 1,
        )?;
        let width =
            parse_optional_positive_primitive_number(&point_tag, "width", asset_id, index + 1)?
                .unwrap_or(defaults.width);
        let radius =
            parse_optional_nonnegative_primitive_number(&point_tag, "radius", asset_id, index + 1)?
                .unwrap_or(defaults.radius);
        let camber =
            parse_optional_nonnegative_primitive_number(&point_tag, "camber", asset_id, index + 1)?
                .unwrap_or(defaults.camber);
        if camber > 1.0 {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("HairGuide \"{id}\" camber must be from zero through one."),
            });
        }
        let roll = attr_value(&point_tag, "roll")
            .map(|_| parse_finite_primitive_number(&point_tag, "roll", asset_id, index + 1))
            .transpose()?
            .unwrap_or(defaults.roll);
        let stiffness = parse_optional_nonnegative_primitive_number(
            &point_tag,
            "stiffness",
            asset_id,
            index + 1,
        )?
        .unwrap_or(defaults.stiffness);
        if stiffness > 1.0 {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("HairGuide \"{id}\" stiffness must be from zero through one."),
            });
        }
        points.push(HairPointNode {
            position: parse_primitive_vec_value::<3>(
                &required_attr_value(&point_tag, "position", index + 1)?,
                "position",
                asset_id,
                index + 1,
                false,
            )?,
            width,
            radius,
            camber,
            roll,
            stiffness,
        });
        index = end_ix + 1;
    }
    if points.len() < 2 {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HairGuide \"{id}\" requires at least two HairPoints."),
        });
    }
    if points
        .windows(2)
        .any(|pair| pair[0].position == pair[1].position)
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("HairGuide \"{id}\" consecutive HairPoints must differ."),
        });
    }
    if let Some(n) = normal {
        let tangent = hair_unit(std::array::from_fn(|i| {
            points[1].position[i] - points[0].position[i]
        }));
        let cross = hair_cross(hair_unit(n), tangent);
        if n.iter().map(|x| x * x).sum::<f32>() < 1.0e-12
            || cross.iter().map(|x| x * x).sum::<f32>() < 1.0e-8
        {
            return Err(GraphParseError {
                line: start + 1,
                message: format!(
                    "HairGuide {id:?} normal must be nonzero and not parallel to its root tangent."
                ),
            });
        }
    }
    Ok((
        HairGuideNode {
            id,
            group: group.to_string(),
            role: role.to_string(),
            normal,
            points,
        },
        close_ix,
    ))
}

fn parse_hair_representations(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(HairCardParseSettings, usize), GraphParseError> {
    let (tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    validate_hair_attributes(&tag, &[], "HairRepresentations", start + 1)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "HairRepresentations")?;
    let mut cards = None;
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if !starts_open_tag(line, "HairCards") || cards.is_some() {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "HairAsset \"{asset_id}\" V1 supports exactly one HairCards representation."
                ),
            });
        }
        let (card_tag, end_ix) = collect_self_closing_block(lines, index)?;
        validate_hair_attributes(
            &card_tag,
            &[
                "id",
                "lengthSegments",
                "widthSegments",
                "thickness",
                "crossSection",
                "tipShape",
            ],
            "HairCards",
            index + 1,
        )?;
        let id = strip_wrappers(&required_attr_value(&card_tag, "id", index + 1)?).to_string();
        let length_segments =
            parse_optional_primitive_u32(&card_tag, "lengthSegments", asset_id, index + 1)?
                .unwrap_or(16);
        let width_segments =
            parse_optional_primitive_u32(&card_tag, "widthSegments", asset_id, index + 1)?
                .unwrap_or(4);
        if !(2..=128).contains(&length_segments) || !(2..=16).contains(&width_segments) {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "HairCards \"{id}\" requires lengthSegments 2..128 and widthSegments 2..16."
                ),
            });
        }
        let thickness =
            parse_optional_positive_primitive_number(&card_tag, "thickness", asset_id, index + 1)?
                .unwrap_or(0.01);
        let cross_section = primitive_string_attribute(&card_tag, "crossSection", "arched");
        if !matches!(cross_section.as_str(), "flat" | "arched" | "v_shape") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "HairCards \"{id}\" crossSection must be flat, arched, or v_shape."
                ),
            });
        }
        let tip_shape = primitive_string_attribute(&card_tag, "tipShape", "point");
        if !matches!(tip_shape.as_str(), "point" | "round" | "blunt") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("HairCards \"{id}\" tipShape must be point, round, or blunt."),
            });
        }
        cards = Some(HairCardParseSettings {
            id,
            length_segments,
            width_segments,
            thickness,
            cross_section,
            tip_shape,
        });
        index = end_ix + 1;
    }
    cards
        .map(|settings| (settings, close_ix))
        .ok_or_else(|| GraphParseError {
            line: start + 1,
            message: format!("HairAsset \"{asset_id}\" HairRepresentations requires HairCards."),
        })
}

fn validate_hair_attributes(
    tag: &str,
    allowed: &[&str],
    tag_name: &str,
    line: usize,
) -> Result<(), GraphParseError> {
    if let Some(attribute) = tag_attribute_names(tag)
        .into_iter()
        .find(|attribute| !allowed.contains(&attribute.as_str()))
    {
        return Err(GraphParseError {
            line,
            message: format!("<{tag_name}> does not support attribute \"{attribute}\"."),
        });
    }
    Ok(())
}

fn validate_primitive_build_budget(
    primitive: &PrimitiveAssetNode,
    line: usize,
) -> Result<(), GraphParseError> {
    let subdivision_levels = primitive
        .modifiers
        .iter()
        .filter_map(|modifier| match modifier {
            PrimitiveModifierNode::Subdivision { levels } => Some(*levels),
            _ => None,
        })
        .sum::<u32>();
    let multiplier = 4_usize.saturating_pow(subdivision_levels);
    let estimated = primitive
        .geometry
        .triangle_count()
        .saturating_mul(multiplier);
    if let Some(max_triangles) = primitive.mesh_build.max_triangles
        && estimated > max_triangles as usize
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{}\" is estimated to generate {estimated} triangles, exceeding MeshBuild maxTriangles={max_triangles}.",
                primitive.id
            ),
        });
    }
    Ok(())
}

fn parse_primitive_loft(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(PrimitiveGeometry, usize), GraphParseError> {
    let (tag, open_end) = collect_tag_block(lines, start, '>', false)?;
    validate_primitive_child_attributes(
        &tag,
        &["closed", "capStart", "capEnd", "segments"],
        asset_id,
        start + 1,
    )?;
    if is_self_closing_tag(&tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("PrimitiveAsset \"{asset_id}\" Loft must contain Section children."),
        });
    }
    let close = find_matching_close_tag(lines, open_end + 1, "Loft")?;
    let segments = parse_primitive_segments(&tag, "segments", 16, asset_id, start + 1)?;
    let closed =
        parse_optional_primitive_bool(&tag, "closed", asset_id, start + 1)?.unwrap_or(true);
    let cap_start =
        parse_optional_primitive_bool(&tag, "capStart", asset_id, start + 1)?.unwrap_or(true);
    let cap_end =
        parse_optional_primitive_bool(&tag, "capEnd", asset_id, start + 1)?.unwrap_or(true);
    let mut sections = Vec::new();
    let mut index = open_end + 1;
    while index < close {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if !starts_open_tag(line, "Section") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!("PrimitiveAsset \"{asset_id}\" Loft accepts Section children."),
            });
        }
        let (section_tag, end) = collect_self_closing_block(lines, index)?;
        validate_primitive_child_attributes(
            &section_tag,
            &["at", "width", "depth", "profile", "offset", "rotation"],
            asset_id,
            index + 1,
        )?;
        let profile = primitive_string_attribute(&section_tag, "profile", "ellipse");
        if !matches!(
            profile.as_str(),
            "ellipse" | "rounded_rect" | "diamond" | "capsule"
        ) {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "PrimitiveAsset \"{asset_id}\" Loft Section profile must be ellipse, rounded_rect, diamond, or capsule."
                ),
            });
        }
        sections.push(PrimitiveLoftSectionNode {
            at: parse_finite_primitive_number(&section_tag, "at", asset_id, index + 1)?,
            width: parse_positive_primitive_number(&section_tag, "width", asset_id, index + 1)?,
            depth: parse_positive_primitive_number(&section_tag, "depth", asset_id, index + 1)?,
            profile,
            offset: parse_optional_primitive_vec::<2>(
                &section_tag,
                "offset",
                asset_id,
                index + 1,
                false,
            )?
            .unwrap_or([0.0; 2]),
            rotation: attr_value(&section_tag, "rotation")
                .map(|_| {
                    parse_finite_primitive_number(&section_tag, "rotation", asset_id, index + 1)
                })
                .transpose()?
                .unwrap_or(0.0),
        });
        index = end + 1;
    }
    if sections.len() < 2 {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("PrimitiveAsset \"{asset_id}\" Loft requires at least two Sections."),
        });
    }
    if sections.windows(2).any(|pair| pair[0].at >= pair[1].at) {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" Loft Sections must use strictly increasing at values."
            ),
        });
    }
    Ok((
        PrimitiveGeometry::Loft {
            segments,
            closed,
            cap_start,
            cap_end,
            sections,
        },
        close,
    ))
}

fn parse_primitive_ribbon(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(PrimitiveGeometry, usize), GraphParseError> {
    let (tag, open_end) = collect_tag_block(lines, start, '>', false)?;
    validate_primitive_child_attributes(
        &tag,
        &["width", "thickness", "facing", "capStart", "capEnd"],
        asset_id,
        start + 1,
    )?;
    if is_self_closing_tag(&tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" Ribbon must contain PathPoint children."
            ),
        });
    }
    let facing = primitive_string_attribute(&tag, "facing", "stable");
    if !matches!(facing.as_str(), "stable" | "camera_safe") {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" Ribbon facing must be stable or camera_safe."
            ),
        });
    }
    let close = find_matching_close_tag(lines, open_end + 1, "Ribbon")?;
    let width = parse_positive_primitive_number(&tag, "width", asset_id, start + 1)?;
    let thickness =
        parse_optional_positive_primitive_number(&tag, "thickness", asset_id, start + 1)?
            .unwrap_or(0.02);
    let cap_start =
        parse_optional_primitive_bool(&tag, "capStart", asset_id, start + 1)?.unwrap_or(true);
    let cap_end =
        parse_optional_primitive_bool(&tag, "capEnd", asset_id, start + 1)?.unwrap_or(true);
    let mut points = Vec::new();
    let mut index = open_end + 1;
    while index < close {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if !starts_open_tag(line, "PathPoint") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "PrimitiveAsset \"{asset_id}\" Ribbon accepts PathPoint children."
                ),
            });
        }
        let (point_tag, end) = collect_self_closing_block(lines, index)?;
        validate_primitive_child_attributes(
            &point_tag,
            &["position", "width", "thickness", "roll"],
            asset_id,
            index + 1,
        )?;
        points.push(PrimitiveRibbonPointNode {
            position: parse_optional_primitive_vec::<3>(
                &point_tag,
                "position",
                asset_id,
                index + 1,
                false,
            )?
            .ok_or_else(|| GraphParseError {
                line: index + 1,
                message: format!("PrimitiveAsset \"{asset_id}\" PathPoint requires position."),
            })?,
            width: parse_optional_positive_primitive_number(
                &point_tag,
                "width",
                asset_id,
                index + 1,
            )?,
            thickness: parse_optional_positive_primitive_number(
                &point_tag,
                "thickness",
                asset_id,
                index + 1,
            )?,
            roll: attr_value(&point_tag, "roll")
                .map(|_| parse_finite_primitive_number(&point_tag, "roll", asset_id, index + 1))
                .transpose()?
                .unwrap_or(0.0),
        });
        index = end + 1;
    }
    if points.len() < 2 {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" Ribbon requires at least two PathPoints."
            ),
        });
    }
    if points
        .windows(2)
        .any(|pair| pair[0].position == pair[1].position)
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" Ribbon consecutive PathPoints must differ."
            ),
        });
    }
    Ok((
        PrimitiveGeometry::Ribbon {
            width,
            thickness,
            cap_start,
            cap_end,
            points,
        },
        close,
    ))
}

fn parse_compound_asset(
    lines: &[&str],
    start: usize,
) -> Result<(CompoundAssetNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: "CompoundAsset must contain at least one <Instance />.".to_string(),
        });
    }
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let rig = attr_value(&open_tag, "rig").map(|value| strip_wrappers(&value).to_string());
    let material_seed = parse_optional_primitive_u64(&open_tag, "materialSeed", &id, start + 1)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "CompoundAsset")?;
    let mut instances = Vec::new();
    let mut skin_binding = None;
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        if starts_open_tag(line, "SkinBinding") {
            if skin_binding.is_some() {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "CompoundAsset \"{id}\" must not contain more than one SkinBinding."
                    ),
                });
            }
            let (tag, open_end_ix) = collect_tag_block(lines, index, '>', false)?;
            validate_primitive_child_attributes(
                &tag,
                &["mode", "maxInfluences", "falloff", "normalize"],
                &id,
                index + 1,
            )?;
            let mode = attr_value(&tag, "mode")
                .map(|value| strip_wrappers(&value).to_ascii_lowercase())
                .unwrap_or_else(|| "automatic".to_string());
            if mode != "automatic" {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("CompoundAsset \"{id}\" SkinBinding mode must be automatic."),
                });
            }
            let max_influences =
                parse_optional_primitive_u64(&tag, "maxInfluences", &id, index + 1)?.unwrap_or(4);
            if !(1..=4).contains(&max_influences) {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "CompoundAsset \"{id}\" SkinBinding maxInfluences must be from 1 through 4."
                    ),
                });
            }
            let falloff = attr_value(&tag, "falloff")
                .map(|_| parse_finite_primitive_number(&tag, "falloff", &id, index + 1))
                .transpose()?
                .unwrap_or(2.5);
            if falloff <= 0.0 {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "CompoundAsset \"{id}\" SkinBinding falloff must be greater than zero."
                    ),
                });
            }
            let normalize =
                parse_optional_primitive_bool(&tag, "normalize", &id, index + 1)?.unwrap_or(true);
            let mut weight_regions = Vec::new();
            let end_ix = if is_self_closing_tag(&tag) {
                open_end_ix
            } else {
                let close = find_matching_close_tag(lines, open_end_ix + 1, "SkinBinding")?;
                let mut region_index = open_end_ix + 1;
                while region_index < close {
                    let line = lines[region_index].trim();
                    if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
                        region_index += 1;
                        continue;
                    }
                    if !starts_open_tag(line, "WeightRegion") {
                        return Err(GraphParseError {
                            line: region_index + 1,
                            message: format!(
                                "CompoundAsset \"{id}\" SkinBinding accepts WeightRegion children."
                            ),
                        });
                    }
                    let (region_tag, region_end) = collect_self_closing_block(lines, region_index)?;
                    validate_primitive_child_attributes(
                        &region_tag,
                        &[
                            "instance",
                            "bone",
                            "center",
                            "radius",
                            "strength",
                            "operation",
                        ],
                        &id,
                        region_index + 1,
                    )?;
                    let operation = primitive_string_attribute(&region_tag, "operation", "add");
                    if !matches!(operation.as_str(), "add" | "replace") {
                        return Err(GraphParseError {
                            line: region_index + 1,
                            message: format!(
                                "CompoundAsset \"{id}\" WeightRegion operation must be add or replace."
                            ),
                        });
                    }
                    let strength = attr_value(&region_tag, "strength")
                        .map(|_| {
                            parse_finite_primitive_number(
                                &region_tag,
                                "strength",
                                &id,
                                region_index + 1,
                            )
                        })
                        .transpose()?
                        .unwrap_or(1.0);
                    if !(0.0..=1.0).contains(&strength) {
                        return Err(GraphParseError {
                            line: region_index + 1,
                            message: format!(
                                "CompoundAsset \"{id}\" WeightRegion strength must be from zero through one."
                            ),
                        });
                    }
                    weight_regions.push(NativeWeightRegionNode {
                        instance: strip_wrappers(&required_attr_value(
                            &region_tag,
                            "instance",
                            region_index + 1,
                        )?)
                        .to_string(),
                        bone: strip_wrappers(&required_attr_value(
                            &region_tag,
                            "bone",
                            region_index + 1,
                        )?)
                        .to_string(),
                        center: parse_optional_primitive_vec::<3>(
                            &region_tag,
                            "center",
                            &id,
                            region_index + 1,
                            false,
                        )?
                        .ok_or_else(|| GraphParseError {
                            line: region_index + 1,
                            message: format!(
                                "CompoundAsset \"{id}\" WeightRegion requires center."
                            ),
                        })?,
                        radius: parse_positive_primitive_number(
                            &region_tag,
                            "radius",
                            &id,
                            region_index + 1,
                        )?,
                        strength,
                        operation,
                    });
                    region_index = region_end + 1;
                }
                close
            };
            skin_binding = Some(NativeSkinBindingNode {
                mode,
                max_influences: max_influences as u32,
                falloff,
                normalize,
                weight_regions,
            });
            index = end_ix + 1;
            continue;
        }
        if starts_open_tag(line, "Mirror") {
            let (mirror_tag, mirror_open_end) = collect_tag_block(lines, index, '>', false)?;
            validate_primitive_child_attributes(&mirror_tag, &["axis", "suffix"], &id, index + 1)?;
            let axis = primitive_string_attribute(&mirror_tag, "axis", "x");
            if axis != "x" {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "CompoundAsset \"{id}\" Mirror currently supports axis=\"x\"."
                    ),
                });
            }
            let suffix = attr_value(&mirror_tag, "suffix")
                .map(|value| parse_string_array(&value, index + 1, "Mirror suffix"))
                .transpose()?
                .unwrap_or_else(|| vec!["_l".to_string(), "_r".to_string()]);
            if suffix.len() != 2 || suffix[0] == suffix[1] {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!(
                        "CompoundAsset \"{id}\" Mirror suffix requires two distinct values."
                    ),
                });
            }
            let mirror_close = find_matching_close_tag(lines, mirror_open_end + 1, "Mirror")?;
            let mut child_index = mirror_open_end + 1;
            let mut child_count = 0;
            while child_index < mirror_close {
                let child_line = lines[child_index].trim();
                if child_line.is_empty()
                    || child_line.starts_with("//")
                    || child_line.starts_with("<!--")
                {
                    child_index += 1;
                    continue;
                }
                if !starts_open_tag(child_line, "Instance") {
                    return Err(GraphParseError {
                        line: child_index + 1,
                        message: format!(
                            "CompoundAsset \"{id}\" Mirror accepts Instance children."
                        ),
                    });
                }
                let (child_tag, child_end) = collect_self_closing_block(lines, child_index)?;
                let original = parse_compound_instance_tag(
                    &child_tag,
                    &id,
                    child_index + 1,
                    instances.len() + 1,
                )?;
                let mirrored = mirror_compound_instance(&original, &suffix[0], &suffix[1]);
                instances.push(original);
                instances.push(mirrored);
                child_count += 1;
                child_index = child_end + 1;
            }
            if child_count == 0 {
                return Err(GraphParseError {
                    line: index + 1,
                    message: format!("CompoundAsset \"{id}\" Mirror requires an Instance."),
                });
            }
            index = mirror_close + 1;
            continue;
        }
        if !starts_open_tag(line, "Instance") {
            return Err(GraphParseError {
                line: index + 1,
                message: format!(
                    "CompoundAsset \"{id}\" accepts SkinBinding, Mirror, and self-closing Instance children."
                ),
            });
        }
        let (tag, end_ix) = collect_self_closing_block(lines, index)?;
        instances.push(parse_compound_instance_tag(
            &tag,
            &id,
            index + 1,
            instances.len() + 1,
        )?);
        index = end_ix + 1;
    }
    if instances.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: format!("CompoundAsset \"{id}\" must contain at least one Instance."),
        });
    }
    let mut ids = HashSet::new();
    if let Some(duplicate) = instances
        .iter()
        .find(|instance| !ids.insert(instance.id.clone()))
    {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "CompoundAsset \"{id}\" has duplicate Instance id \"{}\".",
                duplicate.id
            ),
        });
    }
    Ok((
        CompoundAssetNode {
            id,
            rig,
            material_seed,
            skin_binding,
            instances,
        },
        close_ix,
    ))
}

fn parse_compound_instance_tag(
    tag: &str,
    compound_id: &str,
    line: usize,
    fallback_index: usize,
) -> Result<CompoundAssetInstanceNode, GraphParseError> {
    validate_primitive_child_attributes(
        tag,
        &[
            "id",
            "asset",
            "bone",
            "position",
            "rotation",
            "scale",
            "materialSeed",
            "skin",
            "influences",
        ],
        compound_id,
        line,
    )?;
    let id = attr_value(tag, "id")
        .map(|value| strip_wrappers(&value).to_string())
        .unwrap_or_else(|| format!("instance_{fallback_index}"));
    let asset = strip_wrappers(&required_attr_value(tag, "asset", line)?).to_string();
    let bone = attr_value(tag, "bone").map(|value| strip_wrappers(&value).to_string());
    let position =
        parse_optional_primitive_vec::<3>(tag, "position", &id, line, false)?.unwrap_or([0.0; 3]);
    let rotation =
        parse_optional_primitive_vec::<3>(tag, "rotation", &id, line, false)?.unwrap_or([0.0; 3]);
    let scale = parse_optional_positive_primitive_number(tag, "scale", &id, line)?.unwrap_or(1.0);
    let material_seed = parse_optional_primitive_u64(tag, "materialSeed", &id, line)?;
    let skin = match attr_value(tag, "skin")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .as_deref()
    {
        None | Some("rigid") => NativeSkinMode::Rigid,
        Some("smooth") => NativeSkinMode::Smooth,
        Some(other) => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "CompoundAsset \"{compound_id}\" Instance \"{id}\" skin=\"{other}\" is invalid. Use rigid or smooth."
                ),
            });
        }
    };
    let influences = attr_value(tag, "influences")
        .map(|value| {
            parse_string_array(
                &value,
                line,
                &format!("CompoundAsset {compound_id} Instance {id} influences"),
            )
        })
        .transpose()?
        .unwrap_or_default();
    Ok(CompoundAssetInstanceNode {
        id,
        asset,
        bone,
        position,
        rotation,
        scale,
        material_seed,
        skin,
        influences,
    })
}

fn mirror_compound_instance(
    original: &CompoundAssetInstanceNode,
    source_suffix: &str,
    target_suffix: &str,
) -> CompoundAssetInstanceNode {
    let swap = |value: &str| {
        value.strip_suffix(source_suffix).map_or_else(
            || format!("{value}{target_suffix}"),
            |stem| format!("{stem}{target_suffix}"),
        )
    };
    let swap_if_suffixed = |value: &str| {
        value.strip_suffix(source_suffix).map_or_else(
            || value.to_string(),
            |stem| format!("{stem}{target_suffix}"),
        )
    };
    let mut mirrored = original.clone();
    mirrored.id = swap(&original.id);
    mirrored.bone = original.bone.as_deref().map(&swap_if_suffixed);
    mirrored.position[0] = -mirrored.position[0];
    mirrored.rotation[1] = -mirrored.rotation[1];
    mirrored.rotation[2] = -mirrored.rotation[2];
    mirrored.influences = original
        .influences
        .iter()
        .map(|value| swap_if_suffixed(value))
        .collect();
    mirrored
}

fn parse_material_asset(tag: &str, line: usize) -> Result<MaterialAssetNode, GraphParseError> {
    const ALLOWED: &[&str] = &[
        "id",
        "shading",
        "baseColor",
        "baseColorTexture",
        "metallic",
        "roughness",
        "metallicRoughnessTexture",
        "normalTexture",
        "normalScale",
        "occlusionTexture",
        "occlusionStrength",
        "emissive",
        "emissiveTexture",
        "emissiveStrength",
        "specular",
        "doubleSided",
        "alphaMode",
        "alphaCutoff",
        "transmission",
        "ior",
        "thickness",
        "attenuationColor",
        "attenuationDistance",
        "depthWrite",
        "sortPriority",
        "mapping",
        "textureScale",
        "textureOffset",
        "textureRotation",
        "variationAmount",
    ];
    let id = strip_wrappers(&required_attr_value(tag, "id", line)?).to_string();
    for attribute in tag_attribute_names(tag) {
        if !ALLOWED.contains(&attribute.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("MaterialAsset \"{id}\" does not support \"{attribute}\"."),
            });
        }
    }
    let shading = attr_value(tag, "shading")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "pbr".to_string());
    if shading != "pbr" {
        return Err(GraphParseError {
            line,
            message: format!(
                "MaterialAsset \"{id}\" shading=\"{shading}\" is invalid. V1 supports pbr."
            ),
        });
    }
    let scalar = |attribute: &str, default: f32, min: f32, max: f32| {
        let Some(raw) = attr_value(tag, attribute) else {
            return Ok(default);
        };
        let value = strip_wrappers(&raw)
            .parse::<f32>()
            .map_err(|_| GraphParseError {
                line,
                message: format!("MaterialAsset \"{id}\" {attribute} must be a finite number."),
            })?;
        if !value.is_finite() || !(min..=max).contains(&value) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "MaterialAsset \"{id}\" {attribute} must be from {min} through {max}."
                ),
            });
        }
        Ok(value)
    };
    let texture_ref = |attribute: &str| {
        attr_value(tag, attribute).map(|value| strip_wrappers(&value).to_string())
    };
    let base_color = attr_value(tag, "baseColor")
        .map(|value| parse_primitive_color(&value, &id, line))
        .transpose()?
        .unwrap_or([1.0; 4]);
    let emissive_rgba = attr_value(tag, "emissive")
        .map(|value| parse_primitive_color(&value, &id, line))
        .transpose()?
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let attenuation_rgba = attr_value(tag, "attenuationColor")
        .map(|value| parse_primitive_color(&value, &id, line))
        .transpose()?
        .unwrap_or([1.0; 4]);
    let mapping = attr_value(tag, "mapping")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "uv".to_string());
    if !matches!(mapping.as_str(), "uv" | "box" | "triplanar") {
        return Err(GraphParseError {
            line,
            message: format!(
                "MaterialAsset \"{id}\" mapping=\"{mapping}\" is invalid. Use uv, box, or triplanar."
            ),
        });
    }
    let alpha_mode = attr_value(tag, "alphaMode")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "opaque".to_string());
    if !matches!(alpha_mode.as_str(), "opaque" | "mask" | "blend") {
        return Err(GraphParseError {
            line,
            message: format!(
                "MaterialAsset \"{id}\" alphaMode=\"{alpha_mode}\" is invalid. Use opaque, mask, or blend."
            ),
        });
    }
    let depth_write = attr_value(tag, "depthWrite")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(default_material_depth_write);
    if !matches!(depth_write.as_str(), "auto" | "true" | "false") {
        return Err(GraphParseError {
            line,
            message: format!(
                "MaterialAsset \"{id}\" depthWrite=\"{depth_write}\" is invalid. Use auto, true, or false."
            ),
        });
    }
    let sort_priority = attr_value(tag, "sortPriority")
        .map(|value| {
            strip_wrappers(&value)
                .parse::<i32>()
                .map_err(|_| GraphParseError {
                    line,
                    message: format!(
                        "MaterialAsset \"{id}\" sortPriority must be an integer from -32768 through 32767."
                    ),
                })
        })
        .transpose()?
        .unwrap_or(0);
    if !(-32768..=32767).contains(&sort_priority) {
        return Err(GraphParseError {
            line,
            message: format!(
                "MaterialAsset \"{id}\" sortPriority must be an integer from -32768 through 32767."
            ),
        });
    }
    let texture_scale = parse_optional_primitive_vec::<2>(tag, "textureScale", &id, line, true)?
        .unwrap_or([1.0; 2]);
    let texture_offset = parse_optional_primitive_vec::<2>(tag, "textureOffset", &id, line, false)?
        .unwrap_or([0.0; 2]);
    let variation_amount =
        parse_optional_primitive_vec::<2>(tag, "variationAmount", &id, line, false)?
            .unwrap_or([0.0; 2]);
    if variation_amount.iter().any(|value| *value < 0.0) {
        return Err(GraphParseError {
            line,
            message: format!(
                "MaterialAsset \"{id}\" variationAmount values must be equal to or greater than zero."
            ),
        });
    }
    let double_sided = attr_value(tag, "doubleSided")
        .map(|value| parse_bool(&value, line, "MaterialAsset.doubleSided"))
        .transpose()?
        .unwrap_or(false);
    Ok(MaterialAssetNode {
        id: id.clone(),
        shading,
        base_color,
        base_color_texture: texture_ref("baseColorTexture"),
        metallic_roughness_texture: texture_ref("metallicRoughnessTexture"),
        normal_texture: texture_ref("normalTexture"),
        occlusion_texture: texture_ref("occlusionTexture"),
        emissive_texture: texture_ref("emissiveTexture"),
        base_color_texture_src: None,
        metallic_roughness_texture_src: None,
        normal_texture_src: None,
        occlusion_texture_src: None,
        emissive_texture_src: None,
        metallic: scalar("metallic", 0.0, 0.0, 1.0)?,
        roughness: scalar("roughness", 0.82, 0.04, 1.0)?,
        normal_scale: scalar("normalScale", 1.0, 0.0, 4.0)?,
        occlusion_strength: scalar("occlusionStrength", 1.0, 0.0, 1.0)?,
        emissive: [emissive_rgba[0], emissive_rgba[1], emissive_rgba[2]],
        emissive_strength: scalar("emissiveStrength", 1.0, 0.0, 64.0)?,
        specular: scalar("specular", 1.0, 0.0, 2.0)?,
        double_sided,
        alpha_mode,
        alpha_cutoff: scalar("alphaCutoff", 0.5, 0.0, 1.0)?,
        transmission: scalar("transmission", 0.0, 0.0, 1.0)?,
        ior: scalar("ior", 1.5, 1.0, 3.0)?,
        thickness: scalar("thickness", 0.0, 0.0, 1000.0)?,
        attenuation_color: [
            attenuation_rgba[0],
            attenuation_rgba[1],
            attenuation_rgba[2],
        ],
        attenuation_distance: scalar(
            "attenuationDistance",
            default_material_attenuation_distance(),
            0.0001,
            1_000_000.0,
        )?,
        depth_write,
        sort_priority,
        mapping,
        texture_scale,
        texture_offset,
        texture_rotation: scalar("textureRotation", 0.0, -3600.0, 3600.0)?,
        variation_amount,
    })
}

fn resolve_primitive_material_assets(
    assets: &mut [GraphAssetNode],
    materials: &[MaterialAssetNode],
    line: usize,
) -> Result<(), GraphParseError> {
    let mut material_ids = HashSet::new();
    for material in materials {
        if !material_ids.insert(material.id.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("Duplicate MaterialAsset id: {}", material.id),
            });
        }
    }
    let image_sources = assets
        .iter()
        .filter(|asset| asset.kind == GraphAssetKind::Image)
        .filter_map(|asset| {
            asset
                .external_src()
                .map(|src| (asset.id.clone(), src.to_string()))
        })
        .collect::<HashMap<_, _>>();
    let image_color_spaces = assets
        .iter()
        .filter(|asset| asset.kind == GraphAssetKind::Image)
        .map(|asset| (asset.id.clone(), asset.color_space.clone()))
        .collect::<HashMap<_, _>>();
    for asset in assets.iter_mut() {
        if let GraphAssetSource::Primitive(PrimitiveAssetNode {
            geometry:
                PrimitiveGeometry::HeadSurface {
                    face_layout: Some(layout),
                    ..
                },
            ..
        }) = &mut asset.source
        {
            let resolve_face_texture =
                |texture: &mut FaceTextureNode| -> Result<(), GraphParseError> {
                    texture.source =
                        Some(image_sources.get(&texture.asset).cloned().ok_or_else(|| {
                            GraphParseError {
                                line,
                                message: format!(
                                    "Texture references unknown ImageAsset \"{}\".",
                                    texture.asset
                                ),
                            }
                        })?);
                    Ok(())
                };
            for eye in &mut layout.eyes {
                if let Some(texture) = &mut eye.texture {
                    resolve_face_texture(texture)?;
                }
                if let Some(iris) = &mut eye.iris {
                    if let Some(texture) = &mut iris.texture {
                        resolve_face_texture(texture)?;
                    }
                }
                for liner in &mut eye.eyeliners {
                    if let Some(texture) = &mut liner.texture {
                        resolve_face_texture(texture)?;
                    }
                }
            }
            for texture in layout
                .eyebrows
                .iter_mut()
                .filter_map(|e| e.texture.as_mut())
                .chain(layout.noses.iter_mut().filter_map(|e| e.texture.as_mut()))
                .chain(layout.mouths.iter_mut().filter_map(|e| e.texture.as_mut()))
                .chain(layout.ears.iter_mut().filter_map(|e| e.texture.as_mut()))
            {
                resolve_face_texture(texture)?;
            }
        }
    }
    let resolve_texture = |material: &MaterialAssetNode,
                           slot: &str,
                           reference: &Option<String>|
     -> Result<Option<String>, GraphParseError> {
        let Some(reference) = reference.as_deref() else {
            return Ok(None);
        };
        image_sources
            .get(reference)
            .map(|src| Some(src.clone()))
            .ok_or_else(|| GraphParseError {
                line,
                message: format!(
                    "MaterialAsset \"{}\" {slot} references unknown ImageAsset \"{reference}\".",
                    material.id
                ),
            })
    };
    let mut resolved_materials = HashMap::new();
    for material in materials {
        let mut resolved = material.clone();
        resolved.base_color_texture_src =
            resolve_texture(material, "baseColorTexture", &material.base_color_texture)?;
        resolved.metallic_roughness_texture_src = resolve_texture(
            material,
            "metallicRoughnessTexture",
            &material.metallic_roughness_texture,
        )?;
        resolved.normal_texture_src =
            resolve_texture(material, "normalTexture", &material.normal_texture)?;
        resolved.occlusion_texture_src =
            resolve_texture(material, "occlusionTexture", &material.occlusion_texture)?;
        resolved.emissive_texture_src =
            resolve_texture(material, "emissiveTexture", &material.emissive_texture)?;
        resolved_materials.insert(material.id.as_str(), resolved);
    }
    for asset in assets {
        match &mut asset.source {
            GraphAssetSource::Primitive(primitive) => {
                let Some(material_id) = primitive.material.as_deref() else {
                    continue;
                };
                primitive.material_definition = resolved_materials.get(material_id).cloned();
                if primitive.material_definition.is_none() {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "PrimitiveAsset \"{}\" references unknown MaterialAsset \"{material_id}\".",
                            primitive.id
                        ),
                    });
                }
            }
            GraphAssetSource::Terrain(terrain) => {
                terrain.height_map_src = image_sources.get(terrain.height_map.as_str()).cloned();
                if terrain.height_map_src.is_none() {
                    return Err(GraphParseError {
                        line,
                        message: format!(
                            "TerrainAsset \"{}\" heightMap references unknown ImageAsset \"{}\".",
                            terrain.id, terrain.height_map
                        ),
                    });
                }
                require_linear_terrain_image(
                    &terrain.id,
                    "heightMap",
                    &terrain.height_map,
                    &image_color_spaces,
                    line,
                )?;
                if let Some(material_id) = terrain.material.as_deref() {
                    terrain.material_definition = resolved_materials.get(material_id).cloned();
                    if terrain.material_definition.is_none() {
                        return Err(GraphParseError {
                            line,
                            message: format!(
                                "TerrainAsset \"{}\" references unknown MaterialAsset \"{material_id}\".",
                                terrain.id
                            ),
                        });
                    }
                }
                terrain.layer_definitions = terrain
                    .layers
                    .iter()
                    .map(|material_id| {
                        resolved_materials.get(material_id.as_str()).cloned().ok_or_else(|| {
                            GraphParseError {
                                line,
                                message: format!(
                                    "TerrainAsset \"{}\" layers references unknown MaterialAsset \"{material_id}\".",
                                    terrain.id
                                ),
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(blend_map) = terrain.blend_map.as_deref() {
                    terrain.blend_map_src = image_sources.get(blend_map).cloned();
                    if terrain.blend_map_src.is_none() {
                        return Err(GraphParseError {
                            line,
                            message: format!(
                                "TerrainAsset \"{}\" blendMap references unknown ImageAsset \"{blend_map}\".",
                                terrain.id
                            ),
                        });
                    }
                    require_linear_terrain_image(
                        &terrain.id,
                        "blendMap",
                        blend_map,
                        &image_color_spaces,
                        line,
                    )?;
                }
            }
            GraphAssetSource::Vegetation(vegetation) => {
                vegetation.material_definition = resolve_vegetation_material(
                    vegetation,
                    "material",
                    vegetation.material.as_deref(),
                    &resolved_materials,
                    line,
                )?;
                vegetation.stem_material_definition = resolve_vegetation_material(
                    vegetation,
                    "stemMaterial",
                    vegetation.stem_material.as_deref(),
                    &resolved_materials,
                    line,
                )?;
                vegetation.trunk_material_definition = resolve_vegetation_material(
                    vegetation,
                    "trunkMaterial",
                    vegetation.trunk_material.as_deref(),
                    &resolved_materials,
                    line,
                )?;
                vegetation.foliage_material_definition = resolve_vegetation_material(
                    vegetation,
                    "foliageMaterial",
                    vegetation.foliage_material.as_deref(),
                    &resolved_materials,
                    line,
                )?;
            }
            GraphAssetSource::External { .. } | GraphAssetSource::Compound(_) => {}
        }
    }
    Ok(())
}

fn resolve_vegetation_material(
    vegetation: &VegetationAssetNode,
    slot: &str,
    material_id: Option<&str>,
    materials: &HashMap<&str, MaterialAssetNode>,
    line: usize,
) -> Result<Option<MaterialAssetNode>, GraphParseError> {
    let Some(material_id) = material_id else {
        return Ok(None);
    };
    materials
        .get(material_id)
        .cloned()
        .map(Some)
        .ok_or_else(|| GraphParseError {
            line,
            message: format!(
                "VegetationAsset \"{}\" {slot} references unknown MaterialAsset \"{material_id}\".",
                vegetation.id
            ),
        })
}

fn require_linear_terrain_image(
    terrain_id: &str,
    slot: &str,
    image_id: &str,
    color_spaces: &HashMap<String, Option<String>>,
    line: usize,
) -> Result<(), GraphParseError> {
    if color_spaces
        .get(image_id)
        .and_then(|value| value.as_deref())
        .is_some_and(|value| value.eq_ignore_ascii_case("linear-srgb"))
    {
        return Ok(());
    }
    Err(GraphParseError {
        line,
        message: format!(
            "TerrainAsset \"{terrain_id}\" {slot} ImageAsset \"{image_id}\" must declare colorSpace=\"linear-srgb\"."
        ),
    })
}

fn parse_terrain_asset(
    tag: &str,
    id: &str,
    line: usize,
) -> Result<TerrainAssetNode, GraphParseError> {
    const ALLOWED: &[&str] = &[
        "id",
        "heightMap",
        "size",
        "heightScale",
        "heightOffset",
        "material",
        "layers",
        "blendMap",
        "chunks",
        "lod",
        "collision",
    ];
    for attribute in tag_attribute_names(tag) {
        if !ALLOWED.contains(&attribute.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("TerrainAsset \"{id}\" does not support \"{attribute}\"."),
            });
        }
    }
    let height_map = strip_wrappers(&required_attr_value(tag, "heightMap", line)?)
        .trim()
        .to_string();
    let size = parse_primitive_vec::<2>(tag, "size", id, line)?;
    let height_scale = attr_value(tag, "heightScale")
        .map(|value| strip_wrappers(&value).parse::<f32>())
        .transpose()
        .map_err(|_| GraphParseError {
            line,
            message: format!("TerrainAsset \"{id}\" heightScale must be a finite number."),
        })?
        .unwrap_or(1.0);
    let height_offset = attr_value(tag, "heightOffset")
        .map(|value| strip_wrappers(&value).parse::<f32>())
        .transpose()
        .map_err(|_| GraphParseError {
            line,
            message: format!("TerrainAsset \"{id}\" heightOffset must be a finite number."),
        })?
        .unwrap_or(0.0);
    if !height_scale.is_finite() || !height_offset.is_finite() {
        return Err(GraphParseError {
            line,
            message: format!("TerrainAsset \"{id}\" height values must be finite."),
        });
    }
    let material = attr_value(tag, "material").map(|value| strip_wrappers(&value).to_string());
    let layers = attr_value(tag, "layers")
        .map(|value| parse_string_array(&value, line, "TerrainAsset.layers"))
        .transpose()?
        .unwrap_or_default();
    if layers.len() > 4 {
        return Err(GraphParseError {
            line,
            message: format!("TerrainAsset \"{id}\" supports at most four material layers."),
        });
    }
    let blend_map = attr_value(tag, "blendMap").map(|value| strip_wrappers(&value).to_string());
    if blend_map.is_some() == layers.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("TerrainAsset \"{id}\" must declare layers and blendMap together."),
        });
    }
    if material.is_none() && layers.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("TerrainAsset \"{id}\" requires material or layered materials."),
        });
    }
    let chunk_values =
        parse_optional_primitive_vec::<2>(tag, "chunks", id, line, true)?.unwrap_or([1.0, 1.0]);
    let chunks = chunk_values.map(|value| value as u32);
    if chunk_values
        .iter()
        .any(|value| value.fract() != 0.0 || !(1.0..=32.0).contains(value))
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "TerrainAsset \"{id}\" chunks must contain integers from 1 through 32."
            ),
        });
    }
    let lod = attr_value(tag, "lod")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "auto".to_string());
    if !matches!(lod.as_str(), "auto" | "full" | "half" | "quarter") {
        return Err(GraphParseError {
            line,
            message: format!(
                "TerrainAsset \"{id}\" lod=\"{lod}\" is invalid. Use auto, full, half, or quarter."
            ),
        });
    }
    let collision = match attr_value(tag, "collision")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .as_deref()
        .unwrap_or("solid")
    {
        "none" => PrimitiveCollisionMode::None,
        "solid" => PrimitiveCollisionMode::Solid,
        other => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "TerrainAsset \"{id}\" collision=\"{other}\" is invalid. Use none or solid."
                ),
            });
        }
    };
    Ok(TerrainAssetNode {
        id: id.to_string(),
        height_map,
        height_map_src: None,
        size,
        height_scale,
        height_offset,
        material,
        material_definition: None,
        layers,
        layer_definitions: Vec::new(),
        blend_map,
        blend_map_src: None,
        chunks,
        lod,
        collision,
    })
}

fn parse_vegetation_asset(
    tag: &str,
    id: &str,
    line: usize,
) -> Result<VegetationAssetNode, GraphParseError> {
    const ALLOWED: &[&str] = &[
        "id",
        "kind",
        "height",
        "material",
        "stemMaterial",
        "trunkMaterial",
        "foliageMaterial",
        "density",
        "branchLevels",
        "seed",
        "lod",
        "wind",
        "collision",
    ];
    for attribute in tag_attribute_names(tag) {
        if !ALLOWED.contains(&attribute.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("VegetationAsset \"{id}\" does not support \"{attribute}\"."),
            });
        }
    }
    let kind_name = strip_wrappers(&required_attr_value(tag, "kind", line)?).to_ascii_lowercase();
    let kind = match kind_name.as_str() {
        "tree" => VegetationKind::Tree,
        "shrub" => VegetationKind::Shrub,
        "grass" => VegetationKind::Grass,
        "flower" => VegetationKind::Flower,
        "fern" => VegetationKind::Fern,
        "deadwood" => VegetationKind::Deadwood,
        _ => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "VegetationAsset \"{id}\" kind=\"{kind_name}\" is invalid. Use tree, shrub, grass, flower, fern, or deadwood."
                ),
            });
        }
    };
    let height = strip_wrappers(&required_attr_value(tag, "height", line)?)
        .parse::<f32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!(
                "VegetationAsset \"{id}\" height must be a finite number greater than zero."
            ),
        })?;
    if !height.is_finite() || height <= 0.0 {
        return Err(GraphParseError {
            line,
            message: format!(
                "VegetationAsset \"{id}\" height must be a finite number greater than zero."
            ),
        });
    }
    let material = vegetation_material_attr(tag, "material");
    let stem_material = vegetation_material_attr(tag, "stemMaterial");
    let trunk_material = vegetation_material_attr(tag, "trunkMaterial");
    let foliage_material = vegetation_material_attr(tag, "foliageMaterial");
    validate_vegetation_material_slots(
        id,
        kind,
        material.as_deref(),
        stem_material.as_deref(),
        trunk_material.as_deref(),
        foliage_material.as_deref(),
        line,
    )?;
    let density = parse_vegetation_u32(tag, "density", id, line)?.unwrap_or(match kind {
        VegetationKind::Tree => 28,
        VegetationKind::Shrub => 20,
        VegetationKind::Grass => 18,
        VegetationKind::Flower => 10,
        VegetationKind::Fern => 12,
        VegetationKind::Deadwood => 0,
    });
    if density > 256 || (kind != VegetationKind::Deadwood && density == 0) {
        return Err(GraphParseError {
            line,
            message: format!(
                "VegetationAsset \"{id}\" density must be from 1 through 256; deadwood omits density."
            ),
        });
    }
    if kind == VegetationKind::Deadwood && attr_value(tag, "density").is_some() {
        return Err(GraphParseError {
            line,
            message: format!("VegetationAsset \"{id}\" deadwood does not support density."),
        });
    }
    let branch_levels =
        parse_vegetation_u32(tag, "branchLevels", id, line)?.unwrap_or(match kind {
            VegetationKind::Tree => 3,
            VegetationKind::Shrub => 2,
            VegetationKind::Deadwood => 1,
            VegetationKind::Grass | VegetationKind::Flower | VegetationKind::Fern => 0,
        });
    if branch_levels > 5 {
        return Err(GraphParseError {
            line,
            message: format!("VegetationAsset \"{id}\" branchLevels must be from 0 through 5."),
        });
    }
    if matches!(
        kind,
        VegetationKind::Grass | VegetationKind::Flower | VegetationKind::Fern
    ) && attr_value(tag, "branchLevels").is_some()
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "VegetationAsset \"{id}\" kind=\"{kind_name}\" does not support branchLevels."
            ),
        });
    }
    let seed = attr_value(tag, "seed")
        .map(|value| strip_wrappers(&value).parse::<u64>())
        .transpose()
        .map_err(|_| GraphParseError {
            line,
            message: format!("VegetationAsset \"{id}\" seed must be an unsigned integer."),
        })?
        .unwrap_or(0);
    let lod_name = attr_value(tag, "lod")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "auto".to_string());
    let lod = match lod_name.as_str() {
        "auto" => VegetationLod::Auto,
        "full" => VegetationLod::Full,
        "half" => VegetationLod::Half,
        "quarter" => VegetationLod::Quarter,
        _ => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "VegetationAsset \"{id}\" lod=\"{lod_name}\" is invalid. Use auto, full, half, or quarter."
                ),
            });
        }
    };
    let wind = attr_value(tag, "wind")
        .map(|value| parse_bool(&value, line, "VegetationAsset.wind"))
        .transpose()?
        .unwrap_or(false);
    let collision_name = attr_value(tag, "collision")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .unwrap_or_else(|| "none".to_string());
    let collision = match collision_name.as_str() {
        "none" => PrimitiveCollisionMode::None,
        "solid" if matches!(kind, VegetationKind::Tree | VegetationKind::Deadwood) => {
            PrimitiveCollisionMode::Solid
        }
        "solid" => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "VegetationAsset \"{id}\" kind=\"{kind_name}\" does not support solid collision in V1."
                ),
            });
        }
        _ => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "VegetationAsset \"{id}\" collision=\"{collision_name}\" is invalid. Use none or solid."
                ),
            });
        }
    };
    Ok(VegetationAssetNode {
        id: id.to_string(),
        kind,
        height,
        material,
        material_definition: None,
        stem_material,
        stem_material_definition: None,
        trunk_material,
        trunk_material_definition: None,
        foliage_material,
        foliage_material_definition: None,
        density,
        branch_levels,
        seed,
        lod,
        wind,
        collision,
    })
}

fn vegetation_material_attr(tag: &str, attribute: &str) -> Option<String> {
    attr_value(tag, attribute).map(|value| strip_wrappers(&value).to_string())
}

#[allow(clippy::too_many_arguments)]
fn validate_vegetation_material_slots(
    id: &str,
    kind: VegetationKind,
    material: Option<&str>,
    stem_material: Option<&str>,
    trunk_material: Option<&str>,
    foliage_material: Option<&str>,
    line: usize,
) -> Result<(), GraphParseError> {
    let invalid = |message: String| Err(GraphParseError { line, message });
    match kind {
        VegetationKind::Tree | VegetationKind::Shrub => {
            if trunk_material.is_none() || foliage_material.is_none() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" tree/shrub requires trunkMaterial and foliageMaterial."
                ));
            }
            if material.is_some() || stem_material.is_some() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" tree/shrub uses trunkMaterial and foliageMaterial, not material or stemMaterial."
                ));
            }
        }
        VegetationKind::Grass | VegetationKind::Fern => {
            if material.is_none() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" grass/fern requires material."
                ));
            }
            if stem_material.is_some() || trunk_material.is_some() || foliage_material.is_some() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" grass/fern only supports material."
                ));
            }
        }
        VegetationKind::Flower => {
            if material.is_none() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" flower requires material."
                ));
            }
            if trunk_material.is_some() || foliage_material.is_some() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" flower supports material and optional stemMaterial."
                ));
            }
        }
        VegetationKind::Deadwood => {
            if trunk_material.is_none() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" deadwood requires trunkMaterial."
                ));
            }
            if material.is_some() || stem_material.is_some() || foliage_material.is_some() {
                return invalid(format!(
                    "VegetationAsset \"{id}\" deadwood only supports trunkMaterial."
                ));
            }
        }
    }
    Ok(())
}

fn parse_vegetation_u32(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<Option<u32>, GraphParseError> {
    attr_value(tag, attribute)
        .map(|value| {
            strip_wrappers(&value)
                .parse::<u32>()
                .map_err(|_| GraphParseError {
                    line,
                    message: format!(
                        "VegetationAsset \"{id}\" {attribute} must be an unsigned integer."
                    ),
                })
        })
        .transpose()
}

fn parse_primitive_modifiers(
    lines: &[&str],
    start: usize,
    asset_id: &str,
) -> Result<(Vec<PrimitiveModifierNode>, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    if is_self_closing_tag(&open_tag) {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" <Modifiers> must contain at least one modifier."
            ),
        });
    }
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Modifiers")?;
    let mut modifiers = Vec::new();
    let mut index = open_end_ix + 1;
    while index < close_ix {
        let line = lines[index].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with("<!--") {
            index += 1;
            continue;
        }
        let (tag, end_ix) = collect_self_closing_block(lines, index)?;
        modifiers.push(parse_primitive_modifier(&tag, asset_id, index + 1)?);
        index = end_ix + 1;
    }
    if modifiers.is_empty() {
        return Err(GraphParseError {
            line: start + 1,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" <Modifiers> must contain at least one modifier."
            ),
        });
    }
    Ok((modifiers, close_ix))
}

fn parse_primitive_modifier(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<PrimitiveModifierNode, GraphParseError> {
    let name = [
        "MeshTransform",
        "Taper",
        "Bend",
        "Twist",
        "Subdivision",
        "Smooth",
        "WeightedNormals",
    ]
    .into_iter()
    .find(|name| starts_open_tag(tag.trim(), name))
    .ok_or_else(|| GraphParseError {
        line,
        message: format!("PrimitiveAsset \"{asset_id}\" has an invalid modifier tag."),
    })?;
    let modifier = match name {
        "MeshTransform" => {
            validate_primitive_child_attributes(
                tag,
                &["translate", "rotate", "scale"],
                asset_id,
                line,
            )?;
            PrimitiveModifierNode::Transform {
                translate: parse_optional_primitive_vec::<3>(
                    tag,
                    "translate",
                    asset_id,
                    line,
                    false,
                )?
                .unwrap_or([0.0; 3]),
                rotate: parse_optional_primitive_vec::<3>(tag, "rotate", asset_id, line, false)?
                    .unwrap_or([0.0; 3]),
                scale: parse_optional_primitive_vec::<3>(tag, "scale", asset_id, line, true)?
                    .unwrap_or([1.0; 3]),
            }
        }
        "Taper" => {
            validate_primitive_child_attributes(tag, &["axis", "start", "end"], asset_id, line)?;
            PrimitiveModifierNode::Taper {
                axis: parse_primitive_axis(tag, asset_id, line)?,
                start: parse_positive_primitive_number(tag, "start", asset_id, line)?,
                end: parse_positive_primitive_number(tag, "end", asset_id, line)?,
            }
        }
        "Bend" => {
            validate_primitive_child_attributes(tag, &["axis", "angle", "pivot"], asset_id, line)?;
            PrimitiveModifierNode::Bend {
                axis: parse_primitive_axis(tag, asset_id, line)?,
                angle: parse_finite_primitive_number(tag, "angle", asset_id, line)?,
                pivot: parse_optional_primitive_vec::<3>(tag, "pivot", asset_id, line, false)?
                    .unwrap_or([0.0; 3]),
            }
        }
        "Twist" => {
            validate_primitive_child_attributes(tag, &["axis", "angle"], asset_id, line)?;
            PrimitiveModifierNode::Twist {
                axis: parse_primitive_axis(tag, asset_id, line)?,
                angle: parse_finite_primitive_number(tag, "angle", asset_id, line)?,
            }
        }
        "Subdivision" => {
            validate_primitive_child_attributes(tag, &["levels"], asset_id, line)?;
            let levels = parse_optional_primitive_u32(tag, "levels", asset_id, line)?.unwrap_or(1);
            if !(1..=3).contains(&levels) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "PrimitiveAsset \"{asset_id}\" Subdivision levels must be from 1 through 3."
                    ),
                });
            }
            PrimitiveModifierNode::Subdivision { levels }
        }
        "Smooth" => {
            validate_primitive_child_attributes(tag, &["angle"], asset_id, line)?;
            let angle = parse_finite_primitive_number(tag, "angle", asset_id, line)?;
            if !(0.0..=180.0).contains(&angle) {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "PrimitiveAsset \"{asset_id}\" Smooth angle must be from 0 through 180 degrees."
                    ),
                });
            }
            PrimitiveModifierNode::Smooth { angle }
        }
        "WeightedNormals" => {
            validate_primitive_child_attributes(
                tag,
                &["strength", "keepSharpEdges"],
                asset_id,
                line,
            )?;
            let strength =
                parse_optional_nonnegative_primitive_number(tag, "strength", asset_id, line)?
                    .unwrap_or(1.0);
            if strength > 1.0 {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "PrimitiveAsset \"{asset_id}\" WeightedNormals strength must be from zero through one."
                    ),
                });
            }
            PrimitiveModifierNode::WeightedNormals {
                strength,
                keep_sharp_edges: parse_optional_primitive_bool(
                    tag,
                    "keepSharpEdges",
                    asset_id,
                    line,
                )?
                .unwrap_or(true),
            }
        }
        _ => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{asset_id}\" has unknown modifier <{name} />. Use MeshTransform, Taper, Bend, Twist, Subdivision, Smooth, or WeightedNormals."
                ),
            });
        }
    };
    Ok(modifier)
}

fn parse_primitive_mesh_build(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<PrimitiveMeshBuildNode, GraphParseError> {
    validate_primitive_child_attributes(
        tag,
        &["topology", "triangulation", "quality", "maxTriangles"],
        asset_id,
        line,
    )?;
    let topology = primitive_string_attribute(tag, "topology", "auto");
    if !matches!(topology.as_str(), "auto" | "triangles" | "quads") {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" MeshBuild topology must be auto, triangles, or quads."
            ),
        });
    }
    let triangulation = primitive_string_attribute(tag, "triangulation", "auto");
    if !matches!(
        triangulation.as_str(),
        "auto" | "shortestdiagonal" | "fixed"
    ) {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" MeshBuild triangulation must be auto, shortestDiagonal, or fixed."
            ),
        });
    }
    let quality = primitive_string_attribute(tag, "quality", "standard");
    if !matches!(
        quality.as_str(),
        "draft" | "standard" | "high" | "cinematic"
    ) {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" MeshBuild quality must be draft, standard, high, or cinematic."
            ),
        });
    }
    let max_triangles = parse_optional_primitive_u32(tag, "maxTriangles", asset_id, line)?;
    if max_triangles == Some(0) {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" MeshBuild maxTriangles must be greater than zero."
            ),
        });
    }
    Ok(PrimitiveMeshBuildNode {
        topology,
        triangulation,
        quality,
        max_triangles,
    })
}

fn parse_primitive_lod(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<PrimitiveLodNode, GraphParseError> {
    validate_primitive_child_attributes(
        tag,
        &["mode", "levels", "preserveSilhouette"],
        asset_id,
        line,
    )?;
    let mode = primitive_string_attribute(tag, "mode", "none");
    if !matches!(mode.as_str(), "none" | "auto") {
        return Err(GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{asset_id}\" LOD mode must be none or auto."),
        });
    }
    let levels = parse_optional_primitive_u32(tag, "levels", asset_id, line)?.unwrap_or(1);
    if !(1..=8).contains(&levels) {
        return Err(GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{asset_id}\" LOD levels must be from 1 through 8."),
        });
    }
    Ok(PrimitiveLodNode {
        mode,
        levels,
        preserve_silhouette: parse_optional_primitive_bool(
            tag,
            "preserveSilhouette",
            asset_id,
            line,
        )?
        .unwrap_or(true),
    })
}

fn validate_primitive_child_attributes(
    tag: &str,
    allowed: &[&str],
    asset_id: &str,
    line: usize,
) -> Result<(), GraphParseError> {
    for attribute in tag_attribute_names(tag) {
        if !allowed.contains(&attribute.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{asset_id}\" child does not support attribute \"{attribute}\"."
                ),
            });
        }
    }
    Ok(())
}

fn parse_primitive_axis(
    tag: &str,
    asset_id: &str,
    line: usize,
) -> Result<PrimitiveAxis, GraphParseError> {
    match strip_wrappers(&required_attr_value(tag, "axis", line)?)
        .to_ascii_lowercase()
        .as_str()
    {
        "x" => Ok(PrimitiveAxis::X),
        "y" => Ok(PrimitiveAxis::Y),
        "z" => Ok(PrimitiveAxis::Z),
        other => Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{asset_id}\" modifier axis=\"{other}\" is invalid. Use x, y, or z."
            ),
        }),
    }
}

fn parse_finite_primitive_number(
    tag: &str,
    attribute: &str,
    asset_id: &str,
    line: usize,
) -> Result<f32, GraphParseError> {
    let raw = required_attr_value(tag, attribute, line)?;
    let value = strip_wrappers(&raw)
        .parse::<f32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{asset_id}\" {attribute} must be a finite number."),
        })?;
    if !value.is_finite() {
        return Err(GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{asset_id}\" {attribute} must be a finite number."),
        });
    }
    Ok(value)
}

fn parse_optional_primitive_bool(
    tag: &str,
    attribute: &str,
    asset_id: &str,
    line: usize,
) -> Result<Option<bool>, GraphParseError> {
    attr_value(tag, attribute)
        .map(
            |raw| match strip_wrappers(&raw).to_ascii_lowercase().as_str() {
                "true" => Ok(true),
                "false" => Ok(false),
                _ => Err(GraphParseError {
                    line,
                    message: format!(
                        "PrimitiveAsset \"{asset_id}\" {attribute} must be true or false."
                    ),
                }),
            },
        )
        .transpose()
}

fn primitive_string_attribute(tag: &str, attribute: &str, default: &str) -> String {
    attr_value(tag, attribute)
        .map(|raw| strip_wrappers(&raw).to_ascii_lowercase())
        .unwrap_or_else(|| default.to_string())
}

fn parse_primitive_asset(
    tag: &str,
    id: &str,
    line: usize,
) -> Result<PrimitiveAssetNode, GraphParseError> {
    let shape = strip_wrappers(&required_attr_value(tag, "shape", line)?).to_ascii_lowercase();
    let mut allowed = match shape.as_str() {
        "box" | "wedge" => vec!["id", "shape", "size", "color"],
        "roundedbox" => vec!["id", "shape", "size", "radius", "segments", "color"],
        "ellipsoid" => vec!["id", "shape", "radii", "segments", "rings", "color"],
        "frustum" => vec!["id", "shape", "topSize", "bottomSize", "height", "color"],
        "sphere" => vec!["id", "shape", "radius", "segments", "rings", "color"],
        "capsule" => vec![
            "id", "shape", "radius", "height", "segments", "rings", "color",
        ],
        "plane" => vec!["id", "shape", "size", "segments", "color"],
        "cylinder" | "cone" => {
            vec!["id", "shape", "radius", "height", "segments", "color"]
        }
        "loft" => vec!["id", "shape", "color"],
        "ribbon" => vec!["id", "shape", "color"],
        _ => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" has unknown shape=\"{shape}\". Use box, sphere, capsule, plane, cylinder, cone, wedge, ellipsoid, frustum, roundedBox, loft, or ribbon."
                ),
            });
        }
    };
    allowed.extend([
        "material",
        "bevelRadius",
        "bevelSegments",
        "materialSeed",
        "collision",
        "collider",
        "colliderSize",
        "colliderRadius",
        "colliderHeight",
        "colliderScale",
        "colliderOffset",
        "colliderRotation",
        "colliderMargin",
        "collisionGroup",
        "collisionMask",
        "friction",
        "restitution",
        "density",
    ]);
    for attribute in tag_attribute_names(tag) {
        if !allowed.contains(&attribute.as_str()) {
            let guidance = match shape.as_str() {
                "sphere" => "Use radius=\"...\" and optional segments/rings.",
                "capsule" => "Use radius=\"...\", height=\"...\", and optional segments/rings.",
                "box" | "wedge" | "plane" => "Use size={...}.",
                "roundedbox" => "Use size={...}, radius=\"...\", and optional segments.",
                "ellipsoid" => "Use radii={[x,y,z]} and optional segments/rings.",
                "frustum" => "Use topSize={[x,z]}, bottomSize={[x,z]}, and height=\"...\".",
                "cylinder" | "cone" => "Use radius=\"...\", height=\"...\", and optional segments.",
                "loft" => "Use a nested <Loft> block with at least two <Section /> children.",
                "ribbon" => "Use a nested <Ribbon> block with at least two <PathPoint /> children.",
                _ => unreachable!(),
            };
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" with shape=\"{shape}\" does not support \"{attribute}\". {guidance}"
                ),
            });
        }
    }
    let geometry = match shape.as_str() {
        "box" => PrimitiveGeometry::Box {
            size: parse_primitive_vec::<3>(tag, "size", id, line)?,
        },
        "wedge" => PrimitiveGeometry::Wedge {
            size: parse_primitive_vec::<3>(tag, "size", id, line)?,
        },
        "roundedbox" => PrimitiveGeometry::RoundedBox {
            size: parse_primitive_vec::<3>(tag, "size", id, line)?,
            radius: parse_positive_primitive_number(tag, "radius", id, line)?,
            segments: parse_primitive_segments(tag, "segments", 3, id, line)?,
        },
        "ellipsoid" => {
            let segments = parse_primitive_segments(tag, "segments", 40, id, line)?;
            PrimitiveGeometry::Ellipsoid {
                radii: parse_primitive_vec::<3>(tag, "radii", id, line)?,
                segments,
                rings: parse_primitive_segments(tag, "rings", segments / 2, id, line)?,
            }
        }
        "frustum" => PrimitiveGeometry::Frustum {
            top_size: parse_primitive_vec::<2>(tag, "topSize", id, line)?,
            bottom_size: parse_primitive_vec::<2>(tag, "bottomSize", id, line)?,
            height: parse_positive_primitive_number(tag, "height", id, line)?,
        },
        "plane" => PrimitiveGeometry::Plane {
            size: parse_primitive_vec::<2>(tag, "size", id, line)?,
            segments: parse_primitive_segments(tag, "segments", 1, id, line)?,
        },
        "sphere" => {
            let segments = parse_primitive_segments(tag, "segments", 32, id, line)?;
            PrimitiveGeometry::Sphere {
                radius: parse_positive_primitive_number(tag, "radius", id, line)?,
                segments,
                rings: parse_primitive_segments(tag, "rings", segments / 2, id, line)?,
            }
        }
        "capsule" => {
            let segments = parse_primitive_segments(tag, "segments", 24, id, line)?;
            PrimitiveGeometry::Capsule {
                radius: parse_positive_primitive_number(tag, "radius", id, line)?,
                height: parse_positive_primitive_number(tag, "height", id, line)?,
                segments,
                rings: parse_primitive_segments(tag, "rings", 12, id, line)?,
            }
        }
        "cylinder" => PrimitiveGeometry::Cylinder {
            radius: parse_positive_primitive_number(tag, "radius", id, line)?,
            height: parse_positive_primitive_number(tag, "height", id, line)?,
            segments: parse_primitive_segments(tag, "segments", 32, id, line)?,
        },
        "cone" => PrimitiveGeometry::Cone {
            radius: parse_positive_primitive_number(tag, "radius", id, line)?,
            height: parse_positive_primitive_number(tag, "height", id, line)?,
            segments: parse_primitive_segments(tag, "segments", 32, id, line)?,
        },
        "loft" => PrimitiveGeometry::Loft {
            segments: 16,
            closed: true,
            cap_start: true,
            cap_end: true,
            sections: Vec::new(),
        },
        "ribbon" => PrimitiveGeometry::Ribbon {
            width: 0.1,
            thickness: 0.02,
            cap_start: true,
            cap_end: true,
            points: Vec::new(),
        },
        _ => unreachable!(),
    };
    let color = attr_value(tag, "color")
        .map(|value| parse_primitive_color(&value, id, line))
        .transpose()?
        .unwrap_or([1.0, 1.0, 1.0, 1.0]);
    let material = attr_value(tag, "material").map(|value| strip_wrappers(&value).to_string());
    let bevel_radius =
        parse_optional_nonnegative_primitive_number(tag, "bevelRadius", id, line)?.unwrap_or(0.0);
    let bevel_segments = parse_optional_primitive_u32(tag, "bevelSegments", id, line)?
        .unwrap_or(if bevel_radius > 0.0 { 3 } else { 0 });
    if bevel_radius > 0.0 && shape != "box" {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" bevel is currently supported for shape=\"box\" only."
            ),
        });
    }
    if bevel_radius > 0.0 && !(1..=8).contains(&bevel_segments) {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" bevelSegments must be from 1 through 8 when bevelRadius is greater than zero."
            ),
        });
    }
    if let PrimitiveGeometry::Box { size } = &geometry
        && bevel_radius * 2.0 >= size.iter().copied().fold(f32::INFINITY, f32::min)
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" bevelRadius must be less than half the smallest box dimension."
            ),
        });
    }
    if let PrimitiveGeometry::RoundedBox {
        size,
        radius,
        segments,
    } = &geometry
    {
        if *radius * 2.0 >= size.iter().copied().fold(f32::INFINITY, f32::min) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" roundedBox radius must be less than half the smallest dimension."
                ),
            });
        }
        if !(1..=16).contains(segments) {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" roundedBox segments must be from 1 through 16."
                ),
            });
        }
    }
    if let PrimitiveGeometry::Capsule { radius, height, .. } = &geometry
        && *height < *radius * 2.0
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" capsule height must be at least twice its radius."
            ),
        });
    }
    let material_seed = parse_optional_primitive_u64(tag, "materialSeed", id, line)?;
    let collision = parse_primitive_collision(tag, id, &geometry, line)?;
    Ok(PrimitiveAssetNode {
        id: id.to_string(),
        geometry,
        color,
        material,
        material_definition: None,
        bevel_radius,
        bevel_segments,
        material_seed,
        collision,
        modifiers: Vec::new(),
        mesh_build: PrimitiveMeshBuildNode::default(),
        lod: PrimitiveLodNode::default(),
    })
}

fn parse_primitive_collision(
    tag: &str,
    id: &str,
    geometry: &PrimitiveGeometry,
    line: usize,
) -> Result<PrimitiveCollisionNode, GraphParseError> {
    let mode = match attr_value(tag, "collision")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .as_deref()
        .unwrap_or("none")
    {
        "none" => PrimitiveCollisionMode::None,
        "solid" => PrimitiveCollisionMode::Solid,
        "sensor" => PrimitiveCollisionMode::Sensor,
        other => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" collision=\"{other}\" is invalid. Use none, solid, or sensor."
                ),
            });
        }
    };
    let collider = match attr_value(tag, "collider")
        .map(|value| strip_wrappers(&value).to_ascii_lowercase())
        .as_deref()
        .unwrap_or("auto")
    {
        "auto" => PrimitiveColliderShape::Auto,
        "box" => PrimitiveColliderShape::Box,
        "sphere" => PrimitiveColliderShape::Sphere,
        "capsule" => PrimitiveColliderShape::Capsule,
        "plane" => PrimitiveColliderShape::Plane,
        "cylinder" => PrimitiveColliderShape::Cylinder,
        "cone" => PrimitiveColliderShape::Cone,
        "convex" => PrimitiveColliderShape::Convex,
        "mesh" => PrimitiveColliderShape::Mesh,
        other => {
            return Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" collider=\"{other}\" is invalid. Use auto, box, sphere, capsule, plane, cylinder, cone, convex, or mesh."
                ),
            });
        }
    };
    let collision_configuration_present = [
        "collider",
        "colliderSize",
        "colliderRadius",
        "colliderHeight",
        "colliderScale",
        "colliderOffset",
        "colliderRotation",
        "colliderMargin",
        "collisionGroup",
        "collisionMask",
        "friction",
        "restitution",
        "density",
    ]
    .iter()
    .any(|attribute| attr_value(tag, attribute).is_some());
    if mode == PrimitiveCollisionMode::None && collision_configuration_present {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" has collision=\"none\" but also declares collider settings. Remove those settings or use collision=\"solid\" or \"sensor\"."
            ),
        });
    }

    let effective_shape = if collider == PrimitiveColliderShape::Auto {
        match geometry {
            PrimitiveGeometry::Box { .. } => PrimitiveColliderShape::Box,
            PrimitiveGeometry::Sphere { .. } => PrimitiveColliderShape::Sphere,
            PrimitiveGeometry::Capsule { .. } => PrimitiveColliderShape::Capsule,
            PrimitiveGeometry::Plane { .. } => PrimitiveColliderShape::Plane,
            PrimitiveGeometry::Cylinder { .. } => PrimitiveColliderShape::Cylinder,
            PrimitiveGeometry::Cone { .. } => PrimitiveColliderShape::Cone,
            PrimitiveGeometry::Wedge { .. } => PrimitiveColliderShape::Convex,
            PrimitiveGeometry::Ellipsoid { .. } => PrimitiveColliderShape::Sphere,
            PrimitiveGeometry::Frustum { .. } => PrimitiveColliderShape::Convex,
            PrimitiveGeometry::RoundedBox { .. } => PrimitiveColliderShape::Box,
            PrimitiveGeometry::Loft { .. }
            | PrimitiveGeometry::Ribbon { .. }
            | PrimitiveGeometry::HairCards { .. }
            | PrimitiveGeometry::HeadSurface { .. }
            | PrimitiveGeometry::Mesh { .. } => PrimitiveColliderShape::Convex,
        }
    } else {
        collider
    };
    let size = attr_value(tag, "colliderSize")
        .map(|raw| match effective_shape {
            PrimitiveColliderShape::Plane => {
                parse_positive_primitive_vec_value::<2>(&raw, "colliderSize", id, line)
                    .map(|value| value.to_vec())
            }
            PrimitiveColliderShape::Box
            | PrimitiveColliderShape::Convex
            | PrimitiveColliderShape::Mesh => {
                parse_positive_primitive_vec_value::<3>(&raw, "colliderSize", id, line)
                    .map(|value| value.to_vec())
            }
            _ => Err(GraphParseError {
                line,
                message: format!(
                    "PrimitiveAsset \"{id}\" colliderSize is only valid for box, plane, convex, or mesh colliders."
                ),
            }),
        })
        .transpose()?;
    let radius = parse_optional_positive_primitive_number(tag, "colliderRadius", id, line)?;
    if radius.is_some()
        && !matches!(
            effective_shape,
            PrimitiveColliderShape::Sphere
                | PrimitiveColliderShape::Capsule
                | PrimitiveColliderShape::Cylinder
                | PrimitiveColliderShape::Cone
        )
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" colliderRadius requires a sphere, cylinder, or cone collider."
            ),
        });
    }
    let height = parse_optional_positive_primitive_number(tag, "colliderHeight", id, line)?;
    if height.is_some()
        && !matches!(
            effective_shape,
            PrimitiveColliderShape::Capsule
                | PrimitiveColliderShape::Cylinder
                | PrimitiveColliderShape::Cone
        )
    {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" colliderHeight requires a capsule, cylinder, or cone collider."
            ),
        });
    }
    let scale = parse_optional_primitive_vec::<3>(tag, "colliderScale", id, line, true)?
        .unwrap_or([1.0; 3]);
    let offset = parse_optional_primitive_vec::<3>(tag, "colliderOffset", id, line, false)?
        .unwrap_or([0.0; 3]);
    let rotation = parse_optional_primitive_vec::<3>(tag, "colliderRotation", id, line, false)?
        .unwrap_or([0.0; 3]);
    let margin = parse_optional_nonnegative_primitive_number(tag, "colliderMargin", id, line)?
        .unwrap_or(0.0);
    let friction =
        parse_optional_nonnegative_primitive_number(tag, "friction", id, line)?.unwrap_or(0.5);
    let restitution =
        parse_optional_nonnegative_primitive_number(tag, "restitution", id, line)?.unwrap_or(0.0);
    if restitution > 1.0 {
        return Err(GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{id}\" restitution must be from zero through one."),
        });
    }
    let density =
        parse_optional_positive_primitive_number(tag, "density", id, line)?.unwrap_or(1.0);
    let group = parse_optional_primitive_u32(tag, "collisionGroup", id, line)?.unwrap_or(1);
    let mask = parse_optional_primitive_u32(tag, "collisionMask", id, line)?.unwrap_or(u32::MAX);
    Ok(PrimitiveCollisionNode {
        mode,
        collider,
        size,
        radius,
        height,
        scale,
        offset,
        rotation,
        margin,
        friction,
        restitution,
        density,
        group,
        mask,
    })
}

fn parse_positive_primitive_number(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<f32, GraphParseError> {
    let raw = required_attr_value(tag, attribute, line)?;
    let value = strip_wrappers(&raw)
        .parse::<f32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be a finite number greater than zero."
            ),
        })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be a finite number greater than zero."
            ),
        });
    }
    Ok(value)
}

fn parse_primitive_vec<const N: usize>(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<[f32; N], GraphParseError> {
    let raw = required_attr_value(tag, attribute, line)?;
    parse_primitive_vec_value(&raw, attribute, id, line, true)
}

fn parse_positive_primitive_vec_value<const N: usize>(
    raw: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<[f32; N], GraphParseError> {
    parse_primitive_vec_value(raw, attribute, id, line, true)
}

fn parse_optional_primitive_vec<const N: usize>(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
    positive: bool,
) -> Result<Option<[f32; N]>, GraphParseError> {
    attr_value(tag, attribute)
        .map(|raw| parse_primitive_vec_value(&raw, attribute, id, line, positive))
        .transpose()
}

fn parse_primitive_vec_value<const N: usize>(
    raw: &str,
    attribute: &str,
    id: &str,
    line: usize,
    positive: bool,
) -> Result<[f32; N], GraphParseError> {
    let value = strip_wrappers(raw).trim();
    let value = value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .unwrap_or(value);
    let parts = value
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>();
    if parts.len() != N {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must contain exactly {N} finite numbers."
            ),
        });
    }
    let mut out = [0.0; N];
    for (index, part) in parts.iter().enumerate() {
        out[index] = part.parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must contain exactly {N} finite numbers."
            ),
        })?;
        if !out[index].is_finite() || (positive && out[index] <= 0.0) {
            return Err(GraphParseError {
                line,
                message: if positive {
                    format!(
                        "PrimitiveAsset \"{id}\" {attribute} values must be finite and greater than zero."
                    )
                } else {
                    format!("PrimitiveAsset \"{id}\" {attribute} values must be finite.")
                },
            });
        }
    }
    Ok(out)
}

fn parse_optional_positive_primitive_number(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<Option<f32>, GraphParseError> {
    let Some(raw) = attr_value(tag, attribute) else {
        return Ok(None);
    };
    let value = strip_wrappers(&raw)
        .parse::<f32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be a finite number greater than zero."
            ),
        })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be a finite number greater than zero."
            ),
        });
    }
    Ok(Some(value))
}

fn parse_optional_nonnegative_primitive_number(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<Option<f32>, GraphParseError> {
    let Some(raw) = attr_value(tag, attribute) else {
        return Ok(None);
    };
    let value = strip_wrappers(&raw)
        .parse::<f32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be a finite number equal to or greater than zero."
            ),
        })?;
    if !value.is_finite() || value < 0.0 {
        return Err(GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be a finite number equal to or greater than zero."
            ),
        });
    }
    Ok(Some(value))
}

fn parse_optional_primitive_u32(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<Option<u32>, GraphParseError> {
    let Some(raw) = attr_value(tag, attribute) else {
        return Ok(None);
    };
    strip_wrappers(&raw)
        .parse::<u32>()
        .map(Some)
        .map_err(|_| GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{id}\" {attribute} must be an unsigned integer."),
        })
}

fn parse_optional_primitive_u64(
    tag: &str,
    attribute: &str,
    id: &str,
    line: usize,
) -> Result<Option<u64>, GraphParseError> {
    let Some(raw) = attr_value(tag, attribute) else {
        return Ok(None);
    };
    strip_wrappers(&raw)
        .parse::<u64>()
        .map(Some)
        .map_err(|_| GraphParseError {
            line,
            message: format!("Asset \"{id}\" {attribute} must be an unsigned integer."),
        })
}

fn parse_primitive_segments(
    tag: &str,
    attribute: &str,
    default: u32,
    id: &str,
    line: usize,
) -> Result<u32, GraphParseError> {
    let Some(raw) = attr_value(tag, attribute) else {
        return Ok(default);
    };
    let value = strip_wrappers(&raw)
        .parse::<u32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!(
                "PrimitiveAsset \"{id}\" {attribute} must be an integer from 3 through 256."
            ),
        })?;
    if !(3..=256).contains(&value) {
        return Err(GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{id}\" {attribute} must be from 3 through 256."),
        });
    }
    Ok(value)
}

fn parse_primitive_color(raw: &str, id: &str, line: usize) -> Result<[f32; 4], GraphParseError> {
    let hex = strip_wrappers(raw).trim().strip_prefix('#').unwrap_or("");
    if !matches!(hex.len(), 6 | 8) || !hex.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(GraphParseError {
            line,
            message: format!("PrimitiveAsset \"{id}\" color must use #RRGGBB or #RRGGBBAA."),
        });
    }
    let channel = |offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap() as f32 / 255.0;
    Ok([
        channel(0),
        channel(2),
        channel(4),
        if hex.len() == 8 { channel(6) } else { 1.0 },
    ])
}

pub(crate) fn tag_attribute_names(tag: &str) -> Vec<String> {
    let mut names = Vec::new();
    let bytes = tag.as_bytes();
    let mut index = tag.find(char::is_whitespace).unwrap_or(tag.len());
    while index < bytes.len() {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || matches!(bytes[index], b'/' | b'>') {
            break;
        }
        let start = index;
        while index < bytes.len()
            && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'_' | b'-'))
        {
            index += 1;
        }
        if start == index {
            index += 1;
            continue;
        }
        names.push(tag[start..index].to_string());
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b'=' {
            index += 1;
        }
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index < bytes.len() && bytes[index] == b'"' {
            index += 1;
            while index < bytes.len() && bytes[index] != b'"' {
                index += 1;
            }
            index = (index + 1).min(bytes.len());
        } else if index < bytes.len() && bytes[index] == b'{' {
            let mut depth = 1;
            index += 1;
            while index < bytes.len() && depth > 0 {
                match bytes[index] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                index += 1;
            }
        } else {
            while index < bytes.len() && !bytes[index].is_ascii_whitespace() && bytes[index] != b'>'
            {
                index += 1;
            }
        }
    }
    names
}

fn parse_scene_constraint_node(
    block: &str,
    line: usize,
) -> Result<SceneConstraintNode, GraphParseError> {
    let constraint_type = required_attr_value(block, "type", line)
        .or_else(|_| required_attr_value(block, "kind", line))?;
    let at_raw = attr_value(block, "at").or_else(|| attr_value(block, "from"));
    let at_seconds = at_raw
        .as_deref()
        .map(|value| parse_time_seconds(value, line, "Constraint.at"))
        .transpose()?
        .unwrap_or(0.0)
        .max(0.0);
    let duration_seconds = if let Some(value) = attr_value(block, "duration") {
        parse_time_seconds(&value, line, "Constraint.duration")?.max(0.0)
    } else if let Some(value) = attr_value(block, "to") {
        (parse_time_seconds(&value, line, "Constraint.to")? - at_seconds).max(0.0)
    } else {
        0.0
    };
    Ok(SceneConstraintNode {
        constraint_type: strip_wrappers(&constraint_type).to_ascii_lowercase(),
        source: strip_wrappers(&required_attr_value(block, "source", line)?).to_string(),
        target: strip_wrappers(&required_attr_value(block, "target", line)?).to_string(),
        at_ms: (at_seconds * 1000.0).round() as u64,
        duration_ms: (duration_seconds * 1000.0).round() as u64,
        solver: attr_value(block, "solver")
            .map(|value| strip_wrappers(&value).to_ascii_lowercase())
            .unwrap_or_else(|| "two_bone_ik".to_string()),
        weight: attr_value(block, "weight")
            .map(|value| strip_wrappers(&value).to_string())
            .unwrap_or_else(|| "1".to_string()),
    })
}

fn collect_tag_attr_values(
    input: &str,
    tag_name: &str,
    attr: &str,
) -> Result<Vec<String>, GraphParseError> {
    let mut cursor = 0usize;
    let mut values = Vec::new();
    while let Some(start) = find_open_tag_byte(input, tag_name, cursor) {
        let tag_end = find_tag_end_byte(input, start).ok_or_else(|| GraphParseError {
            line: line_of_byte(input, start),
            message: format!("Unclosed <{tag_name} ... /> tag."),
        })?;
        let tag = &input[start..=tag_end];
        if let Some(raw) = attr_value(tag, attr) {
            let value = strip_wrappers(&raw).to_string();
            if !value.is_empty() {
                values.push(value);
            }
        }
        cursor = tag_end + 1;
    }
    Ok(values)
}

fn infer_process_output_resource(
    process_open: &str,
    process_body: &str,
    line: usize,
) -> Result<String, GraphParseError> {
    if let Some(raw) =
        attr_value(process_open, "output").or_else(|| attr_value(process_open, "present"))
    {
        let id = strip_wrappers(&raw).to_string();
        if !id.is_empty() {
            return Ok(id);
        }
    }
    if let Some(id) = last_tag_attr(process_body, "Output", "id")? {
        return Ok(id);
    }
    if let Some(out) = last_pass_output_resource(process_body)? {
        return Ok(out);
    }
    if let Some(id) = last_tag_attr(process_body, "Tex", "id")? {
        return Ok(id);
    }
    Err(GraphParseError {
        line,
        message: "<Process> must declare output=\"...\" or contain an <Output>, <Pass out={...}>, or <Tex> that can be presented.".to_string(),
    })
}

fn last_tag_attr(
    input: &str,
    tag_name: &str,
    attr: &str,
) -> Result<Option<String>, GraphParseError> {
    let mut cursor = 0usize;
    let mut last = None;
    while let Some(start) = find_open_tag_byte(input, tag_name, cursor) {
        let tag_end = find_tag_end_byte(input, start).ok_or_else(|| GraphParseError {
            line: line_of_byte(input, start),
            message: format!("Unclosed <{tag_name} ... /> tag."),
        })?;
        let tag = &input[start..=tag_end];
        if let Some(raw) = attr_value(tag, attr) {
            let id = strip_wrappers(&raw).to_string();
            if !id.is_empty() {
                last = Some(id);
            }
        }
        cursor = tag_end + 1;
    }
    Ok(last)
}

fn last_pass_output_resource(input: &str) -> Result<Option<String>, GraphParseError> {
    let mut cursor = 0usize;
    let mut last = None;
    while let Some(start) = find_open_tag_byte(input, "Pass", cursor) {
        let tag_end = find_tag_end_byte(input, start).ok_or_else(|| GraphParseError {
            line: line_of_byte(input, start),
            message: "Unclosed <Pass ... /> tag.".to_string(),
        })?;
        let tag = &input[start..=tag_end];
        if let Some(raw) = attr_value(tag, "out")
            && let Some(id) = last_resource_id_from_attr(&raw)
        {
            last = Some(id);
        }
        cursor = tag_end + 1;
    }
    Ok(last)
}

fn last_resource_id_from_attr(raw: &str) -> Option<String> {
    let text = strip_wrappers(raw).trim();
    let mut quoted = Vec::<String>::new();
    let mut in_quote: Option<char> = None;
    let mut current = String::new();
    for ch in text.chars() {
        if let Some(quote) = in_quote {
            if ch == quote {
                if !current.trim().is_empty() {
                    quoted.push(current.trim().to_string());
                }
                current.clear();
                in_quote = None;
            } else {
                current.push(ch);
            }
            continue;
        }
        if ch == '"' || ch == '\'' {
            in_quote = Some(ch);
        }
    }
    if let Some(id) = quoted
        .into_iter()
        .rev()
        .find(|value| value != "tex" && value != "buf" && value != "id")
    {
        return Some(id);
    }

    text.trim_matches(|ch| matches!(ch, '[' | ']' | '{' | '}' | '"' | '\'' | ' '))
        .split(',')
        .filter_map(|part| {
            let token = part
                .trim()
                .trim_start_matches("tex:")
                .trim_start_matches("buf:")
                .trim_matches(|ch| matches!(ch, '"' | '\'' | ' '));
            (!token.is_empty()).then(|| token.to_string())
        })
        .next_back()
}

pub(crate) fn collect_self_closing_block(
    lines: &[&str],
    start: usize,
) -> Result<(String, usize), GraphParseError> {
    collect_tag_block(lines, start, '/', true)
}

pub(crate) fn is_self_closing_tag(block: &str) -> bool {
    let mut in_double_quote = false;
    let mut prev_char: Option<char> = None;
    for ch in block.chars() {
        if ch == '"' {
            in_double_quote = !in_double_quote;
            prev_char = Some(ch);
            continue;
        }
        if !in_double_quote && ch == '>' && prev_char == Some('/') {
            return true;
        }
        prev_char = Some(ch);
    }
    false
}

pub(crate) fn collect_tag_block(
    lines: &[&str],
    start: usize,
    end_char: char,
    requires_self_closing: bool,
) -> Result<(String, usize), GraphParseError> {
    let mut out = String::new();
    let mut in_double_quote = false;
    let mut prev_char: Option<char> = None;
    for (ix, line) in lines.iter().enumerate().skip(start) {
        let trimmed = line.trim();
        out.push_str(trimmed);
        out.push('\n');
        for ch in trimmed.chars() {
            if ch == '"' {
                in_double_quote = !in_double_quote;
                continue;
            }
            if in_double_quote {
                prev_char = Some(ch);
                continue;
            }
            if requires_self_closing {
                // detect '/>' outside quoted attributes only
                if ch == '>' && prev_char == Some('/') {
                    return Ok((out, ix));
                }
            } else if ch == end_char {
                return Ok((out, ix));
            }
            prev_char = Some(ch);
        }
        prev_char = Some('\n');
    }
    Err(GraphParseError {
        line: start + 1,
        message: "Tag block is not closed.".to_string(),
    })
}

pub(crate) fn starts_open_tag(line: &str, tag_name: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix('<') else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(tag_name) else {
        return false;
    };
    matches!(
        rest.chars().next(),
        None | Some(' ') | Some('\t') | Some('\r') | Some('\n') | Some('>') | Some('/')
    )
}

pub(crate) fn starts_close_tag(line: &str, tag_name: &str) -> bool {
    let Some(rest) = line.trim_start().strip_prefix("</") else {
        return false;
    };
    rest.strip_prefix(tag_name)
        .is_some_and(|rest| rest.trim_start().starts_with('>'))
}

fn parse_layer_block(lines: &[&str], start: usize) -> Result<(LayerNode, usize), GraphParseError> {
    let (open_tag, open_end_ix) = collect_tag_block(lines, start, '>', false)?;
    let close_ix = find_matching_close_tag(lines, open_end_ix + 1, "Layer")?;
    let id = strip_wrappers(&required_attr_value(&open_tag, "id", start + 1)?).to_string();
    let mut effects = Vec::<EffectNode>::new();
    let mut i = open_end_ix + 1;

    while i < close_ix {
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('{') {
            i += 1;
            continue;
        }
        if starts_open_tag(line, "Effect") {
            let (tag, end_ix) = collect_self_closing_block(lines, i)?;
            effects.push(parse_effect_node(&tag, i + 1)?);
            i = end_ix + 1;
            continue;
        }
        return Err(GraphParseError {
            line: i + 1,
            message: format!("<Layer> only accepts <Effect /> children for now, got: {line}"),
        });
    }

    Ok((LayerNode { id, effects }, close_ix))
}

fn parse_effect_node(block: &str, line: usize) -> Result<EffectNode, GraphParseError> {
    let id = attr_value(block, "id").map(|v| strip_wrappers(&v).to_string());
    let effect_type = attr_value(block, "type")
        .or_else(|| attr_value(block, "effect"))
        .map(|v| strip_wrappers(&v).to_string())
        .ok_or_else(|| GraphParseError {
            line,
            message: "Effect requires type=\"...\".".to_string(),
        })?;
    let mut params = Vec::<PassParam>::new();
    for key in [
        "sigma",
        "amount",
        "hue",
        "saturation",
        "lightness",
        "alpha",
        "brightness",
        "contrast",
        "opacity",
    ] {
        if let Some(value) = attr_value(block, key) {
            params.push(PassParam {
                key: key.to_string(),
                value: strip_wrappers(&value).to_string(),
            });
        }
    }
    Ok(EffectNode {
        id,
        r#type: effect_type,
        params,
    })
}

pub(crate) fn find_matching_close_tag(
    lines: &[&str],
    start: usize,
    tag_name: &str,
) -> Result<usize, GraphParseError> {
    let mut depth = 0usize;
    let mut ix = start;
    while ix < lines.len() {
        let trimmed = lines[ix].trim_start();
        if starts_close_tag(trimmed, tag_name) {
            if depth == 0 {
                return Ok(ix);
            }
            depth = depth.saturating_sub(1);
            ix += 1;
            continue;
        }
        if starts_open_tag(trimmed, tag_name) {
            let (tag, end_ix) = collect_tag_block(lines, ix, '>', false)?;
            if !is_self_closing_tag(&tag) {
                depth = depth.saturating_add(1);
            }
            ix = end_ix + 1;
            continue;
        }
        ix += 1;
    }
    Err(GraphParseError {
        line: start + 1,
        message: format!("Missing </{tag_name}> closing tag."),
    })
}

fn parse_input_node(block: &str, line: usize) -> Result<InputNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let input_type = attr_value(block, "type")
        .as_deref()
        .map(|raw| parse_input_type(raw, line, "type"))
        .transpose()?
        .unwrap_or(InputType::Video);
    let from = attr_value(block, "from").map(|v| strip_wrappers(&v).to_string());
    let fmt = attr_value(block, "fmt")
        .map(|v| parse_texture_format(&v, line, "fmt"))
        .transpose()?;
    let size = attr_value(block, "size")
        .as_deref()
        .map(|v| parse_size(v, line, "size"))
        .transpose()?;
    let color_space = attr_value(block, "colorSpace")
        .or_else(|| attr_value(block, "color_space"))
        .map(|v| parse_color_space(&v, line, "colorSpace"))
        .transpose()?;
    let alpha = attr_value(block, "alpha")
        .map(|v| parse_alpha_mode(&v, line, "alpha"))
        .transpose()?;

    Ok(InputNode {
        id,
        r#type: input_type,
        from,
        fmt,
        size,
        color_space,
        alpha,
    })
}

fn parse_clip_node(block: &str, line: usize) -> Result<InputNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let input_type = attr_value(block, "type")
        .as_deref()
        .map(|raw| parse_input_type(raw, line, "type"))
        .transpose()?
        .unwrap_or(InputType::Video);
    let from = attr_value(block, "src")
        .or_else(|| attr_value(block, "from"))
        .map(|v| strip_wrappers(&v).to_string());
    let fmt = attr_value(block, "fmt")
        .map(|v| parse_texture_format(&v, line, "fmt"))
        .transpose()?;
    let size = attr_value(block, "size")
        .as_deref()
        .map(|v| parse_size(v, line, "size"))
        .transpose()?;

    Ok(InputNode {
        id,
        r#type: input_type,
        from,
        fmt,
        size,
        color_space: None,
        alpha: None,
    })
}

fn parse_tex_node(block: &str, line: usize) -> Result<TexNode, GraphParseError> {
    let id = required_attr_value(block, "id", line)?;
    let fmt = required_attr_value(block, "fmt", line)?;
    let from = attr_value(block, "from")
        .or_else(|| attr_value(block, "src"))
        .map(|v| strip_wrappers(&v).to_string());
    let input = attr_value(block, "input").map(|v| strip_wrappers(&v).to_string());
    let size = attr_value(block, "size")
        .as_deref()
        .map(|v| parse_size(v, line, "size"))
        .transpose()?;
    let usage = attr_value(block, "usage")
        .as_deref()
        .map(|v| parse_tex_usage_array(v, line, "usage"))
        .transpose()?
        .unwrap_or_default();
    let transient = attr_value(block, "transient")
        .as_deref()
        .map(|v| parse_bool(v, line, "transient"))
        .transpose()?;
    let pingpong = attr_value(block, "pingpong").map(|v| strip_wrappers(&v).to_string());

    Ok(TexNode {
        id: strip_wrappers(&id).to_string(),
        fmt: parse_texture_format(&fmt, line, "fmt")?,
        from,
        input,
        size,
        usage,
        transient,
        pingpong,
    })
}

fn parse_buffer_node(block: &str, line: usize) -> Result<BufferNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let elem_raw = required_attr_value_any(block, &["elemType", "elem_type"], line)?;
    let elem_type = parse_buffer_elem_type(&elem_raw, line, "elemType")?;
    let length = attr_value(block, "length")
        .as_deref()
        .map(|v| parse_u32(v, line, "length"))
        .transpose()?;
    let stride = attr_value(block, "stride")
        .as_deref()
        .map(|v| parse_u32(v, line, "stride"))
        .transpose()?;
    let usage = attr_value(block, "usage")
        .as_deref()
        .map(|v| parse_buffer_usage_array(v, line, "usage"))
        .transpose()?
        .unwrap_or_default();
    let transient = attr_value(block, "transient")
        .as_deref()
        .map(|v| parse_bool(v, line, "transient"))
        .transpose()?;
    let pingpong = attr_value(block, "pingpong").map(|v| strip_wrappers(&v).to_string());

    Ok(BufferNode {
        id,
        elem_type,
        length,
        stride,
        usage,
        transient,
        pingpong,
    })
}

fn parse_output_node(block: &str, line: usize) -> Result<OutputNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let from = attr_value(block, "from").map(|v| strip_wrappers(&v).to_string());
    let to = attr_value(block, "to")
        .as_deref()
        .map(|v| parse_output_target(v, line, "to"))
        .transpose()?
        .unwrap_or(OutputTarget::Screen);
    let fmt = attr_value(block, "fmt")
        .map(|v| parse_texture_format(&v, line, "fmt"))
        .transpose()?;
    let size = attr_value(block, "size")
        .as_deref()
        .map(|v| parse_size(v, line, "size"))
        .transpose()?;
    let color_space = attr_value(block, "colorSpace")
        .or_else(|| attr_value(block, "color_space"))
        .map(|v| parse_color_space(&v, line, "colorSpace"))
        .transpose()?;
    let alpha = attr_value(block, "alpha")
        .map(|v| parse_alpha_mode(&v, line, "alpha"))
        .transpose()?;

    Ok(OutputNode {
        id,
        from,
        to,
        fmt,
        size,
        color_space,
        alpha,
        is_process_implicit: false,
    })
}

fn parse_pass_node(block: &str, line: usize) -> Result<PassNode, GraphParseError> {
    let id = strip_wrappers(&required_attr_value(block, "id", line)?).to_string();
    let kind = attr_value(block, "kind")
        .as_deref()
        .map(|v| parse_pass_kind(v, line, "kind"))
        .transpose()?
        .unwrap_or(PassKind::Compute);
    let role = attr_value(block, "role")
        .as_deref()
        .map(|v| parse_pass_role(v, line, "role"))
        .transpose()?;
    let kernel = attr_value(block, "kernel").map(|v| strip_wrappers(&v).to_string());
    let mode = attr_value(block, "mode").map(|v| strip_wrappers(&v).to_string());
    let effect = strip_wrappers(&required_attr_value(block, "effect", line)?).to_string();
    let transition = attr_value(block, "transition")
        .as_deref()
        .map(|v| parse_transition_mode(v, line, "transition"))
        .transpose()?;
    let transition_fallback = attr_value(block, "transitionFallback")
        .or_else(|| attr_value(block, "transition_fallback"))
        .as_deref()
        .map(|v| parse_transition_fallback(v, line, "transitionFallback"))
        .transpose()?;
    let transition_easing = attr_value(block, "transitionEasing")
        .or_else(|| attr_value(block, "transition_easing"))
        .as_deref()
        .map(|v| parse_transition_easing(v, line, "transitionEasing"))
        .transpose()?;
    let transition_clips = attr_value(block, "transitionClips")
        .or_else(|| attr_value(block, "transition_clips"))
        .as_deref()
        .map(|v| parse_transition_clips(v, line, "transitionClips"))
        .transpose()?;
    let inputs = parse_resource_ref_array(&required_attr_value(block, "in", line)?, line, "in")?;
    let outputs = parse_resource_ref_array(&required_attr_value(block, "out", line)?, line, "out")?;
    let params = parse_params(block);
    let mask = attr_value(block, "mask").map(|v| strip_wrappers(&v).to_string());
    let mask_mode = attr_value(block, "maskMode")
        .or_else(|| attr_value(block, "mask_mode"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "alpha".to_string());
    let mask_invert = attr_value(block, "maskInvert")
        .or_else(|| attr_value(block, "mask_invert"))
        .or_else(|| attr_value(block, "invertMask"))
        .or_else(|| attr_value(block, "invert_mask"))
        .map(|v| strip_wrappers(&v).to_string())
        .unwrap_or_else(|| "false".to_string());
    let iterate = attr_value(block, "iterate")
        .as_deref()
        .map(|v| parse_quality_u32(v, line, "iterate"))
        .transpose()?;
    let pingpong = attr_value(block, "pingpong").map(|v| strip_wrappers(&v).to_string());
    let cache = attr_value(block, "cache")
        .as_deref()
        .map(|v| parse_pass_cache(v, line, "cache"))
        .transpose()?;
    let blend = attr_value(block, "blend")
        .as_deref()
        .map(|v| parse_blend_mode(v, line, "blend"))
        .transpose()?;
    let load_op = attr_value(block, "loadOp")
        .or_else(|| attr_value(block, "load_op"))
        .as_deref()
        .map(|v| parse_load_op(v, line, "loadOp"))
        .transpose()?;
    let store_op = attr_value(block, "storeOp")
        .or_else(|| attr_value(block, "store_op"))
        .as_deref()
        .map(|v| parse_store_op(v, line, "storeOp"))
        .transpose()?;

    Ok(PassNode {
        id,
        kind,
        role,
        kernel,
        mode,
        effect,
        transition,
        transition_fallback,
        transition_easing,
        transition_clips,
        inputs,
        outputs,
        params,
        mask,
        mask_mode,
        mask_invert,
        iterate,
        pingpong,
        cache,
        blend,
        load_op,
        store_op,
    })
}

fn parse_present_node(block: &str, line: usize) -> Result<PresentNode, GraphParseError> {
    let from = strip_wrappers(&required_attr_value(block, "from", line)?).to_string();
    let to = attr_value(block, "to")
        .as_deref()
        .map(|v| parse_present_target(v, line, "to"))
        .transpose()?
        .unwrap_or(PresentTarget::Screen);
    let vsync = attr_value(block, "vsync")
        .as_deref()
        .map(|v| parse_bool(v, line, "vsync"))
        .transpose()?;
    Ok(PresentNode { from, to, vsync })
}

fn parse_params(block: &str) -> Vec<PassParam> {
    let Some(start_ix) = block.find("params={{") else {
        return Vec::new();
    };
    let after = &block[start_ix + "params={{".len()..];
    let Some(end_ix) = after.find("}}") else {
        return Vec::new();
    };
    let body = &after[..end_ix];
    let mut cleaned_body = String::new();
    for line in body.lines() {
        let line = line.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if !cleaned_body.is_empty() {
            cleaned_body.push(' ');
        }
        cleaned_body.push_str(line);
    }
    let mut params = Vec::new();
    for entry in split_top_level_csv(&cleaned_body) {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some(colon_ix) = entry.find(':') else {
            continue;
        };
        let key = entry[..colon_ix].trim().trim_end_matches(',');
        let value = entry[colon_ix + 1..].trim().trim_end_matches(',');
        if key.is_empty() || value.is_empty() {
            continue;
        }
        params.push(PassParam {
            key: key.to_string(),
            value: value.to_string(),
        });
    }
    params
}

fn split_top_level_csv(input: &str) -> Vec<String> {
    let mut out = Vec::<String>::new();
    let mut cur = String::new();
    let mut paren_depth = 0_i32;
    let mut brace_depth = 0_i32;
    let mut bracket_depth = 0_i32;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut escape = false;

    for ch in input.chars() {
        if escape {
            cur.push(ch);
            escape = false;
            continue;
        }
        if ch == '\\' {
            cur.push(ch);
            escape = true;
            continue;
        }
        if in_single_quote {
            cur.push(ch);
            if ch == '\'' {
                in_single_quote = false;
            }
            continue;
        }
        if in_double_quote {
            cur.push(ch);
            if ch == '"' {
                in_double_quote = false;
            }
            continue;
        }
        match ch {
            '\'' => {
                in_single_quote = true;
                cur.push(ch);
            }
            '"' => {
                in_double_quote = true;
                cur.push(ch);
            }
            '(' => {
                paren_depth += 1;
                cur.push(ch);
            }
            ')' => {
                paren_depth -= 1;
                cur.push(ch);
            }
            '{' => {
                brace_depth += 1;
                cur.push(ch);
            }
            '}' => {
                brace_depth -= 1;
                cur.push(ch);
            }
            '[' => {
                bracket_depth += 1;
                cur.push(ch);
            }
            ']' => {
                bracket_depth -= 1;
                cur.push(ch);
            }
            ',' if paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 => {
                let token = cur.trim();
                if !token.is_empty() {
                    out.push(token.to_string());
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    let token = cur.trim();
    if !token.is_empty() {
        out.push(token.to_string());
    }
    out
}

fn parse_fps(block: &str, line: usize) -> Result<f32, GraphParseError> {
    let raw = required_attr_value(block, "fps", line)?;
    let text = strip_wrappers(&raw);
    let fps = text.parse::<f32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid fps value: {}", text),
    })?;
    Ok(fps)
}

pub(crate) fn parse_duration_ms(
    block: &str,
    line: usize,
    default_ms: u64,
) -> Result<u64, GraphParseError> {
    let Some(raw) = attr_value(block, "duration") else {
        return Ok(default_ms);
    };
    let text = strip_wrappers(&raw);
    if let Some(ms) = text.strip_suffix("ms") {
        let val = ms.trim().parse::<f64>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid duration value: {}", text),
        })?;
        return Ok(val.max(0.0).round() as u64);
    }
    if let Some(sec) = text.strip_suffix('s') {
        let val = sec.trim().parse::<f64>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid duration value: {}", text),
        })?;
        return Ok((val.max(0.0) * 1000.0).round() as u64);
    }
    let val = text.parse::<f64>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid duration value: {}", text),
    })?;
    Ok((val.max(0.0) * 1000.0).round() as u64)
}

pub(crate) fn parse_time_seconds(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<f32, GraphParseError> {
    let text = strip_wrappers(raw);
    if let Some(ms) = text.strip_suffix("ms") {
        let val = ms.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} time value: {text}"),
        })?;
        return Ok((val / 1000.0).max(0.0));
    }
    if let Some(sec) = text.strip_suffix('s') {
        let val = sec.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} time value: {text}"),
        })?;
        return Ok(val.max(0.0));
    }
    let val = text.parse::<f32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field} time value: {text}"),
    })?;
    Ok(val.max(0.0))
}

pub(crate) fn parse_signed_time_ms(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<i64, GraphParseError> {
    let text = strip_wrappers(raw);
    if let Some(ms) = text.strip_suffix("ms") {
        let val = ms.trim().parse::<f64>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} time value: {text}"),
        })?;
        return Ok(val.round() as i64);
    }
    if let Some(sec) = text.strip_suffix('s') {
        let val = sec.trim().parse::<f64>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} time value: {text}"),
        })?;
        return Ok((val * 1000.0).round() as i64);
    }
    let val = text.parse::<f64>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field} time value: {text}"),
    })?;
    Ok((val * 1000.0).round() as i64)
}

fn parse_graph_apply_scope(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<GraphApplyScope, GraphParseError> {
    match strip_wrappers(raw).to_ascii_lowercase().as_str() {
        "clip" => Ok(GraphApplyScope::Clip),
        "graph" => Ok(GraphApplyScope::Graph),
        other => Err(GraphParseError {
            line,
            message: format!(
                "Invalid {} '{}'. Expected one of: clip, graph.",
                field, other
            ),
        }),
    }
}

pub(crate) fn parse_size(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<(u32, u32), GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{} must be an array [width,height].", field),
        })?;
    let mut parts = inner.split(',').map(str::trim);
    let Some(w) = parts.next() else {
        return Err(GraphParseError {
            line,
            message: format!("{} is missing width.", field),
        });
    };
    let Some(h) = parts.next() else {
        return Err(GraphParseError {
            line,
            message: format!("{} is missing height.", field),
        });
    };
    let width = w.parse::<u32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {} width: {}", field, w),
    })?;
    let height = h.parse::<u32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {} height: {}", field, h),
    })?;
    Ok((width, height))
}

fn parse_resource_ref_array(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<Vec<ResourceRef>, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{} must be an array of resource refs.", field),
        })?;
    let mut out = Vec::<ResourceRef>::new();
    for item in split_top_level_csv(inner) {
        let token = item.trim();
        if token.is_empty() {
            continue;
        }
        out.push(parse_resource_ref(token, line, field)?);
    }
    if out.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("{} cannot be empty.", field),
        });
    }
    Ok(out)
}

fn parse_string_array(raw: &str, line: usize, field: &str) -> Result<Vec<String>, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{field} must be an array of strings."),
        })?;
    let values = split_top_level_csv(inner)
        .into_iter()
        .map(|value| strip_wrappers(value.trim()).trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("{field} cannot be empty."),
        });
    }
    Ok(values)
}

fn parse_resource_ref(
    token: &str,
    line: usize,
    field: &str,
) -> Result<ResourceRef, GraphParseError> {
    let trimmed = token.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        let body = &trimmed[1..trimmed.len() - 1];
        let entries = parse_inline_object_entries(body);
        if let Some(tex_raw) = entries.get("tex") {
            let sample = entries
                .get("sample")
                .map(|raw| parse_sample_config(raw, line, field))
                .transpose()?;
            return Ok(ResourceRef::Tex {
                tex: strip_wrappers(tex_raw).to_string(),
                sample,
            });
        }
        if let Some(buf_raw) = entries.get("buf") {
            return Ok(ResourceRef::Buffer {
                buf: strip_wrappers(buf_raw).to_string(),
            });
        }
        if let Some(id_raw) = entries.get("id") {
            return Ok(ResourceRef::Id {
                id: strip_wrappers(id_raw).to_string(),
            });
        }
        if let Some(target_raw) = entries.get("target") {
            return Ok(ResourceRef::Id {
                id: strip_wrappers(target_raw).to_string(),
            });
        }
        return Err(GraphParseError {
            line,
            message: format!(
                "{} object ref must contain one of: tex|buf|id|target",
                field
            ),
        });
    }

    let id = strip_wrappers(trimmed);
    if id.is_empty() {
        return Err(GraphParseError {
            line,
            message: format!("{} contains an empty resource id.", field),
        });
    }
    Ok(ResourceRef::Id { id: id.to_string() })
}

fn parse_sample_config(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<SampleConfig, GraphParseError> {
    let raw_trimmed = raw.trim();
    let entries = if raw_trimmed.starts_with('{') && raw_trimmed.ends_with('}') {
        parse_inline_object_entries(&raw_trimmed[1..raw_trimmed.len() - 1])
    } else {
        let text = strip_wrappers(raw).trim();
        if text.is_empty() || !text.contains(':') {
            return Err(GraphParseError {
                line,
                message: format!("{}.sample must be an object.", field),
            });
        }
        parse_inline_object_entries(text)
    };
    let filter = entries
        .get("filter")
        .map(|raw| parse_sample_filter(raw, line, "sample.filter"))
        .transpose()?;
    let address = entries
        .get("address")
        .map(|raw| parse_sample_address(raw, line, "sample.address"))
        .transpose()?;
    Ok(SampleConfig { filter, address })
}

fn parse_inline_object_entries(body: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::<String, String>::new();
    for entry in split_top_level_csv(body) {
        let Some((k, v)) = entry.split_once(':') else {
            continue;
        };
        let key = strip_wrappers(k).trim().to_ascii_lowercase();
        if key.is_empty() {
            continue;
        }
        map.insert(key, v.trim().to_string());
    }
    map
}

fn parse_tex_usage_array(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<Vec<TexUsage>, GraphParseError> {
    parse_enum_array(raw, line, field, parse_tex_usage)
}

fn parse_buffer_usage_array(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<Vec<BufferUsage>, GraphParseError> {
    parse_enum_array(raw, line, field, parse_buffer_usage)
}

fn parse_enum_array<T>(
    raw: &str,
    line: usize,
    field: &str,
    parser: fn(&str, usize, &str) -> Result<T, GraphParseError>,
) -> Result<Vec<T>, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{} must be an array.", field),
        })?;
    let mut out = Vec::new();
    for chunk in split_top_level_csv(inner) {
        let token = chunk.trim();
        if token.is_empty() {
            continue;
        }
        out.push(parser(token, line, field)?);
    }
    Ok(out)
}

fn parse_quality_u32(raw: &str, line: usize, field: &str) -> Result<Quality<u32>, GraphParseError> {
    let raw_trimmed = raw.trim();
    if raw_trimmed.starts_with('{') && raw_trimmed.ends_with('}') {
        let entries = parse_inline_object_entries(&raw_trimmed[1..raw_trimmed.len() - 1]);
        let Some(preview_raw) = entries.get("preview") else {
            return Err(GraphParseError {
                line,
                message: format!("{} quality object missing preview.", field),
            });
        };
        let final_raw = entries.get("final").ok_or_else(|| GraphParseError {
            line,
            message: format!("{} quality object missing final.", field),
        })?;
        return Ok(Quality::Split {
            preview: parse_u32(preview_raw, line, "preview")?,
            r#final: parse_u32(final_raw, line, "final")?,
        });
    }

    let text = strip_wrappers(raw).trim();
    if text.contains(':') {
        let entries = parse_inline_object_entries(text);
        let Some(preview_raw) = entries.get("preview") else {
            return Err(GraphParseError {
                line,
                message: format!("{} quality object missing preview.", field),
            });
        };
        let final_raw = entries.get("final").ok_or_else(|| GraphParseError {
            line,
            message: format!("{} quality object missing final.", field),
        })?;
        return Ok(Quality::Split {
            preview: parse_u32(preview_raw, line, "preview")?,
            r#final: parse_u32(final_raw, line, "final")?,
        });
    }

    Ok(Quality::Uniform(parse_u32(text, line, field)?))
}

fn parse_u32(raw: &str, line: usize, field: &str) -> Result<u32, GraphParseError> {
    let text = strip_wrappers(raw);
    text.parse::<u32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {} value: {}", field, text),
    })
}

pub(crate) fn parse_bool(raw: &str, line: usize, field: &str) -> Result<bool, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} boolean value: {}", field, other),
        }),
    }
}

fn parse_texture_format(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<TextureFormat, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "rgba8" => Ok(TextureFormat::Rgba8),
        "rgba8unorm" => Ok(TextureFormat::Rgba8Unorm),
        "rgba8unorm-srgb" => Ok(TextureFormat::Rgba8UnormSrgb),
        "bgra8unorm" => Ok(TextureFormat::Bgra8Unorm),
        "bgra8unorm-srgb" => Ok(TextureFormat::Bgra8UnormSrgb),
        "rgba16f" => Ok(TextureFormat::Rgba16f),
        "rgba32f" => Ok(TextureFormat::Rgba32f),
        "r16f" => Ok(TextureFormat::R16f),
        "r32f" => Ok(TextureFormat::R32f),
        "depth24plus" => Ok(TextureFormat::Depth24plus),
        "depth32f" => Ok(TextureFormat::Depth32f),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} format: {}", field, other),
        }),
    }
}

fn parse_color_space(raw: &str, line: usize, field: &str) -> Result<ColorSpace, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "srgb" => Ok(ColorSpace::Srgb),
        "linear-srgb" => Ok(ColorSpace::LinearSrgb),
        "display-p3" => Ok(ColorSpace::DisplayP3),
        "rec709" => Ok(ColorSpace::Rec709),
        "rec2020" => Ok(ColorSpace::Rec2020),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_alpha_mode(raw: &str, line: usize, field: &str) -> Result<AlphaMode, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "straight" => Ok(AlphaMode::Straight),
        "premul" => Ok(AlphaMode::Premul),
        "opaque" => Ok(AlphaMode::Opaque),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_input_type(raw: &str, line: usize, field: &str) -> Result<InputType, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "video" => Ok(InputType::Video),
        "image" => Ok(InputType::Image),
        "mask" => Ok(InputType::Mask),
        "depth" => Ok(InputType::Depth),
        "normal" => Ok(InputType::Normal),
        "motion" => Ok(InputType::Motion),
        "audio" => Ok(InputType::Audio),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_tex_usage(raw: &str, line: usize, field: &str) -> Result<TexUsage, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "sampled" => Ok(TexUsage::Sampled),
        "storage" => Ok(TexUsage::Storage),
        "color-attachment" => Ok(TexUsage::ColorAttachment),
        "depth-stencil-attachment" => Ok(TexUsage::DepthStencilAttachment),
        "copy-src" => Ok(TexUsage::CopySrc),
        "copy-dst" => Ok(TexUsage::CopyDst),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_buffer_usage(raw: &str, line: usize, field: &str) -> Result<BufferUsage, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "uniform" => Ok(BufferUsage::Uniform),
        "storage" => Ok(BufferUsage::Storage),
        "vertex" => Ok(BufferUsage::Vertex),
        "index" => Ok(BufferUsage::Index),
        "indirect" => Ok(BufferUsage::Indirect),
        "copy-src" => Ok(BufferUsage::CopySrc),
        "copy-dst" => Ok(BufferUsage::CopyDst),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_buffer_elem_type(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<BufferElemType, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "f32" => Ok(BufferElemType::F32),
        "u32" => Ok(BufferElemType::U32),
        "i32" => Ok(BufferElemType::I32),
        "vec2f" => Ok(BufferElemType::Vec2f),
        "vec4f" => Ok(BufferElemType::Vec4f),
        "mat4f" => Ok(BufferElemType::Mat4f),
        "struct" => Ok(BufferElemType::Struct),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_pass_kind(raw: &str, line: usize, field: &str) -> Result<PassKind, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "compute" => Ok(PassKind::Compute),
        "render" => Ok(PassKind::Render),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_pass_role(raw: &str, line: usize, field: &str) -> Result<PassRole, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "effect" => Ok(PassRole::Effect),
        "transition" => Ok(PassRole::Transition),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_pass_cache(raw: &str, line: usize, field: &str) -> Result<PassCache, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "none" => Ok(PassCache::None),
        "frame" => Ok(PassCache::Frame),
        "static" => Ok(PassCache::Static),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_blend_mode(raw: &str, line: usize, field: &str) -> Result<BlendMode, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "replace" => Ok(BlendMode::Replace),
        "add" => Ok(BlendMode::Add),
        "screen" => Ok(BlendMode::Screen),
        "multiply" => Ok(BlendMode::Multiply),
        "over" => Ok(BlendMode::Over),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_transition_mode(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<PassTransitionMode, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "auto" => Ok(PassTransitionMode::Auto),
        "off" => Ok(PassTransitionMode::Off),
        "force" => Ok(PassTransitionMode::Force),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_transition_fallback(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<PassTransitionFallback, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "under" => Ok(PassTransitionFallback::Under),
        "prev" => Ok(PassTransitionFallback::Prev),
        "next" => Ok(PassTransitionFallback::Next),
        "skip" => Ok(PassTransitionFallback::Skip),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_transition_easing(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<PassTransitionEasing, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "linear" => Ok(PassTransitionEasing::Linear),
        "ease-in" => Ok(PassTransitionEasing::EaseIn),
        "ease-out" => Ok(PassTransitionEasing::EaseOut),
        "ease-in-out" => Ok(PassTransitionEasing::EaseInOut),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_transition_clips(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<PassTransitionClips, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "overlap" => Ok(PassTransitionClips::Overlap),
        "non-overlap" => Ok(PassTransitionClips::NonOverlap),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_load_op(raw: &str, line: usize, field: &str) -> Result<LoadOp, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    if text.starts_with('{') && text.ends_with('}') {
        let entries = parse_inline_object_entries(&text[1..text.len() - 1]);
        if let Some(clear_raw) = entries.get("clear") {
            let clear = parse_vec4_f32(clear_raw, line, "clear")?;
            return Ok(LoadOp::Clear(clear));
        }
    }
    match normalize_ident(raw).as_str() {
        "load" => Ok(LoadOp::Load),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_store_op(raw: &str, line: usize, field: &str) -> Result<StoreOp, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "store" => Ok(StoreOp::Store),
        "discard" => Ok(StoreOp::Discard),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_output_target(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<OutputTarget, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "screen" => Ok(OutputTarget::Screen),
        "file" => Ok(OutputTarget::File),
        "host" => Ok(OutputTarget::Host),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_present_target(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<PresentTarget, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "screen" => Ok(PresentTarget::Screen),
        "host" => Ok(PresentTarget::Host),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_sample_filter(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<SampleFilter, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "nearest" => Ok(SampleFilter::Nearest),
        "linear" => Ok(SampleFilter::Linear),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_sample_address(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<SampleAddress, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "clamp" => Ok(SampleAddress::Clamp),
        "repeat" => Ok(SampleAddress::Repeat),
        "mirror" => Ok(SampleAddress::Mirror),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {} value: {}", field, other),
        }),
    }
}

fn parse_vec4_f32(raw: &str, line: usize, field: &str) -> Result<[f32; 4], GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{} must be [r,g,b,a].", field),
        })?;
    let parts: Vec<&str> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 4 {
        return Err(GraphParseError {
            line,
            message: format!("{} must have 4 values.", field),
        });
    }
    let mut out = [0.0f32; 4];
    for (ix, raw_part) in parts.iter().enumerate() {
        out[ix] = raw_part.parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("{} has invalid number: {}", field, raw_part),
        })?;
    }
    Ok(out)
}

fn normalize_ident(raw: &str) -> String {
    strip_wrappers(raw)
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-")
}

pub(crate) fn required_attr_value(
    block: &str,
    key: &str,
    line: usize,
) -> Result<String, GraphParseError> {
    attr_value(block, key).ok_or_else(|| GraphParseError {
        line,
        message: format!("Missing required attribute: {}", key),
    })
}

pub(crate) fn required_attr_value_any(
    block: &str,
    keys: &[&str],
    line: usize,
) -> Result<String, GraphParseError> {
    for key in keys {
        if let Some(v) = attr_value(block, key) {
            return Ok(v);
        }
    }
    Err(GraphParseError {
        line,
        message: format!("Missing required attribute: {}", keys.join("|")),
    })
}

pub(crate) fn attr_value(block: &str, key: &str) -> Option<String> {
    let start = find_attr_start(block, key)?;
    let mut rest = block[start..].trim_start();
    if !rest.starts_with('=') {
        return None;
    }
    rest = rest[1..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    if let Some(stripped) = rest.strip_prefix('{') {
        let mut depth = 1usize;
        let mut out = String::new();
        for ch in stripped.chars() {
            if ch == '{' {
                depth += 1;
                out.push(ch);
                continue;
            }
            if ch == '}' {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(out);
                }
                out.push(ch);
                continue;
            }
            out.push(ch);
        }
        return None;
    }
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest = &rest[..end];
    Some(rest.to_string())
}

fn find_attr_start(block: &str, key: &str) -> Option<usize> {
    let bytes = block.as_bytes();
    let key_bytes = key.as_bytes();
    if key_bytes.is_empty() || bytes.len() < key_bytes.len() + 1 {
        return None;
    }
    let mut in_double_quote = false;
    let mut i = 0usize;
    while i + key_bytes.len() < bytes.len() {
        let b = bytes[i];
        if b == b'"' {
            in_double_quote = !in_double_quote;
            i += 1;
            continue;
        }
        if in_double_quote {
            i += 1;
            continue;
        }
        if &bytes[i..i + key_bytes.len()] == key_bytes {
            let prev_ok = i == 0
                || bytes[i - 1].is_ascii_whitespace()
                || bytes[i - 1] == b'<'
                || bytes[i - 1] == b'\n';
            let mut j = i + key_bytes.len();
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if prev_ok && j < bytes.len() && bytes[j] == b'=' {
                return Some(i + key_bytes.len());
            }
        }
        i += 1;
    }
    None
}

pub(crate) fn strip_wrappers(raw: &str) -> &str {
    let mut text = raw.trim();
    loop {
        if text.starts_with('{') && text.ends_with('}') && text.len() >= 2 {
            text = text[1..text.len() - 1].trim();
            continue;
        }
        if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
            text = text[1..text.len() - 1].trim();
            continue;
        }
        break;
    }
    text
}

#[cfg(test)]
mod tests {
    use super::{
        ColorSpace, GraphApplyScope, GraphAssetKind, GraphAssetSource, GraphParseError, InputType,
        PassCache, PassKind, PassRole, PassTransitionClips, PassTransitionEasing,
        PassTransitionFallback, PassTransitionMode, PrimitiveColliderShape, PrimitiveCollisionMode,
        PrimitiveGeometry, Quality, ResourceRef, SceneNode, TextureFormat, VegetationKind,
        VegetationLod, is_graph_script, parse_action_library_document, parse_graph_script,
    };
    use crate::scene::model::Scene3DNode;

    #[test]
    fn graph_parser_accepts_basic_example() {
        let script = r#"
<Graph fps={30} duration="2s" size={[256,256]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[256,256]} />
  <Pass id="invert_pulse" kernel="invert_mix.wgsl" effect="invert_mix"
        in={["src"]}
        out={["out"]}
        params={{
          t: "$time.norm",
          mix: "0.5 + 0.5*sin($time.sec*6.28318)"
        }} />
  <Present from="out" />
</Graph>
"#;
        assert!(is_graph_script(script));
        let graph = parse_graph_script(script).expect("graph should parse");
        assert_eq!(graph.textures.len(), 2);
        assert_eq!(graph.passes.len(), 1);
        assert_eq!(graph.present.from, "out");
        assert_eq!(graph.passes[0].kind, PassKind::Compute);
    }

    #[test]
    fn graph_parser_accepts_leading_xml_comment() {
        let script = r##"
<!-- Font note: unavailable font families fall back to renderer defaults. -->
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Scene id="commented_scene">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer id="empty_layer">
            <Text x="24" y="48" value="Comment OK" fontSize="24" color="#111111" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="commented_scene" />
</Graph>
"##;
        assert!(is_graph_script(script));
        let graph = parse_graph_script(script).expect("leading XML comment should parse");
        assert_eq!(graph.scenes[0].id, "commented_scene");
    }

    #[test]
    fn graph_parser_accepts_text_font_weight() {
        let script = r##"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Text x="24" y="48" value="Bold" fontSize="24" fontFamily="Impact" fontWeight="900" color="#111111" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("fontWeight text should parse");
        assert_eq!(graph.texts.len(), 1);
        assert_eq!(graph.texts[0].font_weight.as_deref(), Some("900"));
    }

    #[test]
    fn graph_parser_accepts_text_box_attrs() {
        let script = r##"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Text x="24" y="48" value="Pill" fontSize="24" box="pill" boxColor="#D9251D" boxPadding="54 28" boxRadius="999" color="#ffffff" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("Text box attrs should parse");
        assert_eq!(graph.texts.len(), 1);
        assert_eq!(graph.texts[0].box_style.as_deref(), Some("pill"));
        assert_eq!(graph.texts[0].box_color.as_deref(), Some("#D9251D"));
        assert_eq!(graph.texts[0].box_padding.as_deref(), Some("54 28"));
        assert_eq!(graph.texts[0].box_radius.as_deref(), Some("999"));
    }

    #[test]
    fn graph_parser_accepts_text_gap_alias() {
        let script = r##"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Text x="24" y="48" value="Tight" fontSize="24" textGap="-2" color="#ffffff" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("textGap should parse");
        assert_eq!(graph.texts.len(), 1);
        assert_eq!(graph.texts[0].tracking.as_deref(), Some("-2"));
    }

    #[test]
    fn graph_parser_accepts_text_blur_and_smoothing_attrs() {
        let script = r##"
<Graph fps={30} duration="4s" size={[256,256]}>
  <Background color="#ffffff" />
  <Text x="24" y="96" value="Soft" fontSize="48" renderScale="auto"
        antialias="subpixel"
        softEdge="0.34"
        blur={curve("0:0.4:linear, 1.6:2.8:ease_in_out, 4:0.8:ease_out")}
        color="#111111" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("Text blur/smoothing attrs should parse");
        assert_eq!(graph.texts.len(), 1);
        assert_eq!(graph.texts[0].render_scale, "auto");
        assert_eq!(graph.texts[0].antialias.as_deref(), Some("subpixel"));
        assert_eq!(graph.texts[0].soft_edge.as_deref(), Some("0.34"));
        assert_eq!(
            graph.texts[0].blur.as_deref(),
            Some(r#"curve("0:0.4:linear, 1.6:2.8:ease_in_out, 4:0.8:ease_out")"#)
        );
    }

    #[test]
    fn graph_parser_rejects_leading_plain_text() {
        let script = r##"
Font note: this is not a structured XML comment.
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Scene id="bad_prefix">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="256" height="256" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="bad_prefix" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("plain text before Graph should fail");
        assert!(
            err.message.contains("Only whitespace and XML comments"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_rejects_present_inside_scene() {
        let script = r#"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Scene id="scene0">
    <Present from="scene0" />
  </Scene>
  <Present from="scene0" />
</Graph>
"#;
        let err = parse_graph_script(script).expect_err("nested Present should fail");
        assert!(
            err.message.contains("direct child of <Graph>"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_rejects_nodes_after_root_present() {
        let script = r##"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Present from="scene" />
  <Text x="0" y="0" value="late" fontSize="12" color="#111111" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("Present must be final");
        assert!(
            err.message.contains("final node in <Graph>"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_accepts_process_block_with_root_present() {
        let script = r#"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Process id="final_grade">
    <Tex id="src" fmt="rgba8" size={[256,256]} />
    <Tex id="out" fmt="rgba8" size={[256,256]} />
    <Pass id="fx" kind="compute" effect="opacity"
          in={["src"]} out={["out"]}
          params={{ opacity: "1.0" }} />
  </Process>
  <Present from="final_grade" />
</Graph>
"#;
        let graph = parse_graph_script(script).expect("process alias should parse");
        assert_eq!(graph.present.from, "final_grade");
        assert_eq!(graph.passes.len(), 1);
        assert!(
            graph.outputs.iter().any(|output| {
                output.id == "final_grade" && output.from.as_deref() == Some("out")
            })
        );
    }

    #[test]
    fn graph_parser_rejects_lowercase_process_tag() {
        let script = r#"
<Graph fps={30} duration="1s" size={[256,256]}>
  <process id="final_grade">
    <Tex id="src" fmt="rgba8" size={[256,256]} />
    <Tex id="out" fmt="rgba8" size={[256,256]} />
    <Pass id="fx" kind="compute" effect="opacity"
          in={["src"]} out={["out"]}
          params={{ opacity: "1.0" }} />
  </process>
  <Present from="final_grade" />
</Graph>
"#;
        let err = parse_graph_script(script).expect_err("lowercase Process should fail clearly");
        assert!(
            err.message.contains("Use <Process> with an uppercase P"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_accepts_render_size() {
        let script = r##"
<Graph fps={30} duration="1s" size={[734,555]} renderSize={[3840,2160]}>
  <Background color="#ffffff" />

  <Scene id="scene0">
  </Scene>
  <Present from="scene0" />
</Graph>
"##;
        let graph = parse_graph_script(script).expect("graph should parse");
        assert_eq!(graph.size, (734, 555));
        assert_eq!(graph.render_size, Some((3840, 2160)));
    }

    #[test]
    fn graph_parser_rejects_scene_root_visual_nodes() {
        let script = r##"
<Graph fps={30} duration="1s" size={[80,60]}>
  <Scene id="strict_scene">
    <Rect x="0" y="0" width="80" height="60" color="#ffffff" />
  </Scene>
  <Present from="strict_scene" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("scene root visual nodes should fail");
        assert!(
            err.message
                .contains("<Scene> root only accepts <Defs> and <Timeline>"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_accepts_palette_inside_defs() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="pixel_scene">
    <Defs>
      <Palette id="pixel_palette">
        <Color key="." value="#00000000" />
        <Color key="K" value="#0B0D16" />
        <Color key="S" value="#F4BDAF" />
      </Palette>
    </Defs>
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <PixelGrid id="face_pixels" x="8" y="8" pixelSize="4" palette="pixel_palette">
              <![CDATA[
..KK..
.KSSK.
..KK..
              ]]>
            </PixelGrid>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="pixel_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Defs(defs) = &graph.scenes[0].children[0] else {
            panic!("expected defs child");
        };
        assert_eq!(defs.palettes.len(), 1);
        assert_eq!(defs.palettes[0].id, "pixel_palette");
        assert_eq!(defs.palettes[0].colors.len(), 3);
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[1] else {
            panic!("expected timeline child");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track child");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence child");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer child");
        };
        let SceneNode::PixelGrid(grid) = &layer.children[0] else {
            panic!("expected pixel grid child");
        };
        assert_eq!(grid.palette, "pixel_palette");
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_component_defs_and_use_ref() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="component_scene">
    <Defs>
      <Component id="green_dot">
        <Circle x="0" y="0" radius="5" color="#00ff00" />
      </Component>
    </Defs>
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Use ref="green_dot" x="24" y="24" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="component_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Defs(defs) = &graph.scenes[0].children[0] else {
            panic!("expected defs child");
        };
        assert_eq!(defs.components.len(), 1);
        assert_eq!(defs.components[0].id, "green_dot");

        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[1] else {
            panic!("expected timeline child");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track child");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence child");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer child");
        };
        let SceneNode::Use(use_node) = &layer.children[0] else {
            panic!("expected use child");
        };
        assert_eq!(use_node.ref_id, "green_dot");
        Ok(())
    }

    #[test]
    fn graph_parser_lowers_parametric_component_use() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="parametric_component_scene">
    <Defs>
      <Component id="dot">
        <Param name="radius" type="number" default="4" />
        <Param name="paint" type="color" default="#ff0000" />
        <Circle x="0" y="0" radius={param("radius")} color={param("paint")} />
      </Component>
    </Defs>
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Use id="large_dot" ref="dot" x="20" y="24"
                 params={{ radius: "9", paint: "#00ff00" }} />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="parametric_component_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Defs(defs) = &graph.scenes[0].children[0] else {
            panic!("expected defs");
        };
        assert_eq!(defs.components[0].params.len(), 2);
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[1] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Group(instance) = &layer.children[0] else {
            panic!("parameterized Use should lower to Group");
        };
        assert_eq!(instance.id.as_deref(), Some("large_dot"));
        assert_eq!(instance.x, "20");
        let SceneNode::Circle(circle) = &instance.children[0] else {
            panic!("expected substituted Circle");
        };
        assert_eq!(circle.radius, "9");
        assert_eq!(circle.color, "#00ff00");
        Ok(())
    }

    #[test]
    fn graph_parser_binds_target_for_lowered_puppet_component() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="puppet_component_scene">
    <Defs>
      <Component id="puppet_preset">
        <Param name="targetId" type="text" />
        <PuppetWarp target={param("targetId")} width="64" height="64">
          <PuppetPin id="control" x="32" y="32" targetX="36" targetY="28" />
        </PuppetWarp>
      </Component>
    </Defs>
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Group id="arm_art">
              <Rect x="20" y="20" width="24" height="12" color="#ffffff" />
            </Group>
            <Use id="arm_puppet" ref="puppet_preset"
                 params={{ targetId: "arm_art" }} />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="puppet_component_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[1] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        assert_eq!(layer.children.len(), 1);
        let SceneNode::Puppet(puppet) = &layer.children[0] else {
            panic!("identity Puppet Component should lower directly to Puppet");
        };
        assert_eq!(puppet.id.as_deref(), Some("arm_puppet"));
        assert_eq!(puppet.target.as_deref(), Some("arm_art"));
        assert!(matches!(
            puppet.children.first(),
            Some(SceneNode::Group(group)) if group.id.as_deref() == Some("arm_art")
        ));
        Ok(())
    }

    #[test]
    fn graph_parser_captures_preceding_layer_nodes_for_universal_puppet()
    -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="universal_puppet_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer id="art_layer">
            <Rect id="body" x="8" y="8" width="48" height="48" color="#ffffff" />
            <PuppetWarp id="layer_puppet" target="@layer" capture="before"
                        mesh="alpha" width="64" height="64">
              <PuppetPin id="control" x="32" y="32" targetX="36" targetY="28" />
            </PuppetWarp>
            <Text id="overlay" x="32" y="60" value="UI" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="universal_puppet_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        assert_eq!(layer.children.len(), 2);
        let SceneNode::Puppet(puppet) = &layer.children[0] else {
            panic!("captured artwork should lower to one Puppet surface");
        };
        assert_eq!(puppet.target.as_deref(), Some("@layer"));
        assert_eq!(puppet.capture.as_deref(), Some("before"));
        assert!(matches!(
            puppet.children.first(),
            Some(SceneNode::Rect(rect)) if rect.id.as_deref() == Some("body")
        ));
        assert!(matches!(
            layer.children.get(1),
            Some(SceneNode::Text(text)) if text.id.as_deref() == Some("overlay")
        ));
        Ok(())
    }

    #[test]
    fn graph_parser_composes_multiple_universal_puppets_in_source_order()
    -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="multi_limb_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer id="art_layer">
            <Rect id="body" x="8" y="8" width="48" height="48" color="#ffffff" />
            <PuppetWarp id="left_arm" target="@layer" capture="before"
                        mesh="alpha" width="64" height="64">
              <PuppetPin id="left_control" x="20" y="32" targetX="18" targetY="26" />
            </PuppetWarp>
            <PuppetWarp id="right_arm" target="@layer" capture="before"
                        mesh="alpha" width="64" height="64">
              <PuppetPin id="right_control" x="44" y="32" targetX="46" targetY="26" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="multi_limb_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };

        // Each later universal warp captures the previously lowered surface.
        assert_eq!(layer.children.len(), 1);
        let SceneNode::Puppet(right_arm) = &layer.children[0] else {
            panic!("expected the last universal warp at the layer root");
        };
        assert_eq!(right_arm.id.as_deref(), Some("right_arm"));
        let Some(SceneNode::Puppet(left_arm)) = right_arm.children.first() else {
            panic!("expected the earlier universal warp inside the later warp");
        };
        assert_eq!(left_arm.id.as_deref(), Some("left_arm"));
        assert!(matches!(
            left_arm.children.first(),
            Some(SceneNode::Rect(rect)) if rect.id.as_deref() == Some("body")
        ));
        Ok(())
    }

    #[test]
    fn graph_parser_defaults_universal_puppet_capture_to_before() {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="invalid_universal_puppet">
    <Timeline>
      <Track space="screen">
        <Sequence duration="1s">
          <Layer>
            <Rect x="8" y="8" width="48" height="48" color="#ffffff" />
            <PuppetWarp target="@layer">
              <PuppetPin x="32" y="32" targetX="36" targetY="28" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="invalid_universal_puppet" />
</Graph>
"##;
        parse_graph_script(script).expect("target=@layer should default capture to before");
    }

    #[test]
    fn graph_parser_rejects_invalid_universal_puppet_capture() {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="invalid_universal_puppet">
    <Timeline>
      <Track space="screen">
        <Sequence duration="1s">
          <Layer>
            <Rect x="8" y="8" width="48" height="48" color="#ffffff" />
            <PuppetWarp target="@layer" capture="after">
              <PuppetPin x="32" y="32" targetX="36" targetY="28" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="invalid_universal_puppet" />
</Graph>
"##;
        let error =
            parse_graph_script(script).expect_err("explicit capture modes other than before fail");
        assert!(
            error
                .message
                .contains("target=\"@layer\" requires capture=\"before\"")
        );
    }

    #[test]
    fn graph_parser_keeps_group_target_mode_unchanged() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="group_puppet_scene">
    <Timeline>
      <Track space="screen">
        <Sequence duration="1s">
          <Layer>
            <Group id="hair">
              <Rect x="8" y="8" width="48" height="24" color="#ffffff" />
            </Group>
            <PuppetWarp id="hair_puppet" target="hair" width="64" height="64">
              <PuppetPin x="32" y="16" targetX="36" targetY="12" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="group_puppet_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        assert_eq!(layer.children.len(), 1);
        let SceneNode::Puppet(puppet) = &layer.children[0] else {
            panic!("Group target should retain its existing lowering");
        };
        assert_eq!(puppet.target.as_deref(), Some("hair"));
        assert_eq!(puppet.capture, None);
        assert!(matches!(
            puppet.children.first(),
            Some(SceneNode::Group(group)) if group.id.as_deref() == Some("hair")
        ));
        Ok(())
    }

    #[test]
    fn graph_parser_lowers_seeded_scatter_repeat_deterministically() -> Result<(), GraphParseError>
    {
        let script = r##"
<Graph fps={30} duration="1s" size={[128,128]}>
  <Scene id="scatter_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Repeat id="stars" count="6" distribution="scatter"
                    bounds={[10,20,80,60]} seed="42"
                    scaleRange={[0.5,1.5]} rotationRange={[-10,10]}
                    opacityRange={[0.3,1]}>
              <Circle x="0" y="0" radius="3" color="#ffffff" />
            </Repeat>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scatter_scene" />
</Graph>
"##;
        let first = parse_graph_script(script)?;
        let second = parse_graph_script(script)?;
        assert_eq!(first.scenes, second.scenes);
        let SceneNode::Timeline(timeline) = &first.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Group(scatter) = &layer.children[0] else {
            panic!("scatter Repeat should lower to Group");
        };
        assert_eq!(scatter.children.len(), 6);
        let SceneNode::Group(first_item) = &scatter.children[0] else {
            panic!("expected generated scatter item");
        };
        let SceneNode::Group(second_item) = &scatter.children[1] else {
            panic!("expected generated scatter item");
        };
        assert_ne!(first_item.x, second_item.x);
        assert_ne!(first_item.scale, second_item.scale);
        Ok(())
    }

    #[test]
    fn graph_parser_retains_deterministic_3d_volume_repeat() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[128,128]}>
  <Assets>
    <PrimitiveAsset id="drop" shape="cylinder" radius="0.01" height="0.4" />
  </Assets>
  <Scene id="rain_scene">
    <Timeline>
      <Track space="3d" z="0">
        <Sequence duration="1s">
          <CompositeGroup id="rain" space="3d">
            <Camera3D position={[0,1,4]} target={[0,1,0]} />
            <Repeat id="rain_volume" mode="volume" count="12" seed="77"
                    boundsMin={[-2,0,-2]} boundsMax={[2,4,2]}
                    velocity={[-0.2,-8,0.1]} lifetime="0.6s"
                    phase="random" respawn="random" scaleRange={[0.8,1.2]}>
              <Model asset="drop" castShadow="false" receiveShadow="false" />
            </Repeat>
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="rain_scene" />
</Graph>
"##;
        let first = parse_graph_script(script)?;
        let second = parse_graph_script(script)?;
        assert_eq!(first.scenes, second.scenes);
        let SceneNode::Timeline(timeline) = &first.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Group(group) = &sequence.children[0] else {
            panic!("expected 3D CompositeGroup");
        };
        let repeat = group
            .composite
            .as_ref()
            .and_then(|composite| {
                composite.nodes_3d.iter().find_map(|node| match node {
                    Scene3DNode::VolumeRepeat(repeat) => Some(repeat),
                    _ => None,
                })
            })
            .expect("volume Repeat retained as typed 3D data");
        assert_eq!(repeat.count, 12);
        assert_eq!(repeat.seed, 77);
        assert_eq!(repeat.template.asset, "drop");
        Ok(())
    }

    #[test]
    fn graph_parser_lowers_declarative_grid_layout() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Scene id="layout_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Layout id="cards" mode="grid" x="20" y="30"
                    columns="2" itemWidth="80" itemHeight="50" gap="10">
              <Rect x="0" y="0" width="80" height="50" color="#ff0000" />
              <Rect x="0" y="0" width="80" height="50" color="#00ff00" />
              <Rect x="0" y="0" width="80" height="50" color="#0000ff" />
            </Layout>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="layout_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Group(layout) = &layer.children[0] else {
            panic!("Layout should lower to Group");
        };
        assert_eq!(layout.x, "20");
        assert_eq!(layout.y, "30");
        assert_eq!(layout.children.len(), 3);
        let positions = layout
            .children
            .iter()
            .map(|node| match node {
                SceneNode::Group(item) => (item.x.as_str(), item.y.as_str()),
                _ => panic!("expected layout item Group"),
            })
            .collect::<Vec<_>>();
        assert_eq!(positions, vec![("0", "0"), ("90", "0"), ("0", "60")]);
        Ok(())
    }

    #[test]
    fn graph_parser_lowers_advanced_component_bindings_and_slots() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[240,160]}>
  <Scene id="advanced_component_scene">
    <Defs>
      <Component id="card">
        <Param name="width" type="number" default="100" />
        <Param name="visible" type="boolean" default="true" />
        <Param name="tone" type="enum" values={["#22d3ee","#a3e635"]} default="#22d3ee" />
        <Derived name="halfWidth" value={param("width") * 0.5} />
        <Rect x={derived("halfWidth")} y="0" width={param("width")} height="60"
              color={param("tone")} opacity={param("visible")} />
        <Slot name="label">
          <Text x="8" y="32" value="DEFAULT" fontSize="14" color="#ffffff" />
        </Slot>
      </Component>
    </Defs>
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Use id="custom_card" ref="card" x="20" y="40"
                 params={{ width: "80", visible: "false", tone: "#a3e635" }}>
              <Fill slot="label">
                <Text x="8" y="32" value="CUSTOM" fontSize="14" color="#ffffff" />
              </Fill>
            </Use>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="advanced_component_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[1] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Group(instance) = &layer.children[0] else {
            panic!("expected lowered component Group");
        };
        let SceneNode::Rect(rect) = &instance.children[0] else {
            panic!("expected component Rect");
        };
        assert_eq!(rect.x, "80 * 0.5");
        assert_eq!(rect.width, "80");
        assert_eq!(rect.color, "#a3e635");
        assert_eq!(rect.opacity, "0");
        let SceneNode::Group(slot) = &instance.children[1] else {
            panic!("expected lowered Slot Group");
        };
        let SceneNode::Text(label) = &slot.children[0] else {
            panic!("expected custom slot Text");
        };
        assert_eq!(label.value, "CUSTOM");
        Ok(())
    }

    #[test]
    fn graph_parser_lowers_weighted_variants_and_property_variation_deterministically()
    -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[320,120]}>
  <Scene id="repeat_variants_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Repeat id="marks" count="24" x="20" y="60" xStep="12" seed="77">
              <Variants choose="weighted" seed="91">
                <Circle x="0" y="0" radius="5" color="#ffffff" weight="3" />
                <Rect x="-5" y="-5" width="10" height="10" color="#ffffff" weight="1" />
              </Variants>
              <Vary property="color" values={["#22d3ee","#a3e635","#f472b6"]} />
              <Vary property="scale" range={[0.6,1.4]} />
            </Repeat>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="repeat_variants_scene" />
</Graph>
"##;
        let first = parse_graph_script(script)?;
        let second = parse_graph_script(script)?;
        assert_eq!(first.scenes, second.scenes);
        let SceneNode::Timeline(timeline) = &first.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Group(repeat) = &layer.children[0] else {
            panic!("expected lowered Repeat Group");
        };
        assert_eq!(repeat.children.len(), 24);
        let changed_variant_seed =
            parse_graph_script(&script.replace("seed=\"91\"", "seed=\"92\""))?;
        let SceneNode::Timeline(changed_timeline) = &changed_variant_seed.scenes[0].children[0]
        else {
            panic!("expected changed timeline");
        };
        let SceneNode::Track(changed_track) = &changed_timeline.children[0] else {
            panic!("expected changed track");
        };
        let SceneNode::Sequence(changed_sequence) = &changed_track.children[0] else {
            panic!("expected changed sequence");
        };
        let SceneNode::Layer(changed_layer) = &changed_sequence.children[0] else {
            panic!("expected changed layer");
        };
        let SceneNode::Group(changed_repeat) = &changed_layer.children[0] else {
            panic!("expected changed Repeat Group");
        };
        for (original, changed) in repeat.children.iter().zip(&changed_repeat.children) {
            let (SceneNode::Group(original), SceneNode::Group(changed)) = (original, changed)
            else {
                panic!("expected Repeat item Groups");
            };
            assert_eq!(
                (original.x.as_str(), original.y.as_str()),
                (changed.x.as_str(), changed.y.as_str())
            );
        }
        let mut circles = 0;
        let mut rects = 0;
        for item in &repeat.children {
            let SceneNode::Group(item) = item else {
                panic!("expected Repeat item Group");
            };
            assert_ne!(item.scale, "1");
            match &item.children[0] {
                SceneNode::Circle(circle) => {
                    circles += 1;
                    assert_ne!(circle.color, "#ffffff");
                }
                SceneNode::Rect(rect) => {
                    rects += 1;
                    assert_ne!(rect.color, "#ffffff");
                }
                _ => panic!("expected weighted Circle or Rect"),
            }
        }
        assert!(circles > 0 && rects > 0);
        Ok(())
    }

    #[test]
    fn graph_parser_lowers_layout_padding_alignment_justify_and_span() -> Result<(), GraphParseError>
    {
        let script = r##"
<Graph fps={30} duration="1s" size={[360,220]}>
  <Scene id="advanced_layout_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence duration="1s">
          <Layer>
            <Layout id="cards" mode="grid" width="300" height="160" columns="3"
                    itemWidth="40" itemHeight="30" padding={[10,20]}
                    columnGap="10" rowGap="8" justify="spaceBetween" align="center">
              <Group layoutSpan="2">
                <Rect x="0" y="0" width="40" height="30" color="#22d3ee" />
              </Group>
              <Rect x="0" y="0" width="40" height="30" color="#a3e635" />
              <Rect x="0" y="0" width="40" height="30" color="#f472b6" />
            </Layout>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="advanced_layout_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Group(layout) = &layer.children[0] else {
            panic!("expected lowered Layout Group");
        };
        let positions = layout
            .children
            .iter()
            .map(|node| match node {
                SceneNode::Group(item) => (item.x.as_str(), item.y.as_str()),
                _ => panic!("expected Layout item Group"),
            })
            .collect::<Vec<_>>();
        assert_eq!(positions, vec![("20", "46"), ("240", "46"), ("20", "84")]);
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_removed_symbol_defs() {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="component_scene">
    <Defs>
      <Symbol id="green_dot">
        <Circle x="0" y="0" radius="5" color="#00ff00" />
      </Symbol>
    </Defs>
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="component_scene" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("old Symbol tag must be rejected");
        assert!(
            err.message.contains("<Component>") && err.message.contains("<Symbol"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_rejects_removed_use_symbol_attr() {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="component_scene">
    <Defs>
      <Component id="green_dot">
        <Circle x="0" y="0" radius="5" color="#00ff00" />
      </Component>
    </Defs>
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Use symbol="green_dot" x="24" y="24" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="component_scene" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("old Use symbol attr must be rejected");
        assert!(
            err.message.contains("<Use> requires ref="),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_rejects_palette_outside_defs() {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,64]}>
  <Scene id="pixel_scene">
    <Palette id="pixel_palette">
      <Color key="." value="#00000000" />
    </Palette>
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="pixel_scene" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("root palette should fail");
        assert!(
            err.message
                .contains("<Scene> root only accepts <Defs> and <Timeline>"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_accepts_scene_model_profiles() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[320,240]}>
  <Background color="#ffffff" />

  <ModelProfile id="3d_humanoid_glb_v1" kind="3d" model="hero.glb" />
  <ModelProfile id="2d_humanoid_vector_v1" kind="2d" preset="humanoid_front_v1">
    <Retarget preset="humanoid_v1">
      <Map from="head" to="head" />
    </Retarget>
    <BoneAxisMap>
      <Axis bone="head" turn="x" bend="y" />
      <Bone id="neck" restForward="+z" restSide="+x" />
    </BoneAxisMap>
  </ModelProfile>

  <Scene id="profile_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Character id="hero" rig="face_skeleton" modelProfile="2d_humanoid_vector_v1" x="160" y="120">
              <Path d="M 0 0 L 10 0" stroke="#000000" fill="none" />
            </Character>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Present from="profile_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.model_profiles.len(), 2);
        assert_eq!(graph.model_profiles[0].kind, "3d");
        assert_eq!(graph.model_profiles[1].kind, "2d");
        assert_eq!(graph.model_profiles[1].preset, "humanoid_front_v1");
        assert_eq!(
            graph.model_profiles[1]
                .retarget
                .as_ref()
                .and_then(|retarget| retarget.maps.first())
                .map(|map| map.to.as_str()),
            Some("head")
        );
        assert_eq!(
            graph.model_profiles[1]
                .bone_axis_map
                .as_ref()
                .map(|axis_map| axis_map.axes.len()),
            Some(2)
        );
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Character(character) = &layer.children[0] else {
            panic!("expected character");
        };
        assert_eq!(
            character.model_profile.as_deref(),
            Some("2d_humanoid_vector_v1")
        );
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_character_image_source() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[64,48]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hero" src="data:image/png;base64,AAAA" x="10" y="12" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        let SceneNode::Character(character) = &layer.children[0] else {
            panic!("expected character");
        };

        assert_eq!(character.src.as_deref(), Some("data:image/png;base64,AAAA"));
        assert_eq!(character.x, "10");
        assert_eq!(character.y, "12");
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_action_ik_target() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[120,120]}>
  <Skeleton id="arm">
    <Bone id="upper" x="0" y="0" />
    <Bone id="lower" parent="upper" x="40" y="0" />
    <Bone id="hand" parent="lower" x="40" y="0" />
  </Skeleton>

  <Action id="reach" skeleton="arm" duration="1s">
    <IK root="upper" mid="lower" end="hand" targetX="40" targetY="40" bend="1" />
  </Action>

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hero" rig="arm" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <ApplyAction target="hero" action="reach" />
  <Present from="scene0" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;

        assert_eq!(graph.actions[0].iks.len(), 1);
        assert_eq!(graph.actions[0].iks[0].root, "upper");
        assert_eq!(graph.actions[0].iks[0].target_x, "40");
        assert_eq!(graph.actions[0].iks[0].weight, "1");
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_action_chain_ik_target() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="1s" size={[120,120]}>
  <Skeleton id="finger">
    <Bone id="finger_1" x="0" y="0" />
    <Bone id="finger_2" parent="finger_1" x="0" y="-40" />
    <Bone id="finger_3" parent="finger_2" x="0" y="-32" />
    <Bone id="finger_tip" parent="finger_3" x="0" y="-24" />
  </Skeleton>

  <Action id="curl" skeleton="finger" duration="1s">
    <IK chain="finger_1,finger_2,finger_3,finger_tip"
        targetX="24" targetY="-64" iterations="10" weight="1" />
  </Action>

  <Scene id="scene0">
    <Timeline>
      <Track id="scene_content" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Character id="hand" rig="finger" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <ApplyAction target="hand" action="curl" />
  <Present from="scene0" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;

        assert_eq!(graph.actions[0].iks[0].chain.len(), 4);
        assert_eq!(graph.actions[0].iks[0].root, "finger_1");
        assert_eq!(graph.actions[0].iks[0].end, "finger_tip");
        assert_eq!(graph.actions[0].iks[0].iterations, "10");
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_missing_resource_ref() {
        let script = r#"
<Graph fps={30} duration="2s" size={[256,256]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Pass id="invert" kernel="invert_mix.wgsl" effect="invert_mix" in={["src"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let err = parse_graph_script(script).expect_err("missing tex should fail");
        assert!(
            err.message.contains("output resource not found"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_parses_new_nodes_and_enums() -> Result<(), GraphParseError> {
        let script = r#"
<Graph id="v2" version="2.0" fps={30} duration="2s" size={[1920,1080]}>
  <Input id="clip0" type="video" from="input:clip0" fmt="rgba8unorm-srgb" colorSpace="srgb" />
  <Buffer id="state" elemType="vec4f" usage={["storage","copy-dst"]} />
  <Tex id="work" fmt="rgba16f" usage={["sampled","storage"]} />
  <Output id="screen" to="screen" fmt="bgra8unorm-srgb" colorSpace="srgb" />
  <Pass id="prep" kind="compute" kernel="normalize_input.wgsl" effect="normalize_input"
        in={[{ tex:"clip0", sample:{ filter:"linear", address:"clamp" } }]}
        out={["work"]}
        cache="frame"
        iterate={{ preview: 1, final: 2 }} />
  <Present from="screen" to="screen" vsync={true} />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.id.as_deref(), Some("v2"));
        assert_eq!(graph.version.as_deref(), Some("2.0"));
        assert_eq!(graph.inputs[0].r#type, InputType::Video);
        assert_eq!(graph.inputs[0].fmt, Some(TextureFormat::Rgba8UnormSrgb));
        assert_eq!(graph.inputs[0].color_space, Some(ColorSpace::Srgb));
        assert_eq!(graph.passes[0].cache, Some(PassCache::Frame));
        assert_eq!(
            graph.passes[0].iterate,
            Some(Quality::Split {
                preview: 1,
                r#final: 2
            })
        );
        match &graph.passes[0].inputs[0] {
            ResourceRef::Tex { tex, .. } => assert_eq!(tex, "clip0"),
            other => panic!("unexpected input ref: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_background_text_without_passes() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="3s" size={[1920,1080]}>
  <Background color="#000000" />
  <Text value="hello world"
        x="center"
        y="center"
        fontSize="96"
        renderScale="4x"
        color="#ffffff"
        opacity="min($time.sec / 1.0, 1.0)" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.backgrounds.len(), 1);
        assert_eq!(graph.texts.len(), 1);
        assert_eq!(graph.images.len(), 0);
        assert_eq!(graph.svgs.len(), 0);
        assert_eq!(graph.texts[0].value, "hello world");
        assert_eq!(graph.texts[0].font_size, "96");
        assert_eq!(graph.texts[0].render_scale, "4x");
        assert_eq!(graph.present.from, "scene");
        assert_eq!(graph.resource_size("scene"), Some((1920, 1080)));
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_scene_image_without_passes() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="3s" size={[1920,1080]}>
  <Image src="/tmp/anica-test-image.png"
         x="center"
         y="120"
         scale="0.5 + 0.5*$time.norm"
         opacity="0.8" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.images.len(), 1);
        assert_eq!(graph.images[0].src, "/tmp/anica-test-image.png");
        assert_eq!(graph.images[0].x, "center");
        assert_eq!(graph.images[0].y, "120");
        assert_eq!(graph.images[0].scale, "0.5 + 0.5*$time.norm");
        assert_eq!(graph.present.from, "scene");
        assert_eq!(graph.resource_size("scene"), Some((1920, 1080)));
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_scene_svg_without_passes() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="3s" size={[1920,1080]}>
  <Svg src="/tmp/anica-test-logo.svg"
       x="center"
       y="25%"
       scale="0.5 + 0.5*$time.norm"
       opacity="0.8" />
  <Present from="scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.svgs.len(), 1);
        assert_eq!(graph.svgs[0].src, "/tmp/anica-test-logo.svg");
        assert_eq!(graph.svgs[0].x, "center");
        assert_eq!(graph.svgs[0].y, "25%");
        assert_eq!(graph.svgs[0].scale, "0.5 + 0.5*$time.norm");
        assert_eq!(graph.svgs[0].opacity, "0.8");
        assert_eq!(graph.present.from, "scene");
        assert_eq!(graph.resource_size("scene"), Some((1920, 1080)));
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_scene_timeline_track_sequence_chain() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="3s" size={[320,180]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="bars" z="10">
        <Sequence id="first" from="0.2s" duration="0.5s" out="hold">
          <Rect x="10" y={curve("0:100:linear, 0.5:40:linear")} width="20" height="60" color="#ffffff" />
        </Sequence>
        <Chain id="stagger" from="1s" gap="-0.1s">
          <Sequence id="second" duration="0.5s">
            <Rect x="40" y="40" width="20" height="60" color="#ffffff" />
          </Sequence>
          <Sequence id="third" duration="0.5s">
            <Rect x="70" y="40" width="20" height="60" color="#ffffff" />
          </Sequence>
        </Chain>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline child");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track child");
        };
        assert_eq!(track.id.as_deref(), Some("bars"));
        assert_eq!(track.z, 10);
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence child");
        };
        assert_eq!(sequence.id.as_deref(), Some("first"));
        assert_eq!(sequence.from_ms, 200);
        assert_eq!(sequence.duration_ms, 500);
        assert_eq!(sequence.out, "hold");
        let SceneNode::Chain(chain) = &track.children[1] else {
            panic!("expected chain child");
        };
        assert_eq!(chain.id.as_deref(), Some("stagger"));
        assert_eq!(chain.from_ms, 1000);
        assert_eq!(chain.gap_ms, -100);
        assert_eq!(chain.children.len(), 2);
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_full_text_animator_ast() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="6s" size={[1280,720]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="text" space="screen" z="10">
        <Sequence duration="6s" out="hold">
          <Layer>
            <Text id="hero_caption"
                  value="AI edits your video"
                  x="center"
                  y="center"
                  maxWidth="980"
                  align="center"
                  fontSize="92"
                  lineHeight="1.05"
                  tracking="-0.02em"
                  color="#EAFEFF"
                  stroke="#071018"
                  strokeWidth="6"
                  strokeJoin="round"
                  strokePosition="outside">
              <TextLayout wrap="balance" overflow="fit" safeArea="96,80,96,80" maxLines="3" />
              <TextAnimator id="word_reveal" selector="word" from="0s" duration="0.55s" stagger="0.08s" order="forward">
                <Transform y={curve("0:42:ease_out, 0.45:0:ease_out")}
                           scale={curve("0:0.88:ease_out, 0.45:1:ease_out")}
                           rotation={curve("0:-3:ease_out, 0.45:0:ease_out")} />
                <Style opacity={curve("0:0:linear, 0.22:1:ease_out")}
                       blur={curve("0:14:ease_out, 0.50:0:ease_out")} />
              </TextAnimator>
              <TextAnimator id="active_word_karaoke" selector="word" mode="karaoke" activeWord={floor($time.sec * 2.2)} preRoll="0.10s" postRoll="0.18s">
                <Style color="#FFB000" stroke="#071018" strokeWidth="8" shadowColor="#000000" shadowX="0" shadowY="8" shadowBlur="20" />
                <Effects>
                  <Glow radius="22" intensity="1.4" color="#FFB000" />
                </Effects>
              </TextAnimator>
              <TextAnimator id="char_micro_motion" selector="char" from="0.3s" duration="6s" stagger="0.012s" randomSeed="42">
                <Transform y={noise("freq:1.2, amp:2.5")} rotation={noise("freq:0.8, amp:1.4")} />
              </TextAnimator>
              <TextAnimator id="exit_by_line" selector="line" from="5.2s" duration="0.7s" stagger="0.10s" order="reverse">
                <Transform y={curve("0:0:ease_in, 0.7:-48:ease_in")} scale={curve("0:1:ease_in, 0.7:0.96:ease_in")} />
                <Style opacity={curve("0:1:linear, 0.45:0:ease_in")} blur={curve("0:0:ease_in, 0.7:18:ease_in")} />
              </TextAnimator>
            </Text>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline child");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track child");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence child");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer child");
        };
        let SceneNode::Text(text) = &layer.children[0] else {
            panic!("expected text child");
        };

        assert_eq!(text.id.as_deref(), Some("hero_caption"));
        assert_eq!(text.max_width.as_deref(), Some("980"));
        assert_eq!(text.align.as_deref(), Some("center"));
        assert_eq!(text.stroke.as_deref(), Some("#071018"));
        assert_eq!(text.stroke_width.as_deref(), Some("6"));
        assert_eq!(text.layout.as_ref().expect("layout").wrap, "balance");
        assert_eq!(
            text.layout.as_ref().expect("layout").safe_area.as_deref(),
            Some("96,80,96,80")
        );
        assert_eq!(text.animators.len(), 4);
        assert_eq!(text.animators[0].id.as_deref(), Some("word_reveal"));
        assert_eq!(text.animators[0].duration_ms, Some(550));
        assert_eq!(text.animators[0].stagger_ms, 80);
        assert_eq!(text.animators[1].id.as_deref(), Some("active_word_karaoke"));
        assert!(text.animators[1].is_karaoke());
        assert_eq!(
            text.animators[1].active_word.as_deref(),
            Some("floor($time.sec * 2.2)")
        );
        assert_eq!(text.animators[1].effects.len(), 1);
        let crate::scene::text::TextEffectNode::Glow(glow) = &text.animators[1].effects[0];
        assert_eq!(glow.radius, "22");
        assert_eq!(text.animators[2].random_seed, Some(42));
        assert_eq!(text.animators[3].order, "reverse");

        let prepared = crate::scene::text::prepare_text_layout(text).expect("prepare text layout");
        assert_eq!(prepared.selections.words.len(), 4);
        assert_eq!(prepared.selections.lines.len(), 1);
        assert_eq!(prepared.animator_targets.len(), 4);
        assert_eq!(prepared.animator_targets[0].targets.len(), 4);
        assert_eq!(prepared.animator_targets[0].targets[0].start_ms, 0);
        assert_eq!(prepared.animator_targets[0].targets[1].start_ms, 80);
        assert_eq!(prepared.animator_targets[3].targets.len(), 1);
        assert_eq!(prepared.animator_targets[3].targets[0].start_ms, 5200);
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_top_level_scene_camera() {
        let script = r##"
<Graph fps={30} duration="4s" size={[1280,720]}>
  <Background color="#000000" />
  <Camera id="main_camera"
          target="anime"
          x={curve("0:-0.35:ease_in_out, 2:0.35:ease_in_out, 4:0:ease_in_out")}
          y="0"
          zoom={curve("0:1.0:linear, 2:1.18:ease_in_out, 4:1.0:ease_in_out")}
          fov="35" />
  <Present from="scene" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("top-level scene camera must be rejected");
        assert!(
            err.message.contains("Track role=\"camera\""),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_rejects_scene_camera_mode_attr() {
        let err = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="scene0">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Camera mode="2d" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect_err("Scene Camera mode attr must be rejected");
        assert!(
            err.message.contains("Scene Camera"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_accepts_active_camera_track_and_track_space() -> Result<(), GraphParseError> {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="camera" role="camera">
        <Sequence duration="1s">
          <Camera target="hero" zoom="1.2" />
        </Sequence>
      </Track>
      <Track id="world" space="world">
        <Sequence duration="1s">
          <Layer>
            <Circle id="hero" x="50" y="50" radius="10" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
      <Track id="hud" space="screen">
        <Sequence duration="1s">
          <Layer>
            <Text value="HUD" x="4" y="20" fontSize="12" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(camera_track) = &timeline.children[0] else {
            panic!("expected camera track");
        };
        assert_eq!(camera_track.role.as_deref(), Some("camera"));
        let SceneNode::Track(hud_track) = &timeline.children[2] else {
            panic!("expected hud track");
        };
        assert_eq!(hud_track.space, "screen");
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_scene_track_and_layer_z_depth() -> Result<(), GraphParseError> {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="world" space="world" zDepth="2.5">
        <Sequence duration="1s">
          <Layer zDepth={curve("0:0:linear, 1:1:linear")}>
            <Circle x="50" y="50" radius="10" color="#fff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected track");
        };
        assert_eq!(track.z_depth, "2.5");
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected layer");
        };
        assert_eq!(
            layer.z_depth.as_deref(),
            Some("curve(\"0:0:linear, 1:1:linear\")")
        );
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_camera_container_children() {
        let err = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="scene0">
    <Timeline>
      <Track role="camera">
        <Sequence duration="1s">
          <Camera>
            <Circle x="50" y="50" radius="10" color="#fff" />
          </Camera>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect_err("camera container must be rejected");
        assert!(
            err.message.contains("self-closing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_rejects_camera_track_space() {
        let err = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[100,100]}>
  <Scene id="scene0">
    <Timeline>
      <Track role="camera" space="screen">
        <Sequence duration="1s">
          <Camera zoom="1" />
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##,
        )
        .expect_err("camera track must not set space");
        assert!(
            err.message.contains("must not set space"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_parser_accepts_decimal_duration_two_dp() -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} duration="2.35s" size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="copy" kernel="invert_mix.wgsl" effect="invert_mix" in={["src"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.duration_ms, 2350);
        assert!(graph.duration_explicit);
        Ok(())
    }

    #[test]
    fn graph_parser_defaults_apply_clip_and_duration_when_omitted() -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="copy" kernel="invert_mix.wgsl" effect="invert_mix" in={["src"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.apply, GraphApplyScope::Clip);
        assert_eq!(graph.duration_ms, 2000);
        assert!(!graph.duration_explicit);
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_graph_without_scope() -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="copy" kernel="invert_mix.wgsl" effect="invert_mix" in={["src"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let _graph = parse_graph_script(script)?;
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_removed_scope_attr() {
        let script = r##"
<Graph scope="scene" fps={30} size={[1920,1080]}>
  <Background color="#000000" />

  <Scene id="scene0">
  </Scene>
  <Present from="scene0" />
</Graph>
"##;
        let err = parse_graph_script(script).expect_err("scope should be removed");
        assert!(
            err.message.contains("Graph scope has been removed"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_rejects_missing_pass_effect() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="copy" kernel="invert_mix.wgsl" in={["src"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let err = parse_graph_script(script).expect_err("effect should be required");
        assert!(
            err.message.contains("Missing required attribute: effect"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn graph_parser_accepts_missing_pass_kernel_when_effect_present() -> Result<(), GraphParseError>
    {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba8" from="input:clip0" />
  <Tex id="out" fmt="rgba8" size={[1920,1080]} />
  <Pass id="copy" effect="exposure_contrast" in={["src"]} out={["out"]} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.passes[0].kernel, None);
        assert_eq!(graph.passes[0].effect, "exposure_contrast");
        Ok(())
    }

    #[test]
    fn graph_parser_resolves_targeted_puppet_warp_and_bound_pins() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Scene id="puppet_scene">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="2s">
          <Layer>
            <Group id="character">
              <Group id="left_hand" x="180" y="220">
                <Circle x="0" y="0" radius="24" color="#f2c9b8" />
              </Group>
            </Group>
            <PuppetWarp id="character_warp" target="character" width="640" height="360">
              <PuppetPin id="left_hand_pin" bindTo="left_hand" targetX="210" targetY="200" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="puppet_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected Timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected Track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected Sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected Layer");
        };
        assert_eq!(layer.children.len(), 1);
        let SceneNode::Puppet(puppet) = &layer.children[0] else {
            panic!("target Group should be normalized into Puppet");
        };
        assert!(
            matches!(&puppet.children[0], SceneNode::Group(group) if group.id.as_deref() == Some("character"))
        );
        let pin = puppet
            .children
            .iter()
            .find_map(|node| match node {
                SceneNode::Pin(pin) => Some(pin),
                _ => None,
            })
            .expect("bound pin");
        assert_eq!(pin.bind_to.as_deref(), Some("left_hand"));
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_bone_puppet_solver_and_pin_roles() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Scene id="bone_puppet_scene">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="2s">
          <Layer>
            <Group id="arm">
              <Path d="M 100 100 L 300 220" fill="none" stroke="#ffffff" strokeWidth="40" />
            </Group>
            <PuppetWarp id="arm_rig" target="arm" solver="bones"
                        bend="auto" stretch="0" jointSoftness="24"
                        preserveVolume="true" width="640" height="360">
              <PuppetPin id="shoulder" role="anchor" x="100" y="100"
                         targetX="100" targetY="100" fixed="true" />
              <PuppetPin id="elbow" role="joint" x="200" y="160"
                         targetX="200" targetY="160" />
              <PuppetPin id="wrist" role="control" x="300" y="220"
                         targetX="280" targetY="120" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="bone_puppet_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected Timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected Track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected Sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected Layer");
        };
        let SceneNode::Puppet(puppet) = &layer.children[0] else {
            panic!("expected targeted PuppetWarp");
        };
        assert_eq!(puppet.solver, "bones");
        assert_eq!(puppet.bend, "auto");
        assert_eq!(puppet.joint_softness, "24");
        let roles = puppet
            .children
            .iter()
            .filter_map(|node| match node {
                SceneNode::Pin(pin) => pin.role.as_deref(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(roles, vec!["anchor", "joint", "control"]);
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_chain_puppet_controls() -> Result<(), GraphParseError> {
        let script = r##"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Scene id="chain_puppet_scene">
    <Timeline>
      <Track>
        <Sequence from="0s" duration="2s">
          <Layer>
            <Group id="tail">
              <Path d="M 100 100 C 180 120 240 160 300 220"
                    fill="none" stroke="#ffffff" strokeWidth="40" />
            </Group>
            <PuppetWarp id="tail_rig" target="tail" solver="chain"
                        preserveLength="true" stiffness="0.72" damping="0.84"
                        drag="0.18" overlap="0.12" width="640" height="360">
              <PuppetPin id="tail_root" role="anchor" x="100" y="100"
                         targetX="100" targetY="100" fixed="true" />
              <PuppetPin id="tail_mid" role="chain" parent="tail_root"
                         x="200" y="150" targetX="200" targetY="150" />
              <PuppetPin id="tail_tip" role="control" parent="tail_mid"
                         x="300" y="220" targetX="280" targetY="120" />
            </PuppetWarp>
            <SpringChain target="tail_rig" segments="2" pin="both"
                         stiffness="0.8" damping="0.2" gravity={[0,16]} />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="chain_puppet_scene" />
</Graph>
"##;
        let graph = parse_graph_script(script)?;
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("expected Timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("expected Track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("expected Sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("expected Layer");
        };
        let SceneNode::Puppet(puppet) = &layer.children[0] else {
            panic!("expected targeted PuppetWarp");
        };
        assert_eq!(puppet.solver, "chain");
        assert_eq!(puppet.preserve_length, "true");
        assert_eq!(puppet.stiffness, "0.72");
        assert_eq!(puppet.damping, "0.84");
        assert_eq!(puppet.drag, "0.18");
        assert_eq!(puppet.overlap, "0.12");
        let parents = puppet
            .children
            .iter()
            .filter_map(|node| match node {
                SceneNode::Pin(pin) => Some(pin.parent.as_deref()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(parents, vec![None, Some("tail_root"), Some("tail_mid")]);
        Ok(())
    }

    #[test]
    fn graph_parser_params_support_single_line_multi_key_values() -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_hsla_overlay" effect="hsla_overlay" in={["under"]} out={["out"]}
        params={{ hue: "210.0", saturation: "0.70", lightness: "0.41", alpha: "0.45" }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        let params = &graph.passes[0].params;
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].key, "hue");
        assert_eq!(params[0].value, "\"210.0\"");
        assert_eq!(params[3].key, "alpha");
        assert_eq!(params[3].value, "\"0.45\"");
        Ok(())
    }

    #[test]
    fn graph_parser_params_preserve_curve_with_commas() -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx_hsla_overlay" effect="hsla_overlay" in={["under"]} out={["out"]}
        params={{ hue: "210.0", saturation: "0.70", lightness: "0.41", alpha: curve("0.00:0.0:linear, 2.00:0.45:ease_in_out") }} />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        let params = &graph.passes[0].params;
        assert_eq!(params.len(), 4);
        assert_eq!(params[3].key, "alpha");
        assert!(
            params[3]
                .value
                .contains("curve(\"0.00:0.0:linear, 2.00:0.45:ease_in_out\")")
        );
        Ok(())
    }

    #[test]
    fn graph_parser_parses_pass_transition_fields() -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Input id="under" type="video" from="input:under" />
  <Input id="prev" type="video" from="input:prev" />
  <Input id="next" type="video" from="input:next" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="dissolve" kind="render" role="transition"
        kernel="transition_core.wgsl"
        effect="dissolve"
        in={["prev","next"]} out={["out"]}
        transition="auto"
        transitionFallback="under"
        transitionEasing="ease-in-out"
        transitionClips="overlap" />
  <Present from="out" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        let pass = &graph.passes[0];
        assert_eq!(pass.role, Some(PassRole::Transition));
        assert_eq!(pass.effect, "dissolve");
        assert_eq!(pass.transition, Some(PassTransitionMode::Auto));
        assert_eq!(
            pass.transition_fallback,
            Some(PassTransitionFallback::Under)
        );
        assert_eq!(
            pass.transition_easing,
            Some(PassTransitionEasing::EaseInOut)
        );
        assert_eq!(pass.transition_clips, Some(PassTransitionClips::Overlap));
        Ok(())
    }

    #[test]
    fn graph_parser_accepts_scene_humanoid_actions_ik_and_bone_targets()
    -> Result<(), GraphParseError> {
        let script = r#"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Assets>
    <ModelAsset id="girl_asset" src="girl.glb" />
  </Assets>
  <ModelProfile id="girl_profile" kind="3d" model="girl_asset" preset="humanoid_v1">
    <Retarget preset="humanoid_v1">
      <Map from="Right arm_68" to="upper_arm_r" />
      <Map from="Right elbow_67" to="forearm_r" />
      <Map from="Right wrist_64" to="hand_r" />
    </Retarget>
  </ModelProfile>
  <Action id="reach" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s">
      <Bone id="upper_arm_r" rotationZ="0" />
    </Pose>
    <Pose t="1s">
      <Bone id="upper_arm_r" rotationX="12" rotationZ="-35" />
    </Pose>
    <IK root="upper_arm_r" mid="forearm_r" end="hand_r"
        targetX="0.8" targetY="1.1" targetZ="0.2" plane="xy" weight="1" />
  </Action>
  <ApplyAction target="girl" action="reach" at="0.2s" loop="true"
               weight="0.8" speed="1.2" blendIn="0.1s" blendOut="0.2s"
               mode="additive" mask="right_arm" />
  <Scene id="main_scene">
    <Timeline>
      <Track>
        <Sequence duration="2s">
          <Layer>
            <CompositeGroup id="island" space="3d" depth="true">
              <Camera3D position={[0,1,6]} target={[0,1,0]}
                        hiddenBones={["girl:head"]} />
              <Model id="girl" asset="girl_asset" profile="girl_profile">
                <Play clip="Idle" loop="true" speed="1" blendIn="0.2s" mask="upper_body" />
                <Play clip="Walk" loop="true" speed="1.1" weight="0.35" mask="lower_body" />
              </Model>
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <AnimationTarget node="girl" property="bones.forearm_r.rotationZ">
    <Key time="0s" value="0" />
    <Key time="1s" value="-45" ease="ease_in_out" />
  </AnimationTarget>
  <Present from="main_scene" />
</Graph>
"#;
        let graph = parse_graph_script(script)?;
        assert_eq!(graph.model_profiles[0].kind, "3d");
        assert_eq!(graph.actions[0].iks[0].target_z, "0.2");
        assert_eq!(
            graph.actions[0].poses[1].bones[0].rotation_x.as_deref(),
            Some("12")
        );
        assert_eq!(graph.apply_actions[0].mask, vec!["right_arm"]);
        assert_eq!(graph.apply_actions[0].blend_in, "0.1");
        assert_eq!(
            graph.animation_targets[0].property,
            "bones.forearm_r.rotationZ"
        );
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("3D group");
        };
        let model = group
            .composite
            .as_ref()
            .and_then(|composite| {
                composite.nodes_3d.iter().find_map(|node| match node {
                    Scene3DNode::Model(model) => Some(model),
                    _ => None,
                })
            })
            .expect("3D model");
        assert_eq!(model.profile.as_deref(), Some("girl_profile"));
        assert_eq!(
            model.play.as_ref().and_then(|play| play.clip.as_deref()),
            Some("Idle")
        );
        assert_eq!(model.plays.len(), 1);
        assert_eq!(model.plays[0].clip.as_deref(), Some("Walk"));
        let camera = group
            .composite
            .as_ref()
            .and_then(|composite| {
                composite.nodes_3d.iter().find_map(|node| match node {
                    Scene3DNode::Camera(camera) => Some(camera),
                    _ => None,
                })
            })
            .expect("3D camera");
        assert_eq!(camera.hidden_bones.len(), 1);
        assert_eq!(camera.hidden_bones[0].model, "girl");
        assert_eq!(camera.hidden_bones[0].bone, "head");
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_invalid_camera_hidden_bone_selectors() {
        let script = r#"
<Graph fps={30} duration="1s" size={[640,360]}>
  <Assets>
    <ModelAsset id="actor_asset" src="actor.glb" />
  </Assets>
  <Scene id="TerrainScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d">
              <Model id="actor" asset="actor_asset" />
              <Camera3D position={[0,1,4]} target={[0,1,0]}
                        hiddenBones={["actor:not_a_canonical_bone"]} />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="TerrainScene" />
</Graph>
"#;
        let error = parse_graph_script(script).expect_err("invalid canonical bone must fail");
        assert!(
            error.message.contains("canonical humanoid bone"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn graph_parser_accepts_external_actions_anchors_constraints_and_camera_switches()
    -> Result<(), GraphParseError> {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="6s" size={[1920,1080]}>
  <Assets>
    <ModelAsset id="model_a" src="character-a.glb" />
    <ModelAsset id="model_b" src="character-b.glb" />
    <AnimationAsset id="sneak_walk_source" src="motions/sneak-walk.glb" />
  </Assets>
  <Action id="sneak_walk" source="sneak_walk_source"
          sourceProfile="fbx_humanoid" clip="Sneak Walk"
          skeleton="humanoid_v1" duration="3.2s">
    <Marker id="takeoff" time="0.4s" role="takeoff" />
    <Marker id="contact" time="1.6s" role="contact" />
    <Marker id="landing" time="2.8s" role="landing" />
  </Action>
  <Scene id="FightScene">
    <Timeline>
      <Track>
        <Sequence duration="6s">
          <Layer>
            <CompositeGroup id="stage" space="3d" depth="true">
              <Model id="character_a" asset="model_a"
                     profile="motionloom_humanoid_v1" position="@character_a_start" />
              <Model id="character_b" asset="model_b"
                     profile="motionloom_humanoid_v1" position={[0,0,0]} />
              <Anchor id="character_a_start" relativeTo="character_b"
                      offset={[0,0,-4.2]} space="local" />
              <Anchor id="character_a_contact" relativeTo="character_b"
                      offset={[0,0,-0.72]} space="local" />
              <Camera3D id="camera_feet" position={[0,0.5,5]} target={[0,0.5,0]} />
              <Camera3D id="camera_medium" position={[2,1.2,5]} target={[0,1,0]} />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <ApplyAction target="character_a" action="sneak_walk" at="0s" duration="3.2s"
               rootMotion="match_target" destination="character_a_contact"
               takeoff="character_a_start" contact="character_a_contact"
               landing="character_a_contact"
               colliderProfile="auto" safeMargin="0.01" floorSnap="0.09"
               maxSlides="6" sweepStep="0.03"
               face="character_b" syncGroup="contact_01" syncMarker="contact" />
  <Constraint kind="position" source="character_a.hand_r"
              target="character_b.shoulder_r" from="3.58s" to="4.05s"
              solver="two_bone_ik" />
  <AnimationTarget node="FightScene" property="activeCamera">
    <Key time="0s" value="camera_feet" />
    <Key time="3.2s" value="camera_medium" />
  </AnimationTarget>
  <Present from="FightScene" />
</Graph>
"##,
        )?;
        assert_eq!(graph.assets[2].kind, GraphAssetKind::Animation);
        assert_eq!(graph.assets[2].id, "sneak_walk_source");
        assert_eq!(
            graph.actions[0].source.as_deref(),
            Some("sneak_walk_source")
        );
        assert_eq!(
            graph.actions[0].source_profile.as_deref(),
            Some("fbx_humanoid")
        );
        assert_eq!(graph.actions[0].clip.as_deref(), Some("Sneak Walk"));
        assert_eq!(graph.actions[0].markers.len(), 3);
        assert_eq!(graph.actions[0].markers[1].role.as_deref(), Some("contact"));
        assert_eq!(graph.apply_actions[0].duration_ms, Some(3_200));
        assert_eq!(
            graph.apply_actions[0].root_motion.as_deref(),
            Some("match_target")
        );
        assert_eq!(
            graph.apply_actions[0].destination.as_deref(),
            Some("character_a_contact")
        );
        assert_eq!(
            graph.apply_actions[0].takeoff.as_deref(),
            Some("character_a_start")
        );
        assert_eq!(
            graph.apply_actions[0].contact.as_deref(),
            Some("character_a_contact")
        );
        assert_eq!(
            graph.apply_actions[0].landing.as_deref(),
            Some("character_a_contact")
        );
        assert_eq!(
            graph.apply_actions[0].collider_profile.as_deref(),
            Some("auto")
        );
        assert_eq!(graph.apply_actions[0].safe_margin, "0.01");
        assert_eq!(graph.apply_actions[0].floor_snap, "0.09");
        assert_eq!(graph.apply_actions[0].max_slides, 6);
        assert_eq!(graph.apply_actions[0].sweep_step, "0.03");
        assert_eq!(graph.scene_constraints[0].duration_ms, 470);
        assert_eq!(graph.animation_targets[0].property, "activeCamera");
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("group");
        };
        let composite = group.composite.as_ref().expect("3D composite");
        assert_eq!(
            composite
                .nodes_3d
                .iter()
                .filter(|node| matches!(node, Scene3DNode::Anchor(_)))
                .count(),
            2
        );
        Ok(())
    }

    #[test]
    fn graph_parser_rejects_apply_action_pointing_to_raw_animation_asset() {
        let error = parse_graph_script(
            r#"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <AnimationAsset id="raw_walk" src="walking.glb" />
  </Assets>
  <Tex id="out" fmt="rgba8unorm" size={[320,180]} />
  <ApplyAction target="hero" action="raw_walk" />
  <Present from="out" />
</Graph>
"#,
        )
        .expect_err("raw AnimationAsset ids are not executable Actions");
        assert!(
            error.message.contains("references a raw AnimationAsset"),
            "unexpected error: {}",
            error.message
        );
    }

    #[test]
    fn graph_parser_rejects_apply_action_on_dynamic_rigid_body() {
        let error = parse_graph_script(
            r#"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <ModelAsset id="hero_asset" src="hero.glb" />
  </Assets>
  <Action id="idle" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s">
      <Bone id="hips" y="0" />
    </Pose>
  </Action>
  <ApplyAction target="hero" action="idle" />
  <Scene id="RigidScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
              <Physics gravity={[0,-9.81,0]} />
              <Model id="hero" asset="hero_asset" position={[0,2,0]} />
              <RigidBody id="hero_body" target="hero" dimension="3d"
                         type="dynamic" shape="capsule" radius="0.3" height="1.2" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="RigidScene" />
</Graph>
"#,
        )
        .expect_err("dynamic physics and ApplyAction cannot own the same transform");
        assert!(
            error.message.contains("controlled by a dynamic RigidBody"),
            "unexpected error: {}",
            error.message
        );
    }

    #[test]
    fn graph_parser_accepts_action_contacts_and_auto_contact_correction() {
        let graph = parse_graph_script(
            r#"
<Graph fps={30} duration="3s" size={[640,360]}>
  <Assets>
    <ModelAsset id="character" src="character.glb" />
    <ModelAsset id="deck" src="deck.glb" />
    <AnimationAsset id="clips" src="character.glb" />
  </Assets>
  <Action id="repair_kneel" source="clips" clip="Fixing_Kneeling">
    <Contact id="left_knee_contact" effector="knee_l" target="ground"
             from="18%" to="72%" mode="lock" weight="1" />
    <Contact id="right_foot_contact" effector="foot_r" target="ground"
             from="0.16" to="0.76" mode="lock" weight="0.9" />
  </Action>
  <ApplyAction target="technician" action="repair_kneel"
               ground="ship_deck" contactCorrection="auto" />
  <Scene id="ContactScene">
    <Timeline>
      <Track>
        <Sequence duration="3s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
              <Environment id="ship" asset="deck" collision="surfaces">
                <Surface id="ship_deck" kind="ground" collision="true"
                         height="0" boundsMin={[-2,-0.1,-2]} boundsMax={[2,0.1,2]} />
              </Environment>
              <Model id="technician" asset="character" position={[0,0,0]} />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="ContactScene" />
</Graph>
"#,
        )
        .expect("Action Contact metadata should parse");
        assert_eq!(graph.actions[0].contacts.len(), 2);
        assert!((graph.actions[0].contacts[0].from - 0.18).abs() < 0.0001);
        assert!((graph.actions[0].contacts[1].to - 0.76).abs() < 0.0001);
        assert_eq!(
            graph.apply_actions[0].contact_correction.as_deref(),
            Some("auto")
        );
        assert_eq!(graph.apply_actions[0].ground.as_deref(), Some("ship_deck"));
    }

    #[test]
    fn graph_parser_accepts_semantic_seat_surface_binding() {
        let graph = parse_graph_script(
            r#"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Assets>
    <ModelAsset id="character" src="character.glb" />
    <PrimitiveAsset id="seat_asset" shape="box" size={[2,0.1,0.7]} />
  </Assets>
  <Action id="sit" skeleton="humanoid_v1" duration="2s">
    <Contact id="pelvis_seat" effector="pelvis" target="seat"
             from="0.6" to="1" mode="surface" weight="1" />
    <Pose t="0s">
      <Bone id="hips" y="0" />
    </Pose>
  </Action>
  <ContactSurface id="seat" source="bench" kind="seat" plane="top"
                  forward={[0,0,1]} bounds={[2,0.7]} margin="0.01" />
  <ApplyAction target="actor" action="sit" contactCorrection="auto"
               contactTargets={{ seat: "seat" }} />
  <Scene id="SeatScene">
    <Timeline>
      <Track>
        <Sequence duration="2s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
              <Model id="bench" asset="seat_asset" position={[0,0.5,0]} />
              <Model id="actor" asset="character" position={[0,0,0]} />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="SeatScene" />
</Graph>
"#,
        )
        .expect("semantic seat contact should parse");
        assert_eq!(graph.contact_surfaces[0].source, "bench");
        assert_eq!(graph.actions[0].contacts[0].effector, "pelvis");
        assert_eq!(
            graph.apply_actions[0].contact_targets.get("seat"),
            Some(&"seat".to_string())
        );
        assert!(graph.apply_actions[0].ground.is_none());
    }

    #[test]
    fn graph_parser_accepts_finite_physics_surface_and_scene_gravity() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Assets>
    <ModelAsset id="character" src="character.glb" />
  </Assets>
  <Scene id="PhysicsScene">
    <Timeline>
      <Track>
        <Sequence duration="2s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
              <Physics gravity={[0,-9.81,0]} fixedStep="1/120s" iterations="4" />
              <Surface id="floor" kind="ground" collider="box"
                       center={[0,-0.1,0]} size={[20,0.2,20]} color="#202838" />
              <Model id="actor" asset="character" position={[0,5,0]}
                     collision="kinematic" gravity="scene" ground="floor" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="PhysicsScene" />
</Graph>
"##,
        )
        .expect("finite Scene physics floor should parse");
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("composite group");
        };
        let composite = group.composite.as_ref().expect("3D composite config");
        assert_eq!(composite.physics.as_ref().unwrap().fixed_step, "1/120s");
        assert_eq!(composite.nodes_3d.len(), 2);
        let generated_floor = composite
            .nodes_3d
            .iter()
            .find_map(|node| match node {
                Scene3DNode::Model(model) if model.environment => Some(model),
                _ => None,
            })
            .expect("procedural floor model");
        assert!(matches!(
            generated_floor.primitive.as_ref().map(|asset| &asset.geometry),
            Some(PrimitiveGeometry::Box { size }) if *size == [20.0, 0.2, 20.0]
        ));
        assert_eq!(generated_floor.scale, "1");
        let actor = composite
            .nodes_3d
            .iter()
            .find_map(|node| match node {
                Scene3DNode::Model(model) if model.id.as_deref() == Some("actor") => Some(model),
                _ => None,
            })
            .expect("falling actor");
        assert_eq!(actor.gravity.as_deref(), Some("scene"));
        assert_eq!(actor.ground.as_deref(), Some("floor"));
    }

    #[test]
    fn graph_parser_accepts_complete_scene_3d_lighting_stack() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="2s" size={[640,360]}>
  <Assets>
    <ImageAsset id="studio_hdri" src="studio.hdr" />
    <ModelAsset id="hero_asset" src="hero.glb" />
  </Assets>
  <Scene id="LightingScene">
    <Timeline>
      <Track>
        <Sequence duration="2s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
        <EnvironmentLight id="ibl" asset="studio_hdri" intensity="1.2"
          rotationY="35" visible="true" backgroundIntensity="0.6"
          backgroundBlur="0.2" diffuseIntensity="0.8" specularIntensity="1.4" />
        <DirectionalLight id="sun" direction={[-0.4,-1,-0.3]}
          color="#FFF1DB" intensity="3.5" castShadow="true" shadowStrength="0.85" />
        <PointLight id="practical" position={[2,2,1]} color="#78C8FF"
          intensity="18" range="8" />
        <SpotLight id="rim" position={[-2,3,2]} direction={[0,-1,-0.5]}
          intensity="24" range="10" innerCone="18" outerCone="32" />
        <RectAreaLight id="softbox" position={[0,3,2]} direction={[0,-1,-0.4]}
          intensity="8" width="3" height="2" />
        <AmbientOcclusion id="ao" intensity="0.7" radius="1.2" />
        <ContactShadow id="contact" intensity="0.8" distance="0.3" softness="0.6" />
        <ColorManagement id="grade" toneMapping="aces" exposure="1.1"
          whiteBalance="5800" contrast="1.08" />
        <Camera3D position={[0,1.4,5]} target={[0,1,0]} fov="38" />
        <Model id="hero" asset="hero_asset" material="pbr" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="LightingScene" />
</Graph>
"##,
        )
        .expect("complete Scene 3D lighting stack should parse");
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("group");
        };
        let composite = group.composite.as_ref().expect("3D composite");
        assert_eq!(composite.nodes_3d.len(), 10);
        assert!(matches!(
            composite.nodes_3d[0],
            Scene3DNode::EnvironmentLight(_)
        ));
        assert!(matches!(
            composite.nodes_3d[1],
            Scene3DNode::DirectionalLight(_)
        ));
        assert!(matches!(
            composite.nodes_3d[7],
            Scene3DNode::ColorManagement(_)
        ));
    }

    #[test]
    fn graph_parser_rejects_auto_contact_correction_without_action_contacts() {
        let error = parse_graph_script(
            r#"
<Graph fps={30} duration="1s" size={[640,360]}>
  <Assets>
    <ModelAsset id="character" src="character.glb" />
    <AnimationAsset id="clips" src="character.glb" />
  </Assets>
  <Action id="idle" source="clips" clip="Idle" />
  <ApplyAction target="actor" action="idle" ground="floor"
               contactCorrection="auto" />
  <Scene id="ContactScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
              <Environment id="stage" asset="character" collision="surfaces">
                <Surface id="floor" kind="ground" collision="true"
                         height="0" boundsMin={[-2,-0.1,-2]} boundsMax={[2,0.1,2]} />
              </Environment>
              <Model id="actor" asset="character" position={[0,0,0]} />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="ContactScene" />
</Graph>
"#,
        )
        .expect_err("auto contact correction must require Contact metadata");
        assert!(
            error.message.contains("has no <Contact /> declarations"),
            "unexpected error: {}",
            error.message
        );
    }

    #[test]
    fn graph_parser_accepts_semantic_environment_surfaces_and_node_anchors() {
        let graph = parse_graph_script(
            r#"
<Graph fps={30} duration="3s" size={[640,360]}>
  <Assets>
    <ModelAsset id="roof_asset" src="roof.glb" />
    <ModelAsset id="runner_asset" src="runner.glb" />
  </Assets>
  <Scene id="Rooftop">
    <Timeline>
      <Track>
        <Sequence duration="3s">
          <Layer>
            <CompositeGroup id="island" space="3d" depth="true">
              <Environment id="roof" asset="roof_asset" collision="surfaces"
                           up="+Y" forward="+X" unitScale="0.01"
                           scaleMode="normalize_height">
                <Surface id="roof_floor" kind="ground" space="asset" height="2.4"
                         normal={[0,1,0]} centroid={[0,2.4,0]}
                         boundsMin={[-4,2.35,-3]} boundsMax={[4,2.45,3]}
                         collision="true" collider="plane" />
                <Anchor id="takeoff" surface="roof_floor" uv={[0.2,0.5]}
                        offset={[0,0,0]} />
                <Anchor id="landing" surface="roof_floor" uv={[0.8,0.5]}
                        offset={[0,0,0]} />
              </Environment>
              <EnvironmentDebug surfaces="true" anchors="true"
                                actionPath="true" cameras="true" />
              <Model id="runner" asset="runner_asset" position="@takeoff" />
              <Camera3D id="wide" position="@takeoff" target="@runner"
                        up={[0,1,0]} horizonLock="true" roll="2" fov="40" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="Rooftop" />
</Graph>
"#,
        )
        .expect("semantic Environment graph");
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("group");
        };
        let composite = group.composite.as_ref().expect("3D composite");
        let environment = composite
            .nodes_3d
            .iter()
            .find_map(|node| match node {
                Scene3DNode::Model(model) if model.environment => Some(model),
                _ => None,
            })
            .expect("environment model");
        assert!(environment.r#static);
        assert_eq!(environment.up, "+Y");
        assert_eq!(environment.forward, "+X");
        assert_eq!(environment.unit_scale, "0.01");
        assert_eq!(environment.surfaces[0].id, "roof_floor");
        assert_eq!(environment.surfaces[0].space, "asset");
        assert!(environment.surfaces[0].collision);
        assert_eq!(environment.surfaces[0].collider.as_deref(), Some("plane"));
        assert_eq!(
            environment.surfaces[0].centroid.as_deref(),
            Some("[0,2.4,0]")
        );
        assert_eq!(
            composite
                .nodes_3d
                .iter()
                .filter(|node| matches!(node, Scene3DNode::Anchor(_)))
                .count(),
            2
        );
        assert!(
            composite
                .nodes_3d
                .iter()
                .any(|node| matches!(node, Scene3DNode::Debug(_)))
        );
    }

    #[test]
    fn graph_parser_preserves_legacy_environment_mesh_collision() {
        let graph = parse_graph_script(
            r#"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <ModelAsset id="set" src="set.glb" />
  </Assets>
  <Scene id="Legacy">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d">
              <Environment id="set_model" asset="set" collision="mesh">
                <Surface id="floor" kind="ground" height="0" />
              </Environment>
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="Legacy" />
</Graph>
"#,
        )
        .expect("legacy mesh collision remains parseable");
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("group");
        };
        let environment = group
            .composite
            .as_ref()
            .expect("composite")
            .nodes_3d
            .iter()
            .find_map(|node| match node {
                Scene3DNode::Model(model) if model.environment => Some(model),
                _ => None,
            })
            .expect("environment");
        assert_eq!(environment.collision.as_deref(), Some("mesh"));
        assert!(!environment.surfaces[0].collision);
        assert!(environment.surfaces[0].collider.is_none());
    }

    #[test]
    fn scene_model_defaults_to_authored_scale_and_accepts_physics_debug() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <PrimitiveAsset id="box" shape="box" size={[2,3,4]} color="#FFFFFF" />
  </Assets>
  <Scene id="Main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d">
              <Model id="box_model" asset="box" />
              <PhysicsDebug colliders="true" contacts="true" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="Main" />
</Graph>
"##,
        )
        .expect("authored-scale model and PhysicsDebug");
        let SceneNode::Timeline(timeline) = &graph.scenes[0].children[0] else {
            panic!("timeline");
        };
        let SceneNode::Track(track) = &timeline.children[0] else {
            panic!("track");
        };
        let SceneNode::Sequence(sequence) = &track.children[0] else {
            panic!("sequence");
        };
        let SceneNode::Layer(layer) = &sequence.children[0] else {
            panic!("layer");
        };
        let SceneNode::Group(group) = &layer.children[0] else {
            panic!("group");
        };
        let composite = group.composite.as_ref().expect("composite");
        let model = composite
            .nodes_3d
            .iter()
            .find_map(|node| match node {
                Scene3DNode::Model(model) => Some(model),
                _ => None,
            })
            .expect("model");
        assert_eq!(model.scale_mode, "none");
        assert!(composite.nodes_3d.iter().any(|node| matches!(
            node,
            Scene3DNode::Debug(debug) if debug.colliders && debug.contacts
        )));
    }

    #[test]
    fn primitive_assets_parse_all_v1_shapes() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <PrimitiveAsset id="box" shape="box" size={[1,2,3]} color="#FF000080" />
    <PrimitiveAsset id="sphere" shape="sphere" radius="0.5" segments="32" rings="12" />
    <PrimitiveAsset id="capsule" shape="capsule" radius="0.2" height="0.8" segments="24" rings="12" />
    <PrimitiveAsset id="plane" shape="plane" size={[8,6]} segments="4" />
    <PrimitiveAsset id="cylinder" shape="cylinder" radius="1" height="2" />
    <PrimitiveAsset id="cone" shape="cone" radius="1" height="2" segments="16" />
    <PrimitiveAsset id="wedge" shape="wedge" size={[4,1,3]} />
  </Assets>
  <Scene id="canvas">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="canvas" />
</Graph>
"##,
        )
        .expect("all V1 primitive shapes should parse");
        assert_eq!(graph.assets.len(), 7);
        assert!(matches!(
            graph.assets[0].source,
            GraphAssetSource::Primitive(_)
        ));
        let sphere = graph.assets[1].primitive().expect("typed sphere");
        assert!(matches!(
            sphere.geometry,
            PrimitiveGeometry::Sphere {
                segments: 32,
                rings: 12,
                ..
            }
        ));
        assert_eq!(sphere.collision.mode, PrimitiveCollisionMode::None);
        assert_eq!(sphere.collision.collider, PrimitiveColliderShape::Auto);
    }

    #[test]
    fn primitive_asset_block_parses_advanced_shapes_and_modifiers() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <PrimitiveAsset id="body" shape="ellipsoid" radii={[0.8,1.2,0.55]}
                    segments="24" rings="12" color="#C87848">
      <Modifiers>
        <Taper axis="y" start="1.08" end="0.82" />
        <Twist axis="y" angle="8" />
        <Bend axis="x" angle="-5" pivot={[0,0,0]} />
        <Subdivision levels="1" />
        <WeightedNormals strength="0.75" keepSharpEdges="true" />
      </Modifiers>
      <MeshBuild topology="quads" triangulation="shortestDiagonal"
                 quality="high" maxTriangles="10000" />
      <LOD mode="auto" levels="3" preserveSilhouette="true" />
    </PrimitiveAsset>
    <PrimitiveAsset id="base" shape="roundedBox" size={[2,0.3,1.2]}
                    radius="0.08" segments="3" />
    <PrimitiveAsset id="waist" shape="frustum" topSize={[0.8,0.5]}
                    bottomSize={[0.6,0.42]} height="0.9" />
  </Assets>
  <Scene id="canvas">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="canvas" />
</Graph>
"##,
        )
        .expect("advanced PrimitiveAsset block");
        let body = graph.assets[0].primitive().expect("ellipsoid");
        assert!(matches!(body.geometry, PrimitiveGeometry::Ellipsoid { .. }));
        assert_eq!(body.modifiers.len(), 5);
        assert_eq!(body.mesh_build.topology, "quads");
        assert_eq!(body.mesh_build.quality, "high");
        assert_eq!(body.lod.levels, 3);
        assert!(matches!(
            graph.assets[1].primitive().expect("rounded box").geometry,
            PrimitiveGeometry::RoundedBox { .. }
        ));
        assert!(matches!(
            graph.assets[2].primitive().expect("frustum").geometry,
            PrimitiveGeometry::Frustum { .. }
        ));
    }

    #[test]
    fn primitive_asset_block_enforces_triangle_budget() {
        let error = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <PrimitiveAsset id="too_dense" shape="sphere" radius="1" segments="32" rings="16">
      <Modifiers>
        <Subdivision levels="2" />
      </Modifiers>
      <MeshBuild maxTriangles="100" />
    </PrimitiveAsset>
  </Assets>
  <Scene id="canvas">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="canvas" />
</Graph>
"##,
        )
        .expect_err("triangle budget must reject an oversized build");
        assert!(error.message.contains("exceeding MeshBuild maxTriangles"));
    }

    #[test]
    fn terrain_asset_resolves_height_material_layers_and_blend_map() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[128,128]}>
  <Assets>
    <ImageAsset id="height" src="height.png" colorSpace="linear-srgb" />
    <ImageAsset id="blend" src="blend.png" colorSpace="linear-srgb" />
    <ImageAsset id="soil_color" src="soil.png" colorSpace="srgb" />
    <MaterialAsset id="soil" baseColorTexture="soil_color" roughness="0.9" />
    <MaterialAsset id="grass" baseColor="#4A713B" roughness="0.8" />
    <TerrainAsset id="ground" heightMap="height" size={[40,30]}
                  heightScale="5" heightOffset="-1" layers={["soil","grass"]}
                  blendMap="blend" chunks={[4,3]} lod="half" collision="solid" />
  </Assets>
  <Scene id="TerrainScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="TerrainScene" />
</Graph>"##,
        )
        .expect("terrain graph should parse");
        let terrain = graph
            .assets
            .iter()
            .find_map(|asset| asset.terrain())
            .unwrap();
        assert_eq!(terrain.height_map_src.as_deref(), Some("height.png"));
        assert_eq!(terrain.blend_map_src.as_deref(), Some("blend.png"));
        assert_eq!(terrain.layer_definitions.len(), 2);
        assert_eq!(terrain.chunks, [4, 3]);
        assert_eq!(terrain.lod, "half");
        assert_eq!(terrain.collision, PrimitiveCollisionMode::Solid);
    }

    #[test]
    fn terrain_asset_rejects_blend_map_without_layers() {
        let error = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[128,128]}>
  <Assets>
    <ImageAsset id="height" src="height.png" />
    <ImageAsset id="blend" src="blend.png" />
    <MaterialAsset id="soil" baseColor="#604A38" />
    <TerrainAsset id="ground" heightMap="height" size={[10,10]}
                  material="soil" blendMap="blend" />
  </Assets>
  <Scene id="scene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("blendMap without layers must fail");
        assert!(error.message.contains("layers and blendMap together"));
    }

    #[test]
    fn terrain_asset_requires_linear_height_map_data() {
        let error = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[128,128]}>
  <Assets>
    <ImageAsset id="height" src="height.png" colorSpace="srgb" />
    <MaterialAsset id="soil" baseColor="#604A38" />
    <TerrainAsset id="ground" heightMap="height" size={[10,10]} material="soil" />
  </Assets>
  <Scene id="scene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("height fields must not be gamma decoded");
        assert!(
            error
                .message
                .contains("must declare colorSpace=\"linear-srgb\""),
            "unexpected parser error: {}",
            error.message
        );
    }

    #[test]
    fn vegetation_assets_resolve_conditional_materials_and_runtime_settings() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[128,128]}>
  <Assets>
    <MaterialAsset id="bark" baseColor="#4B3426" roughness="0.95" />
    <MaterialAsset id="leaves" baseColor="#315D2C" roughness="0.88" doubleSided="true" />
    <VegetationAsset id="oak" kind="tree" height="7.5"
      trunkMaterial="bark" foliageMaterial="leaves" density="24"
      branchLevels="3" seed="77" lod="half" wind="true" collision="solid" />
    <VegetationAsset id="fern" kind="fern" height="0.8"
      material="leaves" density="18" seed="9" lod="auto" wind="true" />
  </Assets>
  <Scene id="vegetation_scene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer id="vegetation_empty_layer">
            <Rect id="vegetation_probe" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="vegetation_scene" />
</Graph>"##,
        )
        .expect("V1 vegetation assets should parse");
        let oak = graph
            .assets
            .iter()
            .find_map(|asset| asset.vegetation())
            .unwrap();
        assert_eq!(oak.kind, VegetationKind::Tree);
        assert_eq!(oak.lod, VegetationLod::Half);
        assert!(oak.wind);
        assert!(oak.trunk_material_definition.is_some());
        assert!(oak.foliage_material_definition.is_some());
        assert_eq!(oak.collision, PrimitiveCollisionMode::Solid);
    }

    #[test]
    fn vegetation_asset_rejects_kind_specific_attributes() {
        let error = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[128,128]}>
  <Assets>
    <MaterialAsset id="grass" baseColor="#315D2C" />
    <VegetationAsset id="bad" kind="grass" height="0.8"
      material="grass" branchLevels="2" />
  </Assets>
  <Scene id="scene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("grass must not silently accept branch settings");
        assert!(error.message.contains("does not support branchLevels"));
    }

    #[test]
    fn primitive_asset_resolves_first_class_pbr_material_and_visual_bevel() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <ImageAsset id="stone_color" src="stone.jpg" colorSpace="srgb" />
    <MaterialAsset id="stone" shading="pbr" baseColor="#D8D3CA"
      baseColorTexture="stone_color" metallic="0" roughness="0.84"
      specular="0.28" mapping="triplanar" textureScale={[2.4,2.4]}
      variationAmount={[0.2,0.15]} />
    <PrimitiveAsset id="step" shape="box" size={[4.4,0.32,0.9]}
      material="stone" bevelRadius="0.025" bevelSegments="3"
      materialSeed="76" collision="solid" collider="box" />
  </Assets>
  <Scene id="Main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="Main" />
</Graph>
"##,
        )
        .expect("typed PBR material should resolve into a beveled primitive");
        assert_eq!(graph.material_assets.len(), 1);
        let primitive = graph
            .assets
            .iter()
            .find_map(|asset| asset.primitive())
            .unwrap();
        assert_eq!(primitive.material.as_deref(), Some("stone"));
        assert_eq!(primitive.bevel_radius, 0.025);
        assert_eq!(primitive.bevel_segments, 3);
        assert_eq!(primitive.material_seed, Some(76));
        let material = primitive.material_definition.as_ref().unwrap();
        assert_eq!(
            material.base_color_texture_src.as_deref(),
            Some("stone.jpg")
        );
        assert_eq!(material.mapping, "triplanar");
        assert_eq!(primitive.collision.collider, PrimitiveColliderShape::Box);
    }

    #[test]
    fn primitive_material_parses_transmissive_glass_without_changing_collision() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <MaterialAsset id="glass" shading="pbr" baseColor="#E8F7FA"
      roughness="0.08" specular="1" transmission="0.94" ior="1.52"
      thickness="0.012" attenuationColor="#B7DDE2" attenuationDistance="6"
      depthWrite="auto" sortPriority="3" doubleSided="true" />
    <PrimitiveAsset id="pane" shape="box" size={[0.012,2.35,3.8]}
      material="glass" collision="none" />
  </Assets>
  <Scene id="Main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="Main" />
</Graph>
"##,
        )
        .expect("transmissive glass should parse through the existing MaterialAsset tag");
        let material = &graph.material_assets[0];
        assert_eq!(material.transmission, 0.94);
        assert_eq!(material.ior, 1.52);
        assert_eq!(material.thickness, 0.012);
        assert_eq!(material.attenuation_distance, 6.0);
        assert_eq!(material.depth_write, "auto");
        assert_eq!(material.sort_priority, 3);
        assert_eq!(
            graph.assets[0].primitive().unwrap().collision.mode,
            PrimitiveCollisionMode::None
        );
    }

    #[test]
    fn primitive_material_rejects_invalid_transmission_depth_write() {
        let error = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <MaterialAsset id="glass" transmission="0.8" depthWrite="sometimes" />
  </Assets>
  <Scene id="Main">
    <Timeline />
  </Scene>
  <Present from="Main" />
</Graph>
"##,
        )
        .expect_err("invalid depth write policy must fail clearly");
        assert!(error.message.contains("Use auto, true, or false"));
    }

    #[test]
    fn primitive_material_rejects_unknown_image_asset() {
        let error = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <MaterialAsset id="stone" baseColorTexture="missing" />
    <PrimitiveAsset id="step" shape="box" size={[1,1,1]} material="stone" />
  </Assets>
  <Scene id="Main">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="Main" />
</Graph>
"##,
        )
        .expect_err("unknown PBR texture asset must fail clearly");
        assert!(error.message.contains("unknown ImageAsset \"missing\""));
    }

    #[test]
    fn primitive_collision_defaults_to_auto_and_allows_visual_collider_mismatch() {
        let graph = parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <PrimitiveAsset id="auto_box" shape="box" size={[1,2,3]} collision="solid" />
    <PrimitiveAsset id="sphere_with_box" shape="sphere" radius="1" collision="solid" collider="box" colliderSize={[2,3,4]} colliderOffset={[0,0.5,0]} />
  </Assets>
  <Scene id="canvas">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="canvas" />
</Graph>
"##,
        )
        .expect("universal primitive collision should parse");
        let auto = graph.assets[0]
            .primitive()
            .expect("auto collider primitive");
        assert_eq!(auto.collision.mode, PrimitiveCollisionMode::Solid);
        assert_eq!(auto.collision.collider, PrimitiveColliderShape::Auto);
        let mismatched = graph.assets[1]
            .primitive()
            .expect("mismatched visual and collider primitive");
        assert_eq!(mismatched.collision.collider, PrimitiveColliderShape::Box);
        assert_eq!(
            mismatched.collision.size.as_deref(),
            Some(&[2.0, 3.0, 4.0][..])
        );
        assert_eq!(mismatched.collision.offset, [0.0, 0.5, 0.0]);
    }

    #[test]
    fn primitive_collision_rejects_settings_when_disabled_and_shape_specific_mistakes() {
        let disabled = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <PrimitiveAsset id="x" shape="sphere" radius="1" collider="box" />
  </Assets>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("disabled collision cannot carry collider settings");
        assert!(disabled.message.contains("collision=\"none\""));

        let wrong_size = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <PrimitiveAsset id="x" shape="sphere" radius="1" collision="solid" colliderRadius="1" colliderSize={[1,1,1]} />
  </Assets>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("sphere collider cannot use colliderSize");
        assert!(wrong_size.message.contains("colliderSize is only valid"));
    }

    #[test]
    fn compound_asset_composes_primitive_instances_and_rejects_external_children() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <PrimitiveAsset id="step" shape="box" size={[2,0.2,0.5]} collision="solid" />
    <CompoundAsset id="stairs">
      <Instance id="low" asset="step" position={[0,0.1,0]} />
      <Instance id="high" asset="step" position={[0,0.3,-0.5]} scale="1" />
    </CompoundAsset>
  </Assets>
  <Scene id="canvas">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="canvas" />
</Graph>"##,
        )
        .expect("compound primitive asset should parse");
        let compound = graph.assets[1].compound().expect("typed compound asset");
        assert_eq!(compound.instances.len(), 2);
        assert_eq!(compound.instances[1].position, [0.0, 0.3, -0.5]);

        let invalid = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <ModelAsset id="external" src="model.glb" />
    <CompoundAsset id="invalid">
      <Instance asset="external" />
    </CompoundAsset>
  </Assets>
  <Background id="canvas" color="#000000" />
  <Present from="canvas" />
</Graph>"##,
        )
        .expect_err("compound V1 must reference primitives");
        assert!(invalid.message.contains("must reference a PrimitiveAsset"));
    }

    #[test]
    fn primitive_assets_reject_shape_attribute_conflicts_and_bad_values() {
        let conflict = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <PrimitiveAsset id="x" shape="sphere" size={[1,1,1]} />
  </Assets>
  <Background id="canvas" color="#000000" />
  <Present from="canvas" />
</Graph>"##,
        )
        .expect_err("sphere size must be rejected");
        assert!(
            conflict.message.contains("does not support \"size\""),
            "{conflict:?}"
        );

        let invalid = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <PrimitiveAsset id="x" shape="cone" radius="0" height="2" segments="300" />
  </Assets>
  <Background id="canvas" color="#000000" />
  <Present from="canvas" />
</Graph>"##,
        )
        .expect_err("invalid primitive values must be rejected");
        assert!(invalid.message.contains("greater than zero") || invalid.message.contains("256"));
    }

    #[test]
    fn capsule_and_rigged_compound_assets_parse_as_typed_data() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[64,64]}>
  <Assets>
    <PrimitiveAsset id="limb" shape="capsule" radius="0.1" height="0.4"
                    segments="20" rings="10" collision="solid" />
    <CompoundAsset id="actor" rig="rig">
      <Instance id="arm" asset="limb" bone="upper_arm_l" position={[0,-0.2,0]} />
    </CompoundAsset>
  </Assets>
  <Skeleton id="rig" profile="motionloom_humanoid_v1"
            sourceRig="character1_reference_v1" space="3d">
    <Bone id="root" role="root" position={[0,0,0]} />
    <Bone id="upper_arm_l" role="upper_arm" side="left" parent="root"
          position={[-0.2,1.2,0]} rotation={[0,0,-5]} />
    <BoneAxisMap>
      <Axis bone="upper_arm_l" forward="rotationX:1" side="rotationZ:-1" />
    </BoneAxisMap>
  </Skeleton>
  <Scene id="canvas">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1" height="1" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="canvas" />
</Graph>"##,
        )
        .expect("native rigged compound should parse");
        let primitive = graph.assets[0].primitive().expect("capsule primitive");
        assert!(matches!(
            primitive.geometry,
            PrimitiveGeometry::Capsule { .. }
        ));
        assert_eq!(primitive.collision.collider, PrimitiveColliderShape::Auto);
        let compound = graph.assets[1].compound().expect("compound asset");
        assert_eq!(compound.rig.as_deref(), Some("rig"));
        assert_eq!(compound.instances[0].bone.as_deref(), Some("upper_arm_l"));
        assert_eq!(graph.skeletons[0].space, "3d");
        assert_eq!(
            graph.skeletons[0].source_rig.as_deref(),
            Some("character1_reference_v1")
        );
        let axes = graph.skeletons[0]
            .bone_axis_map
            .as_ref()
            .expect("native rig axis map");
        assert_eq!(axes.axes[0].bone, "upper_arm_l");
        assert_eq!(axes.axes[0].side.as_deref(), Some("rotationZ:-1"));
        assert_eq!(graph.skeletons[0].bones[1].z, "0");
        assert_eq!(graph.skeletons[0].bones[1].rotation_z, "-5");
    }

    #[test]
    fn native_smooth_skin_is_opt_in_and_rigid_instances_keep_their_default() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[64,64]}>
  <Assets>
    <PrimitiveAsset id="limb" shape="capsule" radius="0.1" height="0.5" />
    <CompoundAsset id="hero" rig="rig">
      <SkinBinding mode="automatic" maxInfluences="3" falloff="2.25" normalize="true" />
      <Instance id="soft_arm" asset="limb" bone="arm" skin="smooth"
                influences={["root","arm"]} />
      <Instance id="rigid_hand" asset="limb" bone="arm" />
    </CompoundAsset>
  </Assets>
  <Skeleton id="rig" space="3d">
    <Bone id="root" position={[0,0,0]} />
    <Bone id="arm" parent="root" position={[0,1,0]} />
  </Skeleton>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect("opt-in native skin should parse");
        let compound = graph.assets[1].compound().expect("compound");
        let binding = compound.skin_binding.as_ref().expect("skin binding");
        assert_eq!(binding.max_influences, 3);
        assert_eq!(binding.falloff, 2.25);
        assert_eq!(compound.instances[0].skin, crate::NativeSkinMode::Smooth);
        assert_eq!(compound.instances[0].influences, ["root", "arm"]);
        assert_eq!(compound.instances[1].skin, crate::NativeSkinMode::Rigid);

        let invalid = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[64,64]}>
  <Assets>
    <PrimitiveAsset id="limb" shape="capsule" radius="0.1" height="0.5" />
    <CompoundAsset id="hero" rig="rig">
      <SkinBinding />
      <Instance id="missing_primary" asset="limb" skin="smooth" />
    </CompoundAsset>
  </Assets>
  <Skeleton id="rig" space="3d">
    <Bone id="root" position={[0,0,0]} />
  </Skeleton>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("smooth instances require a primary bone");
        assert!(invalid.message.contains("no primary bone"));
    }

    #[test]
    fn loft_ribbon_weight_regions_and_mirror_are_additive() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[64,64]}>
  <Assets>
    <PrimitiveAsset id="coat" shape="loft">
      <Loft segments="12">
        <Section at="-0.5" width="0.7" depth="0.4" profile="rounded_rect" />
        <Section at="0.5" width="0.9" depth="0.45" profile="ellipse" offset={[0.1,0]} />
      </Loft>
    </PrimitiveAsset>
    <PrimitiveAsset id="hair" shape="ribbon">
      <Ribbon width="0.2" thickness="0.03" facing="camera_safe">
        <PathPoint position={[0,0,0]} />
        <PathPoint position={[0,-0.5,0.1]} width="0.08" roll="12" />
      </Ribbon>
    </PrimitiveAsset>
    <CompoundAsset id="hero" rig="rig">
      <SkinBinding>
        <WeightRegion instance="coat" bone="root" center={[0,0,0]} radius="0.4" operation="replace" />
      </SkinBinding>
      <Instance id="coat" asset="coat" bone="root" skin="smooth" />
      <Mirror axis="x" suffix={["_l","_r"]}>
        <Instance id="hair_l" asset="hair" bone="arm_l" position={[-0.2,0,0]} />
      </Mirror>
    </CompoundAsset>
  </Assets>
  <Skeleton id="rig" space="3d">
    <Bone id="root" position={[0,0,0]} />
    <Bone id="arm_l" parent="root" position={[-1,0,0]} />
    <Bone id="arm_r" parent="root" position={[1,0,0]} />
  </Skeleton>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect("advanced native character geometry should parse");
        let loft = graph.assets[0].primitive().expect("loft");
        assert_eq!(loft.geometry.shape_name(), "loft");
        let ribbon = graph.assets[1].primitive().expect("ribbon");
        assert_eq!(ribbon.geometry.shape_name(), "ribbon");
        let compound = graph.assets[2].compound().expect("compound");
        assert_eq!(compound.instances.len(), 3);
        assert_eq!(compound.instances[2].id, "hair_r");
        assert_eq!(compound.instances[2].bone.as_deref(), Some("arm_r"));
        assert_eq!(compound.instances[2].position[0], 0.2);
        assert_eq!(
            compound.skin_binding.as_ref().unwrap().weight_regions.len(),
            1
        );
    }

    #[test]
    fn hair_asset_preserves_guides_while_selecting_cards() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[64,64]}>
  <Assets>
    <MaterialAsset id="hair" baseColor="#665C60" />
    <HairAsset id="bangs" material="hair" bindBone="head"
               defaultRepresentation="cards" seed="9">
      <HairGroom>
        <HairGroup id="front" role="bang">
          <HairGuide id="center">
            <HairPoint position={[0,0.3,0]} width="0.2" camber="0.1" />
            <HairPoint position={[0,-0.1,0.1]} width="0.14" roll="4" />
            <HairPoint position={[-0.04,-0.4,0.08]} width="0.02" />
          </HairGuide>
        </HairGroup>
      </HairGroom>
      <HairRepresentations>
        <HairCards id="cards" lengthSegments="12" widthSegments="4"
                   thickness="0.01" crossSection="arched" tipShape="point" />
      </HairRepresentations>
      <HairLOD representation="cards" />
    </HairAsset>
  </Assets>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect("guide-authored cards should parse");
        let hair = graph.assets[0].primitive().expect("compiled hair asset");
        let PrimitiveGeometry::HairCards {
            representation_id,
            bind_bone,
            guides,
            ..
        } = &hair.geometry
        else {
            panic!("expected retained hair card geometry");
        };
        assert_eq!(representation_id, "cards");
        assert_eq!(bind_bone.as_deref(), Some("head"));
        assert_eq!(guides[0].id, "center");
        assert_eq!(guides[0].points.len(), 3);
        assert_eq!(hair.material_definition.as_ref().unwrap().id, "hair");
    }

    #[test]
    fn head_asset_keeps_face_layout_optional_for_creature_anatomy() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[64,64]}>
  <Assets>
    <MaterialAsset id="scales" baseColor="#355A42" />
    <HeadAsset id="dragon_head" material="scales" archetype="dragon" symmetry="x">
      <HeadShape size={[1.2,0.8,1.5]} forehead="0.82" cheekWidth="0.9"
                 jawWidth="0.72" chinLength="0.04" />
      <HeadFeature id="muzzle" kind="muzzle" center={[0,-0.12,0.78]}
                   size={[0.52,0.34,0.42]} amount="0.32" offset={[0,0,0.02]} />
      <HeadFeature id="horn_l" kind="horn" center={[-0.48,0.68,-0.12]}
                   size={[0.18,0.30,0.22]} amount="0.18" mirror="x" falloff="sharp" />
      <HeadMorph headWidth="1.08" muzzleLength="1.25" />
    </HeadAsset>
  </Assets>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect("creature heads should not require a human FaceLayout");
        let head = graph.assets[0].primitive().expect("compiled head asset");
        let PrimitiveGeometry::HeadSurface {
            archetype,
            face_layout,
            features,
            morph,
            ..
        } = &head.geometry
        else {
            panic!("expected procedural head surface");
        };
        assert_eq!(archetype, "dragon");
        assert!(face_layout.is_none());
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].offset, [0.0, 0.0, 0.02]);
        assert_eq!(morph.muzzle_length, 1.25);
    }

    #[test]
    fn removed_procedural_box_shorthand_has_migration_error() {
        let error = parse_graph_script(
            r##"<Graph fps={30} duration="1s" size={[1,1]}>
  <Assets>
    <ModelAsset id="x" src="motionloom:box:1:1:1:FFFFFF" />
  </Assets>
  <Background id="canvas" color="#000000" />
  <Present from="canvas" />
</Graph>"##,
        )
        .expect_err("removed shorthand must fail");
        assert!(error.message.contains("shorthand has been removed"));
    }

    #[test]
    fn action_library_declaration_registers_namespaced_actions() {
        let graph = parse_graph_script(
            r##"<Graph fps={30} duration="2s" size={[320,180]}>
  <ActionLibrary id="performance" src="actions.motionloom" actions={["bow"]} />
  <Background color="#000000" />
  <ApplyAction target="actor" action="performance.bow" at="0s" duration="2s" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect("namespaced library action should validate before asset resolution");
        assert_eq!(graph.action_libraries[0].id, "performance");
        assert_eq!(graph.action_libraries[0].actions, ["bow"]);
    }

    #[test]
    fn standalone_action_library_parses_authored_actions() {
        let actions = parse_action_library_document(
            r#"<ActionLibrary id="performance" skeleton="humanoid_v1">
  <!-- Exported Action Editor metadata can span
       more than one line. -->
  <Action id="bow" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s">
      <Bone id="hips" bend="0" />
    </Pose>
    <Pose t="1s">
      <Bone id="hips" bend="8" />
    </Pose>
  </Action>
</ActionLibrary>"#,
        )
        .expect("standalone library should parse");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].id, "bow");
    }

    #[test]
    fn graph_attachment_pairs_inline_socket_with_left_hand() {
        let graph = parse_graph_script(
            r##"<Graph fps={24} duration="2s" size={[640,360]}>
  <Assets>
    <ModelAsset id="character_asset" src="character.glb" />
    <ModelAsset id="sword_asset" src="sword.glb" />
  </Assets>
  <Attachment id="hero_sword_left_grip" object="sword">
    <Socket id="grip" position={[0,-0.12,0]} rotation={[0,0,90]} />
    <Attach target="character.hand_l" mode="snap"
            positionWeight="1" rotationWeight="1" />
  </Attachment>
  <Scene id="AttachmentScene">
    <Timeline>
      <Track>
        <Sequence duration="2s">
          <Layer>
            <CompositeGroup space="3d" depth="true">
              <Model id="character" asset="character_asset" />
              <Model id="sword" asset="sword_asset" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="AttachmentScene" />
</Graph>"##,
        )
        .expect("inline socket attachment should parse");
        let attachment = &graph.attachments[0];
        assert_eq!(attachment.object, "sword");
        assert_eq!(attachment.sockets[0].position, ["0", "-0.12", "0"]);
        assert_eq!(attachment.attaches[0].target_model, "character");
        assert_eq!(attachment.attaches[0].target_bone, "hand_l");
    }

    #[test]
    fn graph_attachment_defaults_to_snap_and_full_weights() {
        let graph = parse_graph_script(
            r##"<Graph fps={24} duration="1s" size={[64,64]}>
  <Assets>
    <ModelAsset id="a" src="a.glb" />
    <ModelAsset id="b" src="b.glb" />
  </Assets>
  <Attachment id="grip" object="prop">
    <Socket id="origin" />
    <Attach target="actor.hand_r" />
  </Attachment>
  <Scene id="AttachmentScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d">
              <Model id="actor" asset="a" />
              <Model id="prop" asset="b" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="AttachmentScene" />
</Graph>"##,
        )
        .expect("attachment defaults should parse");
        let attach = &graph.attachments[0].attaches[0];
        assert_eq!(attach.mode, "snap");
        assert_eq!(attach.position_weight, "1");
        assert_eq!(attach.rotation_weight, "1");
        assert_eq!(attach.drive, "object");
        assert_eq!(attach.socket, "origin");
    }

    #[test]
    fn graph_attachment_pairs_primary_snap_with_secondary_hand_ik() {
        let graph = parse_graph_script(
            r##"<Graph fps={24} duration="1s" size={[64,64]}>
  <Assets>
    <ModelAsset id="hero_asset" src="hero.glb" />
    <ModelAsset id="sword_asset" src="sword.glb" />
  </Assets>
  <Attachment id="two_hand_grip" object="sword">
    <Socket id="main" />
    <Socket id="support" position={[0,0.2,0]} />
    <Attach socket="main" target="hero.hand_r" mode="snap" drive="object" />
    <Attach socket="support" target="hero.hand_l" mode="ik" drive="target"
            positionWeight="1" rotationWeight="0.8" maxStretch="1.05" />
  </Attachment>
  <Scene id="TwoHandScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d">
              <Model id="hero" asset="hero_asset" />
              <Model id="sword" asset="sword_asset" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="TwoHandScene" />
</Graph>"##,
        )
        .expect("two-hand Attachment should parse");
        let attachment = &graph.attachments[0];
        assert_eq!(attachment.sockets.len(), 2);
        assert_eq!(attachment.attaches.len(), 2);
        assert_eq!(attachment.attaches[1].drive, "target");
        assert_eq!(attachment.attaches[1].mode, "ik");
        assert_eq!(attachment.attaches[1].target_bone, "hand_l");
    }

    #[test]
    fn graph_attachment_requires_one_socket_and_one_attach() {
        let error = parse_graph_script(
            r##"<Graph fps={24} duration="1s" size={[64,64]}>
  <Attachment id="broken" object="prop">
    <Socket id="origin" />
  </Attachment>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("an incomplete attachment must fail during parsing");
        assert!(error.message.contains("missing Attach"));
    }

    #[test]
    fn graph_attachment_requires_model_dot_bone_target() {
        let error = parse_graph_script(
            r##"<Graph fps={24} duration="1s" size={[64,64]}>
  <Attachment id="broken" object="prop">
    <Socket id="origin" />
    <Attach target="hand_r" />
  </Attachment>
  <Background color="#000000" />
  <Present from="scene" />
</Graph>"##,
        )
        .expect_err("ambiguous bone-only attachment target must fail");
        assert!(error.message.contains("model-id.canonical-bone"));
    }

    #[test]
    fn graph_attachment_rejects_cycles() {
        let error = parse_graph_script(
            r##"<Graph fps={24} duration="1s" size={[64,64]}>
  <Assets>
    <ModelAsset id="a" src="a.glb" />
    <ModelAsset id="b" src="b.glb" />
  </Assets>
  <Attachment id="a_to_b" object="a_model">
    <Socket id="origin" />
    <Attach target="b_model.hand_r" />
  </Attachment>
  <Attachment id="b_to_a" object="b_model">
    <Socket id="origin" />
    <Attach target="a_model.hand_l" />
  </Attachment>
  <Scene id="AttachmentScene">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <Layer>
            <CompositeGroup space="3d">
              <Model id="a_model" asset="a" />
              <Model id="b_model" asset="b" />
            </CompositeGroup>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="AttachmentScene" />
</Graph>"##,
        )
        .expect_err("attachment cycles must fail before rendering");
        assert!(error.message.contains("cycle"));
    }
}
