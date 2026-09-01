// =========================================
// =========================================
// crates/motionloom/src/scene/model.rs

use serde::{Deserialize, Serialize};

use crate::scene::dsl::{ImageNode, SvgNode};
use crate::scene::text::TextNode;
use crate::simulation::model::{SimulationBindingNode, SimulationResourceNode};

fn default_scene_blend() -> String {
    "normal".to_string()
}

fn default_stroke_style() -> String {
    "solid".to_string()
}

fn default_stroke_roughness() -> String {
    "0".to_string()
}

fn default_stroke_copies() -> String {
    "1".to_string()
}

fn default_stroke_texture() -> String {
    "0".to_string()
}

fn default_stroke_bristles() -> String {
    "0".to_string()
}

fn default_stroke_pressure() -> String {
    "none".to_string()
}

fn default_stroke_pressure_min() -> String {
    "1".to_string()
}

fn default_stroke_pressure_curve() -> String {
    "1".to_string()
}

fn default_scene_zero() -> String {
    "0".to_string()
}

fn default_scene_one() -> String {
    "1".to_string()
}

fn default_scene_source_time() -> String {
    "local".to_string()
}

fn default_scene_out_hold() -> String {
    "hold".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneRootNode {
    pub id: String,
    pub size: Option<(u32, u32)>,
    /// Process-backed effects applied after every Track in this Scene has been
    /// composited. This deliberately references the existing Process pipeline
    /// instead of creating a second effect language.
    #[serde(default)]
    pub effects: Vec<SceneEffectRef>,
    /// Final Scene-local effects. Kept separate from `effects` so the compiler
    /// can preserve an explicit post-composite scope in the render-pass DAG.
    #[serde(default)]
    pub post_effects: Vec<SceneEffectRef>,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneEffectRef {
    pub process: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub params: Vec<SceneEffectParam>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneEffectParam {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
// Boxing Path would break the public AST and its construction API. Keep the
// stable enum shape until an intentional AST version change is made.
#[allow(clippy::large_enum_variant)]
pub enum SceneNode {
    Defs(DefsNode),
    Timeline(SceneTimelineNode),
    Track(SceneTrackNode),
    Sequence(SceneSequenceNode),
    Chain(SceneChainNode),
    Palette(PaletteNode),
    PixelGrid(PixelGridNode),
    Text(Box<TextNode>),
    Image(ImageNode),
    Svg(SvgNode),
    Rect(RectNode),
    Circle(CircleNode),
    Ellipse(EllipseNode),
    Line(LineNode),
    Polyline(PolylineNode),
    Path(PathNode),
    FaceJaw(FaceJawNode),
    Shadow(ShadowNode),
    Group(GroupNode),
    Part(PartNode),
    Repeat(RepeatNode),
    Mask(MaskNode),
    Precompose(PrecomposeNode),
    Use(UseNode),
    Layer(SceneLayerNode),
    Camera(CameraNode),
    Character(CharacterNode),
    Puppet(PuppetNode),
    Pin(PinNode),
    LimbEnvelope(LimbEnvelopeNode),
    LimbRegion(LimbRegionNode),
    MeshTopology(MeshTopologyNode),
    Vertex(VertexNode),
    Triangle(TriangleNode),
    Edge(EdgeNode),
    Region(RegionNode),
    Simulation(SimulationBindingNode),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneTimelineNode {
    pub id: Option<String>,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneTrackNode {
    pub id: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default = "default_scene_track_space")]
    pub space: String,
    pub z: i32,
    /// Stable 2D/3D compositing order. When omitted, legacy `z` remains the
    /// ordering key so existing showcases render unchanged.
    #[serde(default)]
    pub composite_order: Option<i32>,
    #[serde(default = "default_scene_zero")]
    pub z_depth: String,
    #[serde(default)]
    pub effects: Vec<SceneEffectRef>,
    pub children: Vec<SceneNode>,
}

fn default_scene_track_space() -> String {
    "world".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneSequenceNode {
    pub id: Option<String>,
    pub from_ms: u64,
    pub duration_ms: u64,
    pub out: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneChainNode {
    pub id: Option<String>,
    pub from_ms: u64,
    pub gap_ms: i64,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DefsNode {
    pub id: Option<String>,
    pub gradients: Vec<GradientDef>,
    #[serde(default)]
    pub textures: Vec<TextureDef>,
    #[serde(default)]
    pub noises: Vec<NoiseDef>,
    #[serde(default)]
    pub materials: Vec<MaterialDef>,
    #[serde(default)]
    pub brushes: Vec<BrushDef>,
    #[serde(default)]
    pub masks: Vec<MaskNode>,
    #[serde(default)]
    pub precomposes: Vec<PrecomposeNode>,
    #[serde(default)]
    pub components: Vec<ComponentNode>,
    #[serde(default)]
    pub filters: Vec<FilterDef>,
    #[serde(default)]
    pub fonts: Vec<FontDef>,
    #[serde(default)]
    pub palettes: Vec<PaletteNode>,
    #[serde(default)]
    pub simulation: Vec<SimulationResourceNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NoiseDef {
    pub id: String,
    #[serde(default = "default_noise_kind")]
    pub kind: String,
    #[serde(default = "default_texture_scale")]
    pub scale: String,
    #[serde(default = "default_noise_octaves")]
    pub octaves: String,
    #[serde(default = "default_scene_zero")]
    pub seed: String,
    #[serde(default = "default_scene_one")]
    pub contrast: String,
    #[serde(default = "default_scene_zero")]
    pub evolution: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterialDef {
    pub id: String,
    pub texture: Option<String>,
    #[serde(default = "default_texture_strength")]
    pub texture_amount: String,
    pub displacement: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub displacement_amount: String,
    #[serde(default = "default_material_roughness")]
    pub roughness: String,
    #[serde(default = "default_material_specular")]
    pub specular: String,
    #[serde(default = "default_scene_one")]
    pub opacity: String,
    #[serde(default = "default_scene_zero")]
    pub refraction: String,
    #[serde(default = "default_scene_zero")]
    pub dispersion: String,
    #[serde(default = "default_scene_zero")]
    pub glass: String,
}

fn default_noise_kind() -> String {
    "fbm".to_string()
}
fn default_noise_octaves() -> String {
    "4".to_string()
}
fn default_material_roughness() -> String {
    "0.5".to_string()
}
fn default_material_specular() -> String {
    "0.0".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TextureDef {
    pub id: String,
    #[serde(default)]
    pub src: String,
    #[serde(default = "default_texture_kind")]
    pub kind: String,
    #[serde(default = "default_texture_scale")]
    pub scale: String,
    #[serde(default = "default_texture_strength")]
    pub strength: String,
    #[serde(default = "default_texture_contrast")]
    pub contrast: String,
    #[serde(default = "default_scene_zero")]
    pub seed: String,
    #[serde(default = "default_texture_brush_angle")]
    pub brush_angle: String,
    #[serde(default = "default_texture_bump_strength")]
    pub bump_strength: String,
    #[serde(default = "default_texture_relief")]
    pub relief: String,
}

fn default_texture_kind() -> String {
    "paper".to_string()
}

fn default_texture_scale() -> String {
    "42".to_string()
}

fn default_texture_strength() -> String {
    "0.25".to_string()
}

fn default_texture_contrast() -> String {
    "0.5".to_string()
}

fn default_texture_brush_angle() -> String {
    "-8".to_string()
}

fn default_texture_bump_strength() -> String {
    "0.35".to_string()
}

fn default_texture_relief() -> String {
    "0.45".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentNode {
    pub id: String,
    #[serde(default)]
    pub params: Vec<ComponentParamDef>,
    #[serde(default)]
    pub derived: Vec<ComponentDerivedDef>,
    #[serde(default)]
    pub slots: Vec<ComponentSlotDef>,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentParamDef {
    pub name: String,
    pub value_type: String,
    pub default: String,
    #[serde(default)]
    pub values: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentDerivedDef {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentSlotDef {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentParamValue {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentSlotValue {
    pub name: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FontDef {
    pub id: String,
    pub family: Option<String>,
    pub path: Option<String>,
    pub fallback: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilterDef {
    pub id: String,
    pub steps: Vec<FilterStepDef>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FilterStepDef {
    pub kind: String,
    pub radius: Option<String>,
    pub amount: Option<String>,
    pub scale: Option<String>,
    pub seed: Option<String>,
    pub preserve_interior: Option<String>,
    pub saturation: Option<String>,
    pub brightness: Option<String>,
    pub contrast: Option<String>,
    pub opacity: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PaletteNode {
    pub id: String,
    pub colors: Vec<PaletteColorDef>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PaletteColorDef {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PixelGridNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub pixel_size: String,
    pub palette: String,
    pub opacity: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrushDef {
    pub id: String,
    pub stroke: Option<String>,
    pub fill: Option<String>,
    pub stroke_width: Option<String>,
    pub opacity: Option<String>,
    pub line_cap: Option<String>,
    pub line_join: Option<String>,
    pub taper_start: Option<String>,
    pub taper_end: Option<String>,
    pub stroke_style: Option<String>,
    pub stroke_roughness: Option<String>,
    pub stroke_copies: Option<String>,
    pub stroke_texture: Option<String>,
    pub stroke_bristles: Option<String>,
    pub stroke_pressure: Option<String>,
    pub stroke_pressure_min: Option<String>,
    pub stroke_pressure_curve: Option<String>,
    pub blend: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GradientDef {
    Linear(LinearGradientDef),
    Radial(RadialGradientDef),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinearGradientDef {
    pub id: String,
    pub x1: String,
    pub y1: String,
    pub x2: String,
    pub y2: String,
    pub stops: Vec<GradientStop>,
    pub units: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RadialGradientDef {
    pub id: String,
    pub cx: String,
    pub cy: String,
    pub r: String,
    pub fx: Option<String>,
    pub fy: Option<String>,
    pub stops: Vec<GradientStop>,
    pub units: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GradientStop {
    pub offset: f32,
    pub color: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RectNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub width: String,
    pub height: String,
    pub radius: String,
    pub color: String,
    pub stroke: Option<String>,
    pub stroke_width: String,
    pub opacity: String,
    pub rotation: String,
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default = "default_scene_one")]
    pub texture_opacity: String,
    #[serde(default = "default_scene_one")]
    pub texture_scale: String,
    #[serde(default = "default_scene_zero")]
    pub texture_mask: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CircleNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub radius: String,
    pub color: String,
    pub stroke: Option<String>,
    pub stroke_width: String,
    pub opacity: String,
    #[serde(default = "default_scene_zero")]
    pub rotation: String,
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default = "default_scene_one")]
    pub texture_opacity: String,
    #[serde(default = "default_scene_one")]
    pub texture_scale: String,
    #[serde(default = "default_scene_zero")]
    pub texture_mask: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EllipseNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub radius_x: String,
    pub radius_y: String,
    pub color: String,
    pub stroke: Option<String>,
    pub stroke_width: String,
    pub opacity: String,
    #[serde(default = "default_scene_zero")]
    pub rotation: String,
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LineNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub x: String,
    #[serde(default = "default_scene_zero")]
    pub y: String,
    #[serde(default = "default_scene_zero")]
    pub rotation: String,
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    pub x1: String,
    pub y1: String,
    pub x2: String,
    pub y2: String,
    pub width: String,
    pub color: String,
    pub opacity: String,
    pub line_cap: String,
    pub taper_start: String,
    pub taper_end: String,
    #[serde(default = "default_stroke_style")]
    pub stroke_style: String,
    #[serde(default = "default_stroke_roughness")]
    pub stroke_roughness: String,
    #[serde(default = "default_stroke_copies")]
    pub stroke_copies: String,
    #[serde(default = "default_stroke_texture")]
    pub stroke_texture: String,
    #[serde(default = "default_stroke_bristles")]
    pub stroke_bristles: String,
    #[serde(default = "default_stroke_pressure")]
    pub stroke_pressure: String,
    #[serde(default = "default_stroke_pressure_min")]
    pub stroke_pressure_min: String,
    #[serde(default = "default_stroke_pressure_curve")]
    pub stroke_pressure_curve: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PolylineNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub x: String,
    #[serde(default = "default_scene_zero")]
    pub y: String,
    #[serde(default = "default_scene_zero")]
    pub rotation: String,
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    pub points: String,
    pub stroke: String,
    pub stroke_width: String,
    pub opacity: String,
    pub trim_start: String,
    pub trim_end: String,
    pub line_cap: String,
    pub line_join: String,
    pub taper_start: String,
    pub taper_end: String,
    #[serde(default = "default_stroke_style")]
    pub stroke_style: String,
    #[serde(default = "default_stroke_roughness")]
    pub stroke_roughness: String,
    #[serde(default = "default_stroke_copies")]
    pub stroke_copies: String,
    #[serde(default = "default_stroke_texture")]
    pub stroke_texture: String,
    #[serde(default = "default_stroke_bristles")]
    pub stroke_bristles: String,
    #[serde(default = "default_stroke_pressure")]
    pub stroke_pressure: String,
    #[serde(default = "default_stroke_pressure_min")]
    pub stroke_pressure_min: String,
    #[serde(default = "default_stroke_pressure_curve")]
    pub stroke_pressure_curve: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PathNode {
    pub id: Option<String>,
    pub brush: Option<String>,
    #[serde(default)]
    pub material: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub x: String,
    #[serde(default = "default_scene_zero")]
    pub y: String,
    #[serde(default = "default_scene_zero")]
    pub rotation: String,
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    pub d: String,
    pub stroke: String,
    pub fill: Option<String>,
    #[serde(default = "default_path_fill_rule")]
    pub fill_rule: String,
    #[serde(default = "default_path_boolean_op")]
    pub boolean_op: String,
    #[serde(default = "default_scene_zero")]
    pub offset_path: String,
    #[serde(default = "default_scene_zero")]
    pub round_corners: String,
    #[serde(default = "default_scene_false")]
    pub normalize: String,
    pub stroke_width: String,
    #[serde(default = "default_scene_one")]
    pub stroke_width_start: String,
    #[serde(default = "default_scene_one")]
    pub stroke_width_end: String,
    #[serde(default)]
    pub stroke_width_profile: String,
    pub opacity: String,
    pub trim_start: String,
    pub trim_end: String,
    pub line_cap: String,
    pub line_join: String,
    pub taper_start: String,
    pub taper_end: String,
    #[serde(default = "default_stroke_style")]
    pub stroke_style: String,
    #[serde(default = "default_stroke_roughness")]
    pub stroke_roughness: String,
    #[serde(default = "default_stroke_copies")]
    pub stroke_copies: String,
    #[serde(default = "default_stroke_texture")]
    pub stroke_texture: String,
    #[serde(default = "default_stroke_bristles")]
    pub stroke_bristles: String,
    #[serde(default = "default_stroke_pressure")]
    pub stroke_pressure: String,
    #[serde(default = "default_stroke_pressure_min")]
    pub stroke_pressure_min: String,
    #[serde(default = "default_stroke_pressure_curve")]
    pub stroke_pressure_curve: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default = "default_scene_one")]
    pub texture_opacity: String,
    #[serde(default = "default_scene_one")]
    pub texture_scale: String,
    #[serde(default = "default_scene_zero")]
    pub texture_mask: String,
}

fn default_path_fill_rule() -> String {
    "nonzero".to_string()
}

fn default_path_boolean_op() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceJawNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub width: String,
    pub height: String,
    pub cheek_width: String,
    pub chin_width: String,
    pub chin_sharpness: String,
    pub jaw_ease: String,
    pub scale: String,
    pub closed: String,
    pub stroke: String,
    pub fill: Option<String>,
    pub stroke_width: String,
    pub opacity: String,
    pub trim_start: String,
    pub trim_end: String,
    pub line_cap: String,
    pub line_join: String,
    pub taper_start: String,
    pub taper_end: String,
    #[serde(default = "default_stroke_style")]
    pub stroke_style: String,
    #[serde(default = "default_stroke_roughness")]
    pub stroke_roughness: String,
    #[serde(default = "default_stroke_copies")]
    pub stroke_copies: String,
    #[serde(default = "default_stroke_texture")]
    pub stroke_texture: String,
    #[serde(default = "default_stroke_bristles")]
    pub stroke_bristles: String,
    #[serde(default = "default_stroke_pressure")]
    pub stroke_pressure: String,
    #[serde(default = "default_stroke_pressure_min")]
    pub stroke_pressure_min: String,
    #[serde(default = "default_stroke_pressure_curve")]
    pub stroke_pressure_curve: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShadowNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub blur: String,
    pub color: String,
    pub opacity: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroupNode {
    pub id: Option<String>,
    pub brush: Option<String>,
    #[serde(default)]
    pub material: Option<String>,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    pub deform_grid: Option<String>,
    pub grid_from: Option<String>,
    pub grid_to: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub deform_amount: String,
    #[serde(default)]
    pub mask: Option<String>,
    #[serde(default)]
    pub mask_from: Option<String>,
    #[serde(default = "default_scene_mask_mode")]
    pub mask_mode: String,
    #[serde(default = "default_scene_zero")]
    pub mask_feather: String,
    #[serde(default = "default_scene_zero")]
    pub mask_expansion: String,
    #[serde(default)]
    pub effects: Vec<String>,
    /// Generic Process-backed effects scoped to this Group/CompositeGroup.
    #[serde(default)]
    pub process_effects: Vec<SceneEffectRef>,
    /// Present only when this Group was authored as `<CompositeGroup>`.
    #[serde(default)]
    pub composite: Option<CompositeGroupConfig>,
    pub opacity: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositeGroupConfig {
    #[serde(default = "default_scene_track_space")]
    pub space: String,
    #[serde(default)]
    pub composite_order: Option<i32>,
    #[serde(default)]
    pub depth: bool,
    #[serde(default = "default_scene_composite_format")]
    pub format: String,
    /// Optional id of the Camera3D selected by a discrete AnimationTarget.
    #[serde(default)]
    pub active_camera: Option<String>,
    /// Optional deterministic physics settings for this 3D island. Absence
    /// preserves the pre-physics Scene behavior exactly.
    #[serde(default)]
    pub physics: Option<ScenePhysicsConfig>,
    /// True-3D declarations are retained as typed compiler data. They lower to
    /// a 3D render island in the unified render-pass DAG.
    #[serde(default)]
    pub nodes_3d: Vec<Scene3DNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScenePhysicsConfig {
    #[serde(default = "default_scene_physics_gravity")]
    pub gravity: String,
    #[serde(default = "default_scene_physics_fixed_step")]
    pub fixed_step: String,
    #[serde(default = "default_scene_physics_iterations")]
    pub iterations: u32,
}

fn default_scene_physics_gravity() -> String {
    "[0,-9.81,0]".to_string()
}

fn default_scene_physics_fixed_step() -> String {
    "0.008333333".to_string()
}

fn default_scene_physics_iterations() -> u32 {
    4
}

fn default_scene_composite_format() -> String {
    "rgba8unorm".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Scene3DNode {
    Camera(SceneCamera3DNode),
    AtmosphereFog(SceneAtmosphereFogNode),
    EnvironmentLight(SceneEnvironmentLightNode),
    DirectionalLight(SceneDirectionalLightNode),
    PointLight(ScenePointLightNode),
    SpotLight(SceneSpotLightNode),
    RectAreaLight(SceneRectAreaLightNode),
    AmbientOcclusion(SceneAmbientOcclusionNode),
    ContactShadow(SceneContactShadowNode),
    ColorManagement(SceneColorManagementNode),
    VolumeRepeat(SceneVolumeRepeat3DNode),
    Model(SceneModel3DNode),
    RigidBody(crate::simulation::model::RigidBodyNode),
    Anchor(SceneAnchor3DNode),
    Debug(SceneEnvironmentDebugNode),
}

/// Deterministic world-space instances authored with the existing Repeat tag.
/// The template remains a regular Model so rain, snow, dust, and debris share
/// one lifecycle without introducing effect-specific scene tags.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneVolumeRepeat3DNode {
    pub id: Option<String>,
    pub count: u32,
    pub seed: u32,
    pub bounds_min: String,
    pub bounds_max: String,
    pub velocity: String,
    pub lifetime: String,
    pub phase: String,
    pub respawn: String,
    pub scale_range: String,
    pub template: SceneModel3DNode,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneEnvironmentDebugNode {
    #[serde(default = "default_scene_bool_true")]
    pub axes: bool,
    #[serde(default = "default_scene_bool_true")]
    pub bounds: bool,
    #[serde(default = "default_scene_bool_true")]
    pub surfaces: bool,
    #[serde(default = "default_scene_bool_true")]
    pub anchors: bool,
    #[serde(default = "default_scene_bool_true")]
    pub action_path: bool,
    #[serde(default = "default_scene_bool_true")]
    pub cameras: bool,
    #[serde(default)]
    pub colliders: bool,
    #[serde(default)]
    pub contacts: bool,
    #[serde(default)]
    pub sweep: bool,
    #[serde(default)]
    pub corrections: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneAnchor3DNode {
    pub id: String,
    pub relative_to: String,
    pub offset: String,
    pub space: String,
    /// Optional named GLB node used as the anchor origin. `offset` is applied
    /// after the node transform so portable environment markers do not need
    /// hard-coded world coordinates.
    #[serde(default)]
    pub node: Option<String>,
    /// Optional semantic surface owned by `relative_to`. When present, `uv`
    /// selects a stable point inside that surface's asset-space bounds.
    #[serde(default)]
    pub surface: Option<String>,
    #[serde(default)]
    pub uv: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneSurface3DNode {
    pub id: String,
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default = "default_scene_surface_kind")]
    pub kind: String,
    /// Explicit local-space fallback height. Named GLB nodes take precedence.
    #[serde(default = "default_scene_zero")]
    pub height: String,
    #[serde(default = "default_scene_surface_normal")]
    pub normal: String,
    /// `scene` preserves the original height-only behavior. `asset` means the
    /// measurements came from GLB inspection and must pass through the same
    /// normalization transform as the environment mesh.
    #[serde(default = "default_scene_surface_space")]
    pub space: String,
    #[serde(default)]
    pub centroid: Option<String>,
    #[serde(default)]
    pub bounds_min: Option<String>,
    #[serde(default)]
    pub bounds_max: Option<String>,
    /// Opts this semantic surface into Environment collision="surfaces".
    #[serde(default)]
    pub collision: bool,
    /// Optional collision representation. Ground defaults to plane; obstacle
    /// and wall surfaces default to box when this value is omitted.
    #[serde(default)]
    pub collider: Option<String>,
}

fn default_scene_surface_kind() -> String {
    "ground".to_string()
}

fn default_scene_surface_normal() -> String {
    "[0,1,0]".to_string()
}

fn default_scene_surface_space() -> String {
    "scene".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneCamera3DNode {
    pub id: Option<String>,
    pub position: String,
    pub target: String,
    pub fov: String,
    #[serde(default = "default_scene_surface_normal")]
    pub up: String,
    #[serde(default = "default_scene_zero")]
    pub roll: String,
    #[serde(default)]
    pub horizon_lock: bool,
    /// Optional optical depth of field. Omission preserves the legacy sharp render path.
    #[serde(default)]
    pub depth_of_field: Option<SceneDepthOfFieldNode>,
    /// Camera-local bone exclusions affect view passes without mutating the
    /// actor pose or its shadow-casting geometry.
    #[serde(default)]
    pub hidden_bones: Vec<SceneCameraHiddenBoneNode>,
}

/// Camera-owned optical controls resolved by the 3D renderer after the active shot is selected.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneDepthOfFieldNode {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub focus_target: Option<String>,
    #[serde(default)]
    pub focus_distance: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub focus_offset: String,
    #[serde(default = "default_scene_dof_focal_length")]
    pub focal_length: String,
    #[serde(default = "default_scene_dof_f_stop")]
    pub f_stop: String,
    #[serde(default = "default_scene_dof_max_blur")]
    pub max_blur: String,
}

fn default_scene_dof_focal_length() -> String {
    "50".to_string()
}

fn default_scene_dof_f_stop() -> String {
    "2.8".to_string()
}

fn default_scene_dof_max_blur() -> String {
    "10".to_string()
}

/// World-medium fog remains independent from cameras so shot cuts do not alter the atmosphere.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneAtmosphereFogNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_fog_mode")]
    pub mode: String,
    #[serde(default = "default_scene_fog_color")]
    pub color: String,
    #[serde(default = "default_scene_zero")]
    pub density: String,
    #[serde(default = "default_scene_zero")]
    pub start: String,
    #[serde(default = "default_scene_fog_end")]
    pub end: String,
    #[serde(default = "default_scene_zero")]
    pub base_height: String,
    #[serde(default = "default_scene_fog_height_falloff")]
    pub height_falloff: String,
    #[serde(default = "default_scene_zero")]
    pub scattering: String,
    #[serde(default)]
    pub affect_sky: bool,
    /// Optional world-space volume bounds. Omitting both preserves global fog.
    #[serde(default)]
    pub bounds_min: Option<String>,
    #[serde(default)]
    pub bounds_max: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub edge_feather: String,
}

fn default_scene_fog_mode() -> String {
    "linear".to_string()
}

fn default_scene_fog_color() -> String {
    "#FFFFFF".to_string()
}

fn default_scene_fog_end() -> String {
    "100".to_string()
}

fn default_scene_fog_height_falloff() -> String {
    "0.25".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneCameraHiddenBoneNode {
    pub model: String,
    pub bone: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneEnvironmentLightNode {
    pub id: Option<String>,
    pub asset: String,
    pub intensity: String,
    #[serde(default = "default_scene_environment_mapping")]
    pub mapping: String,
    #[serde(default = "default_scene_zero")]
    pub rotation_y: String,
    #[serde(default = "default_scene_bool_true")]
    pub visible: bool,
    #[serde(default = "default_scene_one")]
    pub background_intensity: String,
    #[serde(default = "default_scene_zero")]
    pub background_blur: String,
    #[serde(default = "default_scene_one")]
    pub diffuse_intensity: String,
    #[serde(default = "default_scene_one")]
    pub specular_intensity: String,
}

fn default_scene_environment_mapping() -> String {
    "equirectangular".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneDirectionalLightNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_light_direction")]
    pub direction: String,
    #[serde(default = "default_scene_light_color")]
    pub color: String,
    #[serde(default = "default_scene_one")]
    pub intensity: String,
    #[serde(default = "default_scene_bool_true")]
    pub cast_shadow: bool,
    #[serde(default = "default_scene_shadow_strength")]
    pub shadow_strength: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScenePointLightNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_zero_vec3")]
    pub position: String,
    #[serde(default = "default_scene_light_color")]
    pub color: String,
    #[serde(default = "default_scene_one")]
    pub intensity: String,
    #[serde(default = "default_scene_light_range")]
    pub range: String,
    #[serde(default)]
    pub cast_shadow: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneSpotLightNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_zero_vec3")]
    pub position: String,
    #[serde(default = "default_scene_light_direction")]
    pub direction: String,
    #[serde(default = "default_scene_light_color")]
    pub color: String,
    #[serde(default = "default_scene_one")]
    pub intensity: String,
    #[serde(default = "default_scene_light_range")]
    pub range: String,
    #[serde(default = "default_scene_spot_inner")]
    pub inner_cone: String,
    #[serde(default = "default_scene_spot_outer")]
    pub outer_cone: String,
    #[serde(default)]
    pub cast_shadow: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneRectAreaLightNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_zero_vec3")]
    pub position: String,
    #[serde(default = "default_scene_light_direction")]
    pub direction: String,
    #[serde(default = "default_scene_light_color")]
    pub color: String,
    #[serde(default = "default_scene_one")]
    pub intensity: String,
    #[serde(default = "default_scene_area_size")]
    pub width: String,
    #[serde(default = "default_scene_area_size")]
    pub height: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneAmbientOcclusionNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_one")]
    pub intensity: String,
    #[serde(default = "default_scene_ao_radius")]
    pub radius: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneContactShadowNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_shadow_strength")]
    pub intensity: String,
    #[serde(default = "default_scene_contact_distance")]
    pub distance: String,
    #[serde(default = "default_scene_contact_softness")]
    pub softness: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneColorManagementNode {
    pub id: Option<String>,
    #[serde(default = "default_scene_tone_mapping")]
    pub tone_mapping: String,
    #[serde(default = "default_scene_one")]
    pub exposure: String,
    #[serde(default = "default_scene_white_balance")]
    pub white_balance: String,
    #[serde(default = "default_scene_one")]
    pub contrast: String,
}

fn default_scene_zero_vec3() -> String {
    "[0,0,0]".to_string()
}
fn default_scene_light_direction() -> String {
    "[-0.4,-1,-0.35]".to_string()
}
fn default_scene_light_color() -> String {
    "#FFFFFF".to_string()
}
fn default_scene_light_range() -> String {
    "10".to_string()
}
fn default_scene_spot_inner() -> String {
    "22".to_string()
}
fn default_scene_spot_outer() -> String {
    "35".to_string()
}
fn default_scene_area_size() -> String {
    "2".to_string()
}
fn default_scene_shadow_strength() -> String {
    "0.8".to_string()
}
fn default_scene_ao_radius() -> String {
    "1".to_string()
}
fn default_scene_contact_distance() -> String {
    "0.25".to_string()
}
fn default_scene_contact_softness() -> String {
    "0.5".to_string()
}
fn default_scene_tone_mapping() -> String {
    "aces".to_string()
}
fn default_scene_white_balance() -> String {
    "6500".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneModel3DNode {
    pub id: Option<String>,
    pub asset: String,
    /// Private typed geometry used when Surface lowers through the Model path.
    #[serde(default)]
    pub primitive: Option<crate::dsl::PrimitiveAssetNode>,
    /// Optional canonical humanoid profile used to resolve Action bone ids.
    #[serde(default)]
    pub profile: Option<String>,
    /// Optional explicit rig id retained for compatibility with the 3D backend.
    #[serde(default)]
    pub rig: Option<String>,
    /// Optional retarget id retained for advanced host-generated graphs.
    #[serde(default)]
    pub retarget: Option<String>,
    pub position: String,
    #[serde(default)]
    pub position_x: Option<String>,
    #[serde(default)]
    pub position_y: Option<String>,
    #[serde(default)]
    pub position_z: Option<String>,
    pub rotation: String,
    #[serde(default)]
    pub rotation_x: Option<String>,
    #[serde(default)]
    pub rotation_y: Option<String>,
    #[serde(default)]
    pub rotation_z: Option<String>,
    pub scale: String,
    /// True when this Model was authored as `<Environment>`. It still lowers
    /// through the same GLB/PBR renderer as Model; this flag only carries
    /// environment semantics for authoring, grounding and inspection.
    #[serde(default)]
    pub environment: bool,
    #[serde(default)]
    pub r#static: bool,
    #[serde(default)]
    pub collision: Option<String>,
    /// `scene` opts this model into the enclosing Physics gravity vector.
    /// Missing/`none` keeps all existing authored placement semantics.
    #[serde(default)]
    pub gravity: Option<String>,
    /// Optional finite Surface used as the deterministic landing target.
    #[serde(default)]
    pub ground: Option<String>,
    /// Coordinate declaration for environment assets. Defaults match glTF.
    #[serde(default = "default_scene_environment_up")]
    pub up: String,
    #[serde(default = "default_scene_environment_forward")]
    pub forward: String,
    #[serde(default = "default_scene_one")]
    pub unit_scale: String,
    #[serde(default = "default_scene_environment_scale_mode")]
    pub scale_mode: String,
    #[serde(default = "default_scene_bool_true")]
    pub cast_shadow: bool,
    #[serde(default = "default_scene_bool_true")]
    pub receive_shadow: bool,
    #[serde(default)]
    pub surfaces: Vec<SceneSurface3DNode>,
    #[serde(default = "default_scene_model_exposure")]
    pub exposure: String,
    #[serde(default)]
    pub material_bindings: Vec<SceneMaterialBindingNode>,
    /// Embedded GLB animation playback. This remains optional for rig-only assets.
    #[serde(default)]
    pub play: Option<SceneModelPlayNode>,
    /// Additional clip layers blended after `play`. The singular field keeps
    /// existing serialized graphs and one-clip DSL behavior unchanged.
    #[serde(default)]
    pub plays: Vec<SceneModelPlayNode>,
    /// Per-frame editor overrides synthesized from bones.* AnimationTarget paths.
    #[serde(default)]
    pub bone_overrides: Vec<SceneModelBoneOverrideNode>,
}

fn default_scene_model_exposure() -> String {
    "1".to_string()
}

fn default_scene_environment_up() -> String {
    "+Y".to_string()
}

fn default_scene_environment_forward() -> String {
    "+Z".to_string()
}

fn default_scene_environment_scale_mode() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneModelPlayNode {
    #[serde(default)]
    pub clip: Option<String>,
    #[serde(default = "default_scene_bool_true")]
    pub r#loop: bool,
    #[serde(default = "default_scene_one")]
    pub speed: String,
    #[serde(default = "default_scene_one")]
    pub weight: String,
    #[serde(default = "default_scene_zero")]
    pub blend_in: String,
    #[serde(default = "default_scene_zero")]
    pub blend_out: String,
    #[serde(default)]
    pub mask: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneModelBoneOverrideNode {
    pub bone: String,
    #[serde(default)]
    pub x: Option<String>,
    #[serde(default)]
    pub y: Option<String>,
    #[serde(default)]
    pub z: Option<String>,
    #[serde(default)]
    pub rotation_x: Option<String>,
    #[serde(default)]
    pub rotation_y: Option<String>,
    #[serde(default)]
    pub rotation_z: Option<String>,
    #[serde(default)]
    pub scale: Option<String>,
}

fn default_scene_bool_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneMaterialBindingNode {
    pub material: String,
    #[serde(default)]
    pub definition: Option<String>,
    #[serde(default)]
    pub texture: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PuppetNode {
    pub id: Option<String>,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub capture: Option<String>,
    #[serde(default = "default_scene_puppet_solver")]
    pub solver: String,
    #[serde(default = "default_scene_puppet_mesh")]
    pub mesh: String,
    #[serde(default = "default_scene_puppet_density")]
    pub density: String,
    #[serde(default = "default_scene_puppet_bend")]
    pub bend: String,
    #[serde(default = "default_scene_zero")]
    pub stretch: String,
    #[serde(default = "default_scene_puppet_joint_softness")]
    pub joint_softness: String,
    #[serde(default = "default_scene_true")]
    pub preserve_volume: String,
    #[serde(default = "default_scene_false")]
    pub preserve_outside: String,
    #[serde(default = "default_scene_true")]
    pub preserve_length: String,
    #[serde(default = "default_scene_chain_stiffness")]
    pub stiffness: String,
    #[serde(default = "default_scene_chain_damping")]
    pub damping: String,
    #[serde(default = "default_scene_chain_drag")]
    pub drag: String,
    #[serde(default = "default_scene_chain_overlap")]
    pub overlap: String,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    #[serde(default = "default_scene_puppet_width")]
    pub width: String,
    #[serde(default = "default_scene_puppet_height")]
    pub height: String,
    #[serde(default = "default_scene_one")]
    pub amount: String,
    pub opacity: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PinNode {
    pub id: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub bind_to: Option<String>,
    #[serde(default)]
    pub vertex: Option<String>,
    /// Explicit parent id keeps chain topology stable when pins are reordered.
    #[serde(default)]
    pub parent: Option<String>,
    pub x: Option<String>,
    pub y: Option<String>,
    pub target_x: Option<String>,
    pub target_y: Option<String>,
    #[serde(default = "default_scene_pin_radius")]
    pub radius: String,
    #[serde(default = "default_scene_one")]
    pub strength: String,
    /// Local clockwise rotation in degrees. This is primarily used by
    /// role="bend" pins, but remains valid on any soft-solver pin.
    #[serde(default = "default_scene_zero")]
    pub rotation: String,
    /// Local uniform scale around the pin source.
    #[serde(default = "default_scene_one")]
    pub scale: String,
    #[serde(default = "default_scene_pin_falloff")]
    pub falloff: String,
    #[serde(default = "default_scene_false")]
    pub fixed: String,
}

/// A closed, non-rendering path that limits a bone rig to an exact limb area.
///
/// The runtime triangulates this path when no explicit MeshTopology is present.
/// `alpha_clip` keeps transparent source pixels transparent, while `hand_from`
/// names the pin where the rigid end-effector region begins.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LimbEnvelopeNode {
    pub id: Option<String>,
    pub d: String,
    #[serde(default = "default_scene_true")]
    pub alpha_clip: String,
    #[serde(default)]
    pub hand_from: Option<String>,
}

/// One exact influence area within a bone-rigged limb.
///
/// `role="anchor"` binds the area to the upper bone, `role="joint"` blends
/// both bones at the bend, and `role="control"` binds it to the lower bone.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LimbRegionNode {
    pub id: Option<String>,
    pub role: String,
    pub d: String,
    #[serde(default = "default_scene_true")]
    pub alpha_clip: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshTopologyNode {
    pub id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexNode {
    pub id: String,
    pub x: String,
    pub y: String,
    #[serde(default)]
    pub sample_x: Option<String>,
    #[serde(default)]
    pub sample_y: Option<String>,
    #[serde(default)]
    pub bone: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TriangleNode {
    pub id: Option<String>,
    pub a: String,
    pub b: String,
    pub c: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EdgeNode {
    pub id: Option<String>,
    pub a: String,
    pub b: String,
    #[serde(default = "default_scene_false")]
    pub boundary: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionNode {
    pub id: String,
    #[serde(default)]
    pub vertices: String,
    #[serde(default)]
    pub triangles: String,
    #[serde(default = "default_scene_one")]
    pub weight: String,
}

fn default_scene_puppet_mesh() -> String {
    "auto".to_string()
}

fn default_scene_puppet_solver() -> String {
    "soft".to_string()
}

fn default_scene_puppet_density() -> String {
    "medium".to_string()
}

fn default_scene_puppet_bend() -> String {
    "auto".to_string()
}

fn default_scene_puppet_joint_softness() -> String {
    "32".to_string()
}

fn default_scene_chain_stiffness() -> String {
    "0.72".to_string()
}

fn default_scene_chain_damping() -> String {
    "0.84".to_string()
}

fn default_scene_chain_drag() -> String {
    "0.18".to_string()
}

fn default_scene_chain_overlap() -> String {
    "0.12".to_string()
}

fn default_scene_puppet_width() -> String {
    "512".to_string()
}

fn default_scene_puppet_height() -> String {
    "512".to_string()
}

fn default_scene_pin_radius() -> String {
    "120".to_string()
}

fn default_scene_pin_falloff() -> String {
    "smooth".to_string()
}

fn default_scene_true() -> String {
    "true".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PartNode {
    pub id: Option<String>,
    pub label: Option<String>,
    pub role: Option<String>,
    pub attach_to: Option<String>,
    pub brush: Option<String>,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    pub opacity: String,
    pub anchor_x: String,
    pub anchor_y: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepeatNode {
    pub id: Option<String>,
    pub count: String,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    pub opacity: String,
    pub x_step: String,
    pub y_step: String,
    pub rotation_step: String,
    pub scale_step: String,
    pub opacity_step: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskNode {
    pub id: Option<String>,
    #[serde(default)]
    pub follow: Option<String>,
    pub shape: String,
    pub x: String,
    pub y: String,
    pub width: String,
    pub height: String,
    pub radius: String,
    pub d: Option<String>,
    #[serde(default = "default_scene_zero")]
    pub feather: String,
    pub opacity: String,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrecomposeNode {
    pub id: String,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    pub size: Option<(u32, u32)>,
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UseNode {
    pub id: Option<String>,
    pub ref_id: String,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    pub opacity: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
    #[serde(default)]
    pub params: Vec<ComponentParamValue>,
    #[serde(default)]
    pub slots: Vec<ComponentSlotValue>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneLayerNode {
    pub id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub is_3d: bool,
    pub x: String,
    pub y: String,
    #[serde(default = "default_scene_zero")]
    pub z: String,
    #[serde(default = "default_scene_zero")]
    pub rotation_x: String,
    #[serde(default = "default_scene_zero")]
    pub rotation_y: String,
    pub rotation: String,
    #[serde(default = "default_scene_perspective")]
    pub perspective: String,
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    #[serde(default)]
    pub z_depth: Option<String>,
    pub opacity: String,
    #[serde(default = "default_scene_blend")]
    pub blend: String,
    #[serde(default)]
    pub effect: Option<String>,
    #[serde(default)]
    pub process_effects: Vec<SceneEffectRef>,
    #[serde(default)]
    pub space: Option<String>,
    #[serde(default = "default_scene_source_time")]
    pub source_time: String,
    #[serde(default)]
    pub time_offset_ms: i64,
    #[serde(default = "default_scene_one")]
    pub playback_rate: String,
    #[serde(default = "default_scene_out_hold")]
    pub out: String,
    #[serde(default)]
    pub mask: Option<String>,
    #[serde(default)]
    pub mask_from: Option<String>,
    #[serde(default = "default_scene_mask_mode")]
    pub mask_mode: String,
    #[serde(default = "default_scene_zero")]
    pub mask_feather: String,
    #[serde(default = "default_scene_zero")]
    pub mask_expansion: String,
    #[serde(default)]
    pub matte: Option<String>,
    #[serde(default)]
    pub matte_from: Option<String>,
    #[serde(default = "default_scene_matte_mode")]
    pub matte_mode: String,
    #[serde(default = "default_scene_false")]
    pub invert_matte: String,
    #[serde(default)]
    pub children: Vec<SceneNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraNode {
    pub id: Option<String>,
    pub x: String,
    pub y: String,
    pub target_x: Option<String>,
    pub target_y: Option<String>,
    pub anchor_x: String,
    pub anchor_y: String,
    #[serde(default = "default_scene_zero")]
    pub offset_x: String,
    #[serde(default = "default_scene_zero")]
    pub offset_y: String,
    #[serde(default = "default_scene_zero")]
    pub shake_x: String,
    #[serde(default = "default_scene_zero")]
    pub shake_y: String,
    pub zoom: String,
    pub rotation: String,
    pub opacity: String,
    pub follow: Option<String>,
    #[serde(default)]
    pub dead_zone: Option<String>,
    pub viewport: Option<String>,
    pub world_bounds: Option<String>,
    pub children: Vec<SceneNode>,
}

fn default_scene_mask_mode() -> String {
    "alpha".to_string()
}

fn default_scene_perspective() -> String {
    "900".to_string()
}

fn default_scene_matte_mode() -> String {
    "alpha".to_string()
}

fn default_scene_false() -> String {
    "false".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterNode {
    pub id: Option<String>,
    #[serde(default)]
    pub src: Option<String>,
    #[serde(default)]
    pub rig: Option<String>,
    #[serde(default)]
    pub model_profile: Option<String>,
    pub x: String,
    pub y: String,
    pub rotation: String,
    pub scale: String,
    #[serde(default = "default_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_scene_zero")]
    pub transform_origin_y: String,
    pub opacity: String,
    pub children: Vec<SceneNode>,
}
