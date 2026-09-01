// =========================================
// =========================================
// crates/motionloom/src/world/model.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldGraph {
    pub id: Option<String>,
    pub version: Option<String>,
    pub fps: f32,
    pub duration_ms: u64,
    pub duration_explicit: bool,
    pub size: (u32, u32),
    pub render_size: Option<(u32, u32)>,
    pub model_profiles: Vec<WorldModelProfile>,
    pub worlds: Vec<WorldNode>,
    pub retargets: Vec<WorldRetarget>,
    pub actions: Vec<WorldAction>,
    pub apply_actions: Vec<WorldApplyAction>,
    /// Animation-only GLB sources used by Scene ApplyAction nodes.
    #[serde(default)]
    pub animation_assets: Vec<WorldAnimationAsset>,
    /// Cross-actor constraints lowered from the public Scene DSL.
    #[serde(default)]
    pub constraints: Vec<WorldConstraint>,
    /// Frame-local lighting lowered from the public Scene 3D DSL.
    #[serde(default)]
    pub lighting: WorldLighting,
    pub present: WorldPresent,
}

/// Internal lighting payload shared by Scene3DRenderer and the retained GPU renderer.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldLighting {
    #[serde(default)]
    pub environment: Option<WorldEnvironmentLighting>,
    #[serde(default)]
    pub lights: Vec<WorldLight>,
    #[serde(default = "default_world_ao_intensity")]
    pub ao_intensity: f32,
    #[serde(default = "default_world_ao_radius")]
    pub ao_radius: f32,
    #[serde(default)]
    pub contact_shadow_intensity: f32,
    #[serde(default = "default_world_contact_distance")]
    pub contact_shadow_distance: f32,
    #[serde(default = "default_world_contact_softness")]
    pub contact_shadow_softness: f32,
    #[serde(default)]
    pub color_management: WorldColorManagement,
    /// Optional world-space atmospheric fog. None keeps the previous shader result.
    #[serde(default)]
    pub atmosphere_fog: Option<WorldAtmosphereFog>,
}

impl Default for WorldLighting {
    fn default() -> Self {
        Self {
            environment: None,
            lights: Vec::new(),
            ao_intensity: default_world_ao_intensity(),
            ao_radius: default_world_ao_radius(),
            contact_shadow_intensity: 0.0,
            contact_shadow_distance: default_world_contact_distance(),
            contact_shadow_softness: default_world_contact_softness(),
            color_management: WorldColorManagement::default(),
            atmosphere_fog: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldAtmosphereFog {
    pub mode: String,
    pub color: [f32; 3],
    pub density: f32,
    pub start: f32,
    pub end: f32,
    pub base_height: f32,
    pub height_falloff: f32,
    pub scattering: f32,
    pub affect_sky: bool,
    /// Finite world-space bounds enable local fog; None retains global fog.
    #[serde(default)]
    pub bounds_min: Option<[f32; 3]>,
    #[serde(default)]
    pub bounds_max: Option<[f32; 3]>,
    #[serde(default)]
    pub edge_feather: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldEnvironmentLighting {
    pub src: String,
    pub mapping: String,
    pub intensity: f32,
    pub rotation_y_degrees: f32,
    pub visible: bool,
    pub background_intensity: f32,
    pub background_blur: f32,
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldLightKind {
    Directional,
    Point,
    Spot,
    RectArea,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldLight {
    pub id: Option<String>,
    pub kind: WorldLightKind,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_degrees: f32,
    pub outer_cone_degrees: f32,
    pub width: f32,
    pub height: f32,
    pub cast_shadow: bool,
    pub shadow_strength: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldColorManagement {
    pub tone_mapping: String,
    pub exposure: f32,
    pub white_balance_kelvin: f32,
    pub contrast: f32,
}

impl Default for WorldColorManagement {
    fn default() -> Self {
        Self {
            tone_mapping: "aces".to_string(),
            exposure: 1.0,
            white_balance_kelvin: 6500.0,
            contrast: 1.0,
        }
    }
}

fn default_world_ao_intensity() -> f32 {
    0.18
}
fn default_world_ao_radius() -> f32 {
    1.0
}
fn default_world_contact_distance() -> f32 {
    0.25
}
fn default_world_contact_softness() -> f32 {
    0.5
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldAnimationAsset {
    pub id: String,
    pub src: String,
    pub profile: String,
    pub clip: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldConstraint {
    pub constraint_type: String,
    pub source: String,
    pub target: String,
    /// Internal Scene bridge target used for deterministic environment contact.
    /// Public DSL constraints continue to resolve actor.bone targets unchanged.
    #[serde(default)]
    pub target_point: Option<[f32; 3]>,
    pub at_ms: u64,
    pub duration_ms: u64,
    pub solver: String,
    pub weight: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldNode {
    pub id: String,
    pub background: Option<WorldBackground>,
    pub camera: WorldCamera,
    pub actors: Vec<WorldActor>,
    pub directional_characters: Vec<WorldDirectionalCharacter>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorldBackgroundFit {
    Cover,
    Contain,
    Stretch,
}

impl Default for WorldBackgroundFit {
    fn default() -> Self {
        Self::Cover
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldBackground {
    pub id: Option<String>,
    pub src: Option<String>,
    pub fit: WorldBackgroundFit,
    pub color: String,
    pub opacity: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorldCameraControl {
    Orbit,
    Free,
}

impl Default for WorldCameraControl {
    fn default() -> Self {
        Self::Orbit
    }
}

pub type WorldCameraMode = WorldCameraControl;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorldCameraProjection {
    Perspective,
    Orthographic,
}

impl Default for WorldCameraProjection {
    fn default() -> Self {
        Self::Perspective
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldCamera {
    pub id: Option<String>,
    #[serde(default, alias = "mode")]
    pub control: WorldCameraControl,
    pub projection: WorldCameraProjection,
    pub target: Option<String>,
    pub x: String,
    pub y: String,
    pub z: String,
    pub target_x: String,
    pub target_y: String,
    pub target_z: String,
    pub yaw: String,
    pub pitch: String,
    pub roll: String,
    #[serde(default = "default_world_camera_up_x")]
    pub up_x: String,
    #[serde(default = "default_world_camera_up_y")]
    pub up_y: String,
    #[serde(default = "default_world_camera_up_z")]
    pub up_z: String,
    pub distance: String,
    pub zoom: String,
    pub fov: String,
    pub orthographic_scale: Option<String>,
    /// Optional camera optics. None retains the no-post-process fast path.
    #[serde(default)]
    pub depth_of_field: Option<WorldDepthOfField>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldDepthOfField {
    pub focus_distance: String,
    pub focal_length_mm: String,
    pub f_stop: String,
    pub max_blur_px: String,
}

impl Default for WorldCamera {
    fn default() -> Self {
        Self {
            id: Some("camera".to_string()),
            control: WorldCameraControl::Orbit,
            projection: WorldCameraProjection::Perspective,
            target: None,
            x: "0".to_string(),
            y: "0".to_string(),
            z: "0".to_string(),
            target_x: "0".to_string(),
            target_y: "1.0".to_string(),
            target_z: "0".to_string(),
            yaw: "0".to_string(),
            pitch: "0".to_string(),
            roll: "0".to_string(),
            up_x: "0".to_string(),
            up_y: "1".to_string(),
            up_z: "0".to_string(),
            distance: "3.2".to_string(),
            zoom: "1".to_string(),
            fov: "35".to_string(),
            orthographic_scale: None,
            depth_of_field: None,
        }
    }
}

fn default_world_camera_up_x() -> String {
    "0".to_string()
}

fn default_world_camera_up_y() -> String {
    "1".to_string()
}

fn default_world_camera_up_z() -> String {
    "0".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldActor {
    pub id: String,
    pub model: String,
    /// Typed procedural geometry bypasses external asset resolution.
    #[serde(default)]
    pub primitive: Option<crate::dsl::PrimitiveAssetNode>,
    /// Heightfield terrain uses the same retained PBR mesh path as primitives.
    #[serde(default)]
    pub terrain: Option<crate::dsl::TerrainAssetNode>,
    /// Procedural vegetation retains its generation and wind metadata beside
    /// the actor while sharing the normal PBR mesh pipeline.
    #[serde(default)]
    pub vegetation: Option<crate::dsl::VegetationAssetNode>,
    pub path_style: WorldPathStyle,
    pub hide_meshes: Vec<String>,
    pub hide_materials: Vec<String>,
    /// Canonical bones hidden only from the active camera's view passes.
    /// Shadow rendering intentionally ignores this list.
    #[serde(default)]
    pub camera_hidden_bones: Vec<String>,
    pub profile: Option<String>,
    pub rig: Option<String>,
    pub retarget: Option<String>,
    pub x: String,
    pub y: String,
    pub z: String,
    pub yaw: String,
    pub pitch: String,
    pub roll: String,
    /// Runtime-owned rotations can bypass Euler decomposition and reach the
    /// GPU as the same normalized quaternion used by physics.
    #[serde(default)]
    pub rotation_quaternion: Option<[f32; 4]>,
    pub scale: String,
    /// `none` preserves the authored glTF origin and units. The legacy
    /// `normalize_height` mode bottom-centres the mesh and treats `scale` as
    /// a target height.
    #[serde(default = "default_world_actor_scale_mode")]
    pub scale_mode: String,
    pub opacity: String,
    pub material: Option<WorldMaterial>,
    pub play: Option<WorldPlay>,
    #[serde(default)]
    pub plays: Vec<WorldPlay>,
}

fn default_world_actor_scale_mode() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldDirectionalCharacter {
    pub id: String,
    pub sheet: Option<String>,
    pub path_style: WorldPathStyle,
    pub x: String,
    pub y: String,
    pub scale: String,
    pub yaw: String,
    pub opacity: String,
    pub play_sprite: Option<WorldSpritePlayback>,
    pub directions: Vec<WorldDirectionFrame>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldSpritePlayback {
    pub fps: String,
    pub r#loop: bool,
    pub frames: u32,
    pub columns: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub start: u32,
    pub margin_x: u32,
    pub margin_y: u32,
    pub spacing_x: u32,
    pub spacing_y: u32,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldDirectionFrame {
    pub name: Option<String>,
    pub angle: Option<f32>,
    pub camera_pitch: Option<f32>,
    pub image: Option<String>,
    pub rect: Option<(u32, u32, u32, u32)>,
    pub anchor: Option<(f32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorldPathStyle {
    Relative,
    Absolute,
}

impl Default for WorldPathStyle {
    fn default() -> Self {
        Self::Relative
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorldMaterialStyle {
    Toon,
    Pbr,
    Unlit,
}

impl Default for WorldMaterialStyle {
    fn default() -> Self {
        Self::Toon
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldMaterial {
    pub style: WorldMaterialStyle,
    pub outline: bool,
    pub outline_width: String,
    #[serde(default = "default_world_material_exposure")]
    pub exposure: String,
}

fn default_world_material_exposure() -> String {
    "1".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldPlay {
    pub clip: Option<String>,
    pub r#loop: bool,
    pub speed: String,
    #[serde(default = "default_world_action_speed")]
    pub weight: String,
    #[serde(default = "default_world_action_zero")]
    pub blend_in: String,
    #[serde(default = "default_world_action_zero")]
    pub blend_out: String,
    #[serde(default)]
    pub mask: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WorldRetarget {
    pub id: String,
    pub actor: Option<String>,
    pub preset: String,
    pub maps: Vec<WorldRetargetMap>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WorldRetargetMap {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldModelProfile {
    pub id: String,
    pub model: String,
    pub preset: String,
    pub retarget: Option<WorldProfileRetarget>,
    pub bone_axis_map: Option<WorldBoneAxisMap>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldProfileRetarget {
    pub preset: String,
    pub maps: Vec<WorldRetargetMap>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldBoneAxisMap {
    pub axes: Vec<WorldBoneAxis>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldBoneAxis {
    pub bone: String,
    pub forward: Option<String>,
    pub side: Option<String>,
    pub twist: Option<String>,
    pub bend: Option<String>,
    pub turn: Option<String>,
    pub rest_forward: Option<String>,
    pub rest_side: Option<String>,
    pub rest_twist: Option<String>,
    pub rest_bend: Option<String>,
    pub rest_turn: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldAction {
    pub id: String,
    pub skeleton: String,
    pub intent: Option<String>,
    pub duration_ms: u64,
    pub poses: Vec<WorldActionPose>,
    #[serde(default)]
    pub iks: Vec<WorldActionIk>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldActionIk {
    pub root: String,
    pub mid: String,
    pub end: String,
    pub target_x: String,
    pub target_y: String,
    pub target_z: String,
    pub pole_x: Option<String>,
    pub pole_y: Option<String>,
    pub pole_z: Option<String>,
    pub plane: String,
    pub bend: String,
    pub weight: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldActionPose {
    pub t: f32,
    pub label: Option<String>,
    pub bones: Vec<WorldActionBone>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldActionBone {
    pub id: String,
    pub x: Option<String>,
    pub y: Option<String>,
    pub z: Option<String>,
    pub rotation: Option<String>,
    pub rotation_x: Option<String>,
    pub rotation_y: Option<String>,
    pub rotation_z: Option<String>,
    pub forward: Option<String>,
    pub side: Option<String>,
    pub twist: Option<String>,
    pub bend: Option<String>,
    pub turn: Option<String>,
    pub scale: Option<String>,
    pub opacity: Option<String>,
    /// Interpolation from this key to the next key. Omission preserves the
    /// original linear Action behavior.
    #[serde(default)]
    pub interpolation: Option<String>,
    #[serde(default)]
    pub in_tangent: Option<String>,
    #[serde(default)]
    pub out_tangent: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct WorldApplyAction {
    pub target: String,
    pub action: String,
    pub at_ms: u64,
    pub r#loop: bool,
    pub weight: String,
    #[serde(default = "default_world_action_speed")]
    pub speed: String,
    #[serde(default = "default_world_action_zero")]
    pub blend_in: String,
    #[serde(default = "default_world_action_zero")]
    pub blend_out: String,
    #[serde(default = "default_world_action_mode")]
    pub mode: String,
    #[serde(default)]
    pub mask: Vec<String>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub root_motion: Option<String>,
    #[serde(default)]
    pub destination: Option<String>,
    #[serde(default)]
    pub face: Option<String>,
    #[serde(default)]
    pub sync_group: Option<String>,
    #[serde(default)]
    pub sync_marker: Option<String>,
}

fn default_world_action_speed() -> String {
    "1".to_string()
}

fn default_world_action_zero() -> String {
    "0".to_string()
}

fn default_world_action_mode() -> String {
    "override".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct WorldPresent {
    pub from: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldTime {
    pub frame: u32,
    pub fps: f32,
    pub duration_ms: u64,
}

impl WorldTime {
    pub fn time_sec(self) -> f32 {
        if self.fps <= f32::EPSILON {
            0.0
        } else {
            self.frame as f32 / self.fps
        }
    }

    pub fn time_norm(self) -> f32 {
        let duration_sec = self.duration_ms as f32 / 1000.0;
        if duration_sec <= f32::EPSILON {
            0.0
        } else {
            (self.time_sec() / duration_sec).clamp(0.0, 1.0)
        }
    }
}

impl WorldGraph {
    pub fn presented_world(&self) -> Option<&WorldNode> {
        self.worlds
            .iter()
            .find(|world| world.id == self.present.from)
    }

    pub fn output_size(&self) -> (u32, u32) {
        self.render_size.unwrap_or(self.size)
    }
}
