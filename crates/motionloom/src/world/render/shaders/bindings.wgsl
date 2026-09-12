// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/bindings.wgsl

struct Params {
    canvas: vec4<f32>,
    model: vec4<f32>,
    actor: vec4<f32>,
    actor_rotation: vec4<f32>,
    camera0: vec4<f32>,
    camera1: vec4<f32>,
    camera2: vec4<f32>,
    camera3: vec4<f32>,
    style: vec4<f32>,
    material0: vec4<f32>,
    material1: vec4<f32>,
    material2: vec4<f32>,
    material3: vec4<f32>,
    material4: vec4<f32>,
    material5: vec4<f32>,
    material6: vec4<f32>,
    material7: vec4<f32>,
    cel_material0: vec4<f32>,
    cel_material1: vec4<f32>,
    vegetation: vec4<f32>,
    hidden0: vec4<f32>,
    hidden1: vec4<f32>,
    hidden2: vec4<f32>,
    hidden3: vec4<f32>,
    hidden4: vec4<f32>,
    hidden5: vec4<f32>,
    hidden6: vec4<f32>,
    hidden7: vec4<f32>,
    previous_model: vec4<f32>,
    previous_actor: vec4<f32>,
    previous_actor_rotation: vec4<f32>,
    previous_vegetation: vec4<f32>,
    motion0: vec4<f32>,
};

struct Light {
    position_kind: vec4<f32>,
    direction_range: vec4<f32>,
    color_intensity: vec4<f32>,
    spot_area: vec4<f32>,
};

struct Lighting {
    environment0: vec4<f32>,
    environment1: vec4<f32>,
    environment2: vec4<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
    fog0: vec4<f32>,
    fog1: vec4<f32>,
    fog2: vec4<f32>,
    fog3: vec4<f32>,
    fog4: vec4<f32>,
    optics0: vec4<f32>,
    dof_style: vec4<f32>,
    render_compat: vec4<f32>,
    camera0: vec4<f32>,
    camera1: vec4<f32>,
    camera2: vec4<f32>,
    camera3: vec4<f32>,
    previous_camera0: vec4<f32>,
    previous_camera1: vec4<f32>,
    previous_camera2: vec4<f32>,
    previous_camera3: vec4<f32>,
    preview0: vec4<f32>,
    preview1: vec4<f32>,
    shadow0: vec4<f32>,
    shadow1: vec4<f32>,
    shadow2: vec4<f32>,
    shadow3: vec4<f32>,
    surface0: vec4<f32>,
    surface1: vec4<f32>,
    surface2: vec4<f32>,
    surface3: vec4<f32>,
    cel0: vec4<f32>,
    cel1: vec4<f32>,
    cel2: vec4<f32>,
    universal_color: vec4<f32>,
    universal_tone: vec4<f32>,
    universal_shadow: vec4<f32>,
    universal_highlight: vec4<f32>,
    lights: array<Light, 8>,
};

struct BoneMatrices {
    matrices: array<mat4x4<f32>>,
};

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<f32>,
    @location(3) weights: vec4<f32>,
    @location(4) uv: vec2<f32>,
    @location(5) color: vec4<f32>,
    @location(6) tangent: vec3<f32>,
    @location(7) bitangent: vec3<f32>,
    @location(8) outline_normal: vec3<f32>,
};

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) normal: vec3<f32>,
    @location(4) tangent: vec3<f32>,
    @location(5) bitangent: vec3<f32>,
    @location(6) hidden_weight: f32,
    @location(7) @interpolate(flat) instance_id: u32,
    @location(8) face_forward: vec3<f32>,
    @location(9) face_right: vec3<f32>,
    @location(10) previous_clip: vec4<f32>,
    @location(11) current_clip: vec4<f32>,
};

// Each invocation selects its own packed instance; fragment selection is flat.
@group(0) @binding(0) var<storage, read> instance_params: array<Params>;
var<private> params: Params;
@group(0) @binding(1) var<storage, read> bones: BoneMatrices;
@group(0) @binding(2) var actor_texture: texture_2d<f32>;
@group(0) @binding(3) var actor_sampler: sampler;
@group(0) @binding(4) var normal_texture: texture_2d<f32>;
@group(0) @binding(5) var metallic_roughness_texture: texture_2d<f32>;
@group(0) @binding(6) var emissive_texture: texture_2d<f32>;
@group(0) @binding(7) var cel_texture: texture_2d<f32>;
@group(0) @binding(8) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(0) var<uniform> lighting: Lighting;
@group(1) @binding(1) var environment_texture: texture_2d<f32>;
@group(1) @binding(2) var environment_sampler: sampler;
@group(1) @binding(3) var shadow_texture: texture_depth_2d;
@group(1) @binding(4) var shadow_sampler: sampler_comparison;
@group(2) @binding(0) var opaque_scene_texture: texture_2d<f32>;
@group(2) @binding(1) var opaque_scene_sampler: sampler;
