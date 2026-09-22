// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/volumetric_composite.wgsl

struct FroxelParams {
    grid: vec4<f32>, camera0: vec4<f32>, camera1: vec4<f32>, camera2: vec4<f32>, camera3: vec4<f32>,
    medium0: vec4<f32>, medium1: vec4<f32>, atmosphere0: vec4<f32>, bounds_min: vec4<f32>, bounds_max: vec4<f32>,
    light0: vec4<f32>, light1: vec4<f32>, light2: vec4<f32>, light3: vec4<f32>,
    caustics0: vec4<f32>, caustics1: vec4<f32>,
    shadow0: vec4<f32>, shadow1: vec4<f32>, shadow2: vec4<f32>, shadow3: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: FroxelParams;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var integrated: texture_3d<f32>;
@group(0) @binding(4) var scene_depth: texture_depth_2d;
@group(0) @binding(5) var injection: texture_3d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_texture));
    let uv = position.xy / dimensions;
    let scene = textureSampleLevel(scene_texture, scene_sampler, uv, 0.0);
    let reverse_depth = textureLoad(scene_depth, vec2<u32>(position.xy), 0);
    if (reverse_depth <= 0.000001 && params.atmosphere0.w < 0.5) {
        return scene;
    }
    var distance = params.bounds_max.w;
    if (reverse_depth > 0.000001) {
        distance = clamp(params.camera0.w / reverse_depth, params.camera0.w, params.bounds_max.w);
    }
    let z = clamp(log(distance / params.camera0.w) /
        log(params.bounds_max.w / params.camera0.w), 0.0, 0.9999);
    let volume = textureSampleLevel(integrated, scene_sampler, vec3<f32>(uv, z), 0.0);
    let local = textureSampleLevel(injection, scene_sampler, vec3<f32>(uv, z), 0.0);
    let debug_view = u32(floor(params.light0.w * 0.5) + 0.5);
    if (debug_view == 1u || debug_view == 2u || debug_view == 3u || debug_view == 6u) {
        return vec4<f32>(local.rgb, 1.0);
    }
    let optical_depth = vec3<f32>(volume.a);
    if (debug_view == 4u) { return vec4<f32>(optical_depth, 1.0); }
    if (debug_view == 5u) { return vec4<f32>(exp(-optical_depth), 1.0); }
    let transmittance = exp(-optical_depth);
    return vec4<f32>(scene.rgb * transmittance + volume.rgb, scene.a);
}
