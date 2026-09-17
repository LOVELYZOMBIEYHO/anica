// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/volumetric_inject.wgsl

struct FroxelParams {
    grid: vec4<f32>,
    camera0: vec4<f32>,
    camera1: vec4<f32>,
    camera2: vec4<f32>,
    camera3: vec4<f32>,
    medium0: vec4<f32>,
    medium1: vec4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    light0: vec4<f32>,
    light1: vec4<f32>,
    light2: vec4<f32>,
    light3: vec4<f32>,
    caustics0: vec4<f32>,
    caustics1: vec4<f32>,
    shadow0: vec4<f32>,
    shadow1: vec4<f32>,
    shadow2: vec4<f32>,
    shadow3: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: FroxelParams;
@group(0) @binding(1) var injection: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var shadow_texture: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

fn slice_distance(slice: f32) -> f32 {
    let t = slice / max(params.grid.z, 1.0);
    return params.camera0.w * pow(params.bounds_max.w / params.camera0.w, t);
}

fn caustic_pattern(world: vec3<f32>) -> f32 {
    let p = world.xz / max(params.caustics0.y, 0.0001);
    let time = params.grid.w * params.caustics0.z;
    let a = sin(p.x * 1.31 + p.y * 0.77 + time);
    let b = sin(p.x * -0.63 + p.y * 1.67 - time * 1.23);
    let c = sin(p.x * 1.91 - p.y * 0.41 + time * 0.71);
    return pow(clamp((a + b + c) * 0.1667 + 0.5, 0.0, 1.0), 5.0);
}

fn shadow_visibility(world: vec3<f32>) -> f32 {
    if (fract(params.light0.w * 0.5) * 2.0 < 0.5) { return 1.0; }
    let relative = world - params.shadow3.xyz;
    let coordinate = vec3<f32>(
        dot(relative, params.shadow0.xyz) / params.shadow0.w * 0.5 + 0.5,
        0.5 - dot(relative, params.shadow1.xyz) / params.shadow1.w * 0.5,
        dot(relative, params.shadow2.xyz) / params.shadow2.w + 0.5
    );
    if (any(coordinate.xy < vec2<f32>(0.0)) || any(coordinate.xy > vec2<f32>(1.0)) ||
        coordinate.z < 0.0 || coordinate.z > 1.0) { return 1.0; }
    return textureSampleCompareLevel(
        shadow_texture,
        shadow_sampler,
        coordinate.xy,
        coordinate.z - params.shadow3.w
    );
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(injection);
    if (any(id >= dimensions)) { return; }
    let pixel = (vec2<f32>(id.xy) + 0.5) * params.grid.xy;
    let screen = vec2<f32>(pixel.x - params.camera3.w * params.camera2.w * 0.5,
                           params.camera2.w * 0.5 - pixel.y);
    let ray = normalize(params.camera3.xyz +
        params.camera1.xyz * (screen.x / params.camera2.w) +
        params.camera2.xyz * (screen.y / params.camera2.w));
    let distance = slice_distance(f32(id.z) + 0.5);
    let world = params.camera0.xyz + ray * distance;
    if (params.bounds_min.w > 0.5 &&
        (any(world < params.bounds_min.xyz) || any(world > params.bounds_max.xyz))) {
        textureStore(injection, id, vec4<f32>(0.0));
        return;
    }
    var light_direction = normalize(-params.light0.xyz);
    var light_attenuation = 1.0;
    if (params.light3.z > 1.5) {
        let delta = params.light2.xyz - world;
        let light_distance = max(length(delta), 0.001);
        light_direction = delta / light_distance;
        let normalized_distance = light_distance / max(params.light2.w, 0.001);
        light_attenuation = pow(clamp(1.0 - pow(normalized_distance, 4.0), 0.0, 1.0), 2.0) /
            max(light_distance * light_distance, 0.25);
        let cone = dot(normalize(-light_direction), normalize(params.light0.xyz));
        light_attenuation *= smoothstep(params.light3.y, params.light3.x, cone);
    }
    let view_to_light = dot(-ray, light_direction);
    let g = params.medium1.w;
    let phase = (1.0 - g * g) /
        max(4.0 * 3.14159265 * pow(1.0 + g * g - 2.0 * g * view_to_light, 1.5), 0.0001);
    let visibility = shadow_visibility(world);
    var radiance = params.light1.rgb * params.light1.w * phase * visibility * light_attenuation;
    var caustic_radiance = vec3<f32>(0.0);
    if (params.caustics1.w > 0.5) {
        let depth_attenuation = exp(-max(0.0, params.bounds_max.y - world.y) * params.caustics0.w);
        caustic_radiance = params.caustics1.rgb * caustic_pattern(world) * params.caustics0.x * depth_attenuation;
        radiance += caustic_radiance;
    }
    let debug_view = u32(floor(params.light0.w * 0.5) + 0.5);
    if (debug_view == 1u) { radiance = vec3<f32>(params.medium0.w); }
    if (debug_view == 2u) { radiance = vec3<f32>(visibility); }
    if (debug_view == 6u) { radiance = caustic_radiance; }
    textureStore(injection, id, vec4<f32>(radiance * params.medium1.rgb, params.medium0.w));
}
