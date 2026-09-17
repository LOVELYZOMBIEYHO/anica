// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/volumetric_integrate.wgsl

struct FroxelParams {
    grid: vec4<f32>, camera0: vec4<f32>, camera1: vec4<f32>, camera2: vec4<f32>, camera3: vec4<f32>,
    medium0: vec4<f32>, medium1: vec4<f32>, bounds_min: vec4<f32>, bounds_max: vec4<f32>,
    light0: vec4<f32>, light1: vec4<f32>, light2: vec4<f32>, light3: vec4<f32>,
    caustics0: vec4<f32>, caustics1: vec4<f32>,
    shadow0: vec4<f32>, shadow1: vec4<f32>, shadow2: vec4<f32>, shadow3: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: FroxelParams;
@group(0) @binding(1) var injection: texture_3d<f32>;
@group(0) @binding(2) var integrated: texture_storage_3d<rgba16float, write>;

fn slice_distance(slice: f32) -> f32 {
    let t = slice / max(params.grid.z, 1.0);
    return params.camera0.w * pow(params.bounds_max.w / params.camera0.w, t);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dimensions = textureDimensions(integrated);
    if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
    var transmittance = vec3<f32>(1.0);
    var accumulated = vec3<f32>(0.0);
    var density_integral = 0.0;
    for (var z = 0u; z < dimensions.z; z = z + 1u) {
        let cell = textureLoad(injection, vec3<u32>(id.xy, z), 0);
        let step_length = slice_distance(f32(z) + 1.0) - slice_distance(f32(z));
        let extinction = (params.medium0.rgb + params.medium1.rgb) * cell.a;
        let step_transmittance = exp(-extinction * step_length);
        accumulated += transmittance * cell.rgb * step_length;
        transmittance *= step_transmittance;
        density_integral += cell.a * step_length;
        textureStore(integrated, vec3<u32>(id.xy, z), vec4<f32>(accumulated, density_integral));
    }
}
