// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/fog.wgsl

// Return fog-path length, edge weight, and representative height for either
// the legacy global medium or an authored local box volume.
fn atmosphere_fog_ray_sample(
    ray_direction: vec3<f32>,
    ray_length: f32,
    fallback_height: f32,
) -> vec3<f32> {
    if (lighting.fog3.w < 0.5) {
        return vec3<f32>(ray_length, 1.0, fallback_height);
    }

    let epsilon = vec3<f32>(0.000001);
    let direction_sign = select(vec3<f32>(-1.0), vec3<f32>(1.0), ray_direction >= vec3<f32>(0.0));
    let safe_direction = select(direction_sign * epsilon, ray_direction, abs(ray_direction) >= epsilon);
    let lower = (lighting.fog3.xyz - lighting.camera0.xyz) / safe_direction;
    let upper = (lighting.fog4.xyz - lighting.camera0.xyz) / safe_direction;
    let slab_min = min(lower, upper);
    let slab_max = max(lower, upper);
    let entry = max(max(slab_min.x, max(slab_min.y, slab_min.z)), 0.0);
    let exit = min(min(slab_max.x, min(slab_max.y, slab_max.z)), ray_length);
    if (exit <= entry) {
        return vec3<f32>(0.0, 0.0, fallback_height);
    }

    let midpoint = lighting.camera0.xyz + ray_direction * ((entry + exit) * 0.5);
    let edge_distance3 = min(midpoint - lighting.fog3.xyz, lighting.fog4.xyz - midpoint);
    let edge_distance = max(min(edge_distance3.x, min(edge_distance3.y, edge_distance3.z)), 0.0);
    var edge_weight = 1.0;
    if (lighting.fog4.w > 0.000001) {
        edge_weight = smoothstep(0.0, lighting.fog4.w, edge_distance);
    }
    return vec3<f32>(exit - entry, edge_weight, midpoint.y);
}
fn atmosphere_fog_amount(world_position: vec3<f32>) -> f32 {
    if (lighting.fog2.w < 0.5) {
        return 0.0;
    }
    let camera_to_surface = world_position - lighting.camera0.xyz;
    let distance = length(camera_to_surface);
    if (distance <= 0.000001) {
        return 0.0;
    }
    let sample = atmosphere_fog_ray_sample(camera_to_surface / distance, distance, world_position.y);
    let fog_distance = max(sample.x - lighting.fog0.z, 0.0);
    if (lighting.fog0.x < 1.5) {
        return smoothstep(
            lighting.fog0.z,
            max(lighting.fog0.w, lighting.fog0.z + 0.001),
            sample.x,
        ) * sample.y;
    }
    let exponential = 1.0 - exp(-lighting.fog0.y * fog_distance);
    if (lighting.fog0.x < 2.5) {
        return clamp(exponential * sample.y, 0.0, 1.0);
    }
    let height_density = exp(-max(sample.z - lighting.fog1.w, 0.0) * lighting.fog2.x);
    return clamp(exponential * height_density * sample.y, 0.0, 1.0);
}
