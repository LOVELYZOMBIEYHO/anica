// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/environment.wgsl

fn direction_to_environment_uv(direction: vec3<f32>) -> vec2<f32> {
    let rotated_x = direction.x * cos(lighting.environment0.y) - direction.z * sin(lighting.environment0.y);
    let rotated_z = direction.x * sin(lighting.environment0.y) + direction.z * cos(lighting.environment0.y);
    let normalized = normalize(vec3<f32>(rotated_x, direction.y, rotated_z));
    let u = 0.5 + atan2(normalized.z, normalized.x) / (2.0 * 3.14159265);
    let v = acos(clamp(normalized.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}
fn sample_environment(direction: vec3<f32>, lod: f32) -> vec3<f32> {
    return textureSampleLevel(
        environment_texture,
        environment_sampler,
        direction_to_environment_uv(direction),
        clamp(lod, 0.0, lighting.environment0.z)
    ).rgb;
}
