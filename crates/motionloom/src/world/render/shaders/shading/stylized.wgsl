// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/shading/stylized.wgsl

fn stylized_intensity(normal: vec3<f32>, light: vec3<f32>) -> f32 {
    return clamp(
        (dot(normal, light) + lighting.surface0.z) / (1.0 + lighting.surface0.z),
        0.0,
        1.0,
    );
}

fn shade_stylized(
    base_color: vec3<f32>,
    specular: vec3<f32>,
    radiance: vec3<f32>,
    intensity: f32,
    n_dot_l: f32,
) -> vec3<f32> {
    return (
        base_color / 3.14159265 * intensity
        + specular * lighting.surface1.y * n_dot_l
    ) * radiance;
}
