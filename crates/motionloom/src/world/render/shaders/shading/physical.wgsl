// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/shading/physical.wgsl

fn shade_physical(
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    radiance: vec3<f32>,
    n_dot_l: f32,
) -> vec3<f32> {
    return (diffuse + specular) * radiance * n_dot_l;
}

fn shade_wrapped_physical(
    normal: vec3<f32>,
    light: vec3<f32>,
    diffuse: vec3<f32>,
    specular: vec3<f32>,
    radiance: vec3<f32>,
    n_dot_l: f32,
) -> vec3<f32> {
    let wrapped = clamp(
        (dot(normal, light) + lighting.surface0.z) / (1.0 + lighting.surface0.z),
        0.0,
        1.0,
    );
    return (diffuse * wrapped + specular * lighting.surface1.y * n_dot_l) * radiance;
}
