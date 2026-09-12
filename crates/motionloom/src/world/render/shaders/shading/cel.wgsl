// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/shading/cel.wgsl

fn shade_cel(
    normal: vec3<f32>,
    light: vec3<f32>,
    radiance: vec3<f32>,
    base_color: vec3<f32>,
    specular: vec3<f32>,
    n_dot_l: f32,
    face_band: f32,
) -> vec3<f32> {
    let n = dot(normal, light) * 0.5 + 0.5;
    let feather = lighting.cel0.y;
    let steps = max(lighting.surface0.y, 2.0);
    var band = 0.0;
    for (var i = 1.0; i < steps; i += 1.0) {
        let threshold = clamp(
            lighting.cel0.x + (i / (steps - 1.0) - 0.5) * 0.5,
            0.0,
            1.0,
        );
        band += smoothstep(threshold - feather, threshold + feather, n) / (steps - 1.0);
    }
    if (face_band >= 0.0) {
        band = face_band;
    }
    let shadow_color = select(
        lighting.cel1.rgb,
        params.cel_material1.rgb,
        params.cel_material1.w > 0.5,
    );
    let shade = mix(pow(shadow_color, vec3<f32>(2.2)), vec3<f32>(1.0), band);
    return (
        base_color * shade / 3.14159265
        + specular * lighting.surface1.y * n_dot_l
    ) * radiance;
}
