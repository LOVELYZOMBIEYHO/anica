// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/presets/filmic_bokeh_v1.wgsl

// A stable spiral disk produces smooth lens blur without directional banding.
// Signed circle-of-confusion keeps near and far defocus visually symmetric.
fn filmic_bokeh(uv: vec2<f32>, distance: f32) -> vec4<f32> {
    let signed_blur = clamp(
        (lighting.optics0.x - distance) * lighting.dof_style.z,
        -lighting.dof_style.w,
        lighting.dof_style.w,
    );
    let aspect = lighting.camera3.w;
    let scale = vec2<f32>(signed_blur, -signed_blur * aspect);
    let sample_count = 41u;
    var accumulated = textureSampleLevel(scene_color, scene_sampler, uv, 0.0);
    var total_weight = 1.0;
    for (var i = 1u; i < sample_count; i = i + 1u) {
        let fraction = (f32(i) - 0.5) / f32(sample_count - 1u);
        let radius = sqrt(fraction) * 0.4;
        let angle = f32(i) * 2.39996323;
        let offset = vec2<f32>(cos(angle), sin(angle)) * radius * scale;
        // A soft rim weight avoids a hard-edged disk while retaining highlights.
        let weight = 1.0 - 0.18 * fraction;
        accumulated += textureSampleLevel(
            scene_color,
            scene_sampler,
            clamp(uv + offset, vec2<f32>(0.0001), vec2<f32>(0.9999)),
            0.0,
        ) * weight;
        total_weight += weight;
    }
    return accumulated / total_weight;
}
