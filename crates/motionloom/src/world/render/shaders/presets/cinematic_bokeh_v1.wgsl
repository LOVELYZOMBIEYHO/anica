// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/presets/cinematic_bokeh_v1.wgsl

// Deterministic disk integration in linear HDR. Separate near coverage allows
// defocused foreground to spread over sharp pixels without blurring the subject.
fn cinematic_bokeh(uv: vec2<f32>, center: vec4<f32>, distance: f32, radius: f32) -> vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_color));
    let count = u32(lighting.dof_style.y);
    let focus = lighting.optics0.x;
    let maximum = lighting.optics0.w;
    var far_color = vec4<f32>(0.0);
    var far_weight = 0.0;
    var near_color = vec4<f32>(0.0);
    var near_weight = 0.0;
    var coverage = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let r = sqrt((f32(i) + 0.5) / f32(count));
        let angle = f32(i) * 2.39996323;
        let disk = vec2<f32>(cos(angle), sin(angle)) * r;

        // Background gather rejects nearer occluders and sharp surfaces.
        let far_uv = clamp(uv + disk * radius / dimensions, 0.5 / dimensions, 1.0 - 0.5 / dimensions);
        let far_distance = view_distance(textureLoad(scene_depth, vec2<i32>(far_uv * dimensions), 0));
        let far_coc = circle_of_confusion(far_distance, dimensions.y);
        let behind = smoothstep(distance * 0.98, distance * 1.01, far_distance);
        let support = smoothstep(r * radius - 1.0, r * radius + 1.0, far_coc);
        let fw = behind * support;
        far_color += textureSampleLevel(scene_color, scene_sampler, far_uv, 0.0) * fw;
        far_weight += fw;

        // Search the full allowed disk even at a sharp center. Coverage is
        // normalized by each source footprint instead of darkening silhouettes.
        let near_uv = clamp(uv + disk * maximum / dimensions, 0.5 / dimensions, 1.0 - 0.5 / dimensions);
        let near_distance = view_distance(textureLoad(scene_depth, vec2<i32>(near_uv * dimensions), 0));
        let near_coc = circle_of_confusion(near_distance, dimensions.y);
        let is_near = 1.0 - smoothstep(focus * 0.98, focus, near_distance);
        let occludes = 1.0 - smoothstep(distance, distance * 1.03, near_distance);
        let footprint = 1.0 - smoothstep(near_coc - 0.75, near_coc + 0.75, r * maximum);
        let nw = is_near * occludes * footprint * smoothstep(0.5, 2.0, near_coc)
            * maximum * maximum / max(near_coc * near_coc, 1.0);
        near_color += textureSampleLevel(scene_color, scene_sampler, near_uv, 0.0) * nw;
        near_weight += nw;
        coverage += nw / f32(count);
    }
    let far = far_color / max(far_weight, 0.00001);
    let far_mix = smoothstep(0.35, 1.5, radius) * step(focus, distance) * step(0.001, far_weight);
    let base = mix(center, far, far_mix);
    let near = near_color / max(near_weight, 0.00001);
    return mix(base, near, clamp(coverage, 0.0, 1.0));
}
