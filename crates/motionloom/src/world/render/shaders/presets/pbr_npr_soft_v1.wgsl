// Soft PBR + NPR finish. The scene is lit by the regular physical surface
// shader first; this pass only simplifies tonal transitions and adds a very
// light painted edge response. It deliberately avoids a hard cartoon outline.
fn pbr_npr_soft_v1(center: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_color));
    let pixel = vec2<i32>(clamp(uv * dimensions, vec2<f32>(0.0), dimensions - 1.0));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    if (center_depth <= 0.000001) { return center; }

    let center_distance = view_distance(center_depth);
    let center_normal = preview_decode_normal(textureLoad(preview_gbuffer, pixel, 0).xy);
    let material = textureLoad(preview_material, pixel, 0);
    let roughness = material.r;
    let metallic = material.g;
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0)
    );

    var wash = center;
    var wash_weight = 1.0;
    var edge = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let sample_uv = clamp(
            uv + offsets[i] * 0.85 / dimensions,
            vec2<f32>(0.0001),
            vec2<f32>(0.9999)
        );
        let sample_pixel = vec2<i32>(sample_uv * dimensions);
        let sample_depth = textureLoad(scene_depth, sample_pixel, 0);
        if (sample_depth > 0.000001) {
            let sample_distance = view_distance(sample_depth);
            let sample_normal = preview_decode_normal(
                textureLoad(preview_gbuffer, sample_pixel, 0).xy
            );
            let depth_delta = abs(sample_distance - center_distance) /
                max(center_distance, 0.1);
            let normal_delta = 1.0 - clamp(dot(center_normal, sample_normal), 0.0, 1.0);
            let bilateral = exp(-depth_delta * 90.0) * exp(-normal_delta * 12.0);
            let sample = resolve_display(
                textureSampleLevel(scene_color, scene_sampler, sample_uv, 0.0)
            );
            let color = sample.rgb / max(sample.a, 0.00001);
            let weight = bilateral * sample.a * mix(0.08, 0.22, roughness);
            wash += color * weight;
            wash_weight += weight;
            edge = max(edge, max(
                smoothstep(0.008, 0.08, depth_delta),
                smoothstep(0.08, 0.42, normal_delta)
            ));
        }
    }
    wash /= wash_weight;

    let luminance_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luminance = clamp(dot(wash, luminance_weights), 0.0, 1.0);
    // Eight soft bands remove digital-looking micro-gradients while retaining
    // the PBR highlight shape. Metallic highlights receive even less grading.
    let bands = luminance * 8.0;
    let stepped_luminance = (floor(bands) + smoothstep(0.18, 0.82, fract(bands))) / 8.0;
    let highlight_protection = smoothstep(0.62, 0.96, luminance) * (0.45 + 0.55 * metallic);
    let tone_strength = 0.16 * (1.0 - 0.72 * highlight_protection);
    let painted_luminance = mix(luminance, stepped_luminance, tone_strength);
    var painted = vec3<f32>(painted_luminance) + (wash - vec3<f32>(luminance)) * 1.015;

    // Cool translucent shadows and warm paper-like highlights create a soft
    // illustration palette without replacing the authored material colours.
    let palette = mix(
        vec3<f32>(0.94, 0.965, 1.0),
        vec3<f32>(1.0, 0.985, 0.955),
        smoothstep(0.22, 0.82, luminance)
    );
    painted *= mix(vec3<f32>(1.0), palette, 0.075);
    painted *= 1.0 - edge * 0.022 * (1.0 - highlight_protection);

    // Preserve the source at polished specular peaks so leather, water and
    // metal continue to read as physical materials.
    let physical_weight = highlight_protection * 0.42;
    return clamp(mix(painted, center, physical_weight), vec3<f32>(0.0), vec3<f32>(1.0));
}
