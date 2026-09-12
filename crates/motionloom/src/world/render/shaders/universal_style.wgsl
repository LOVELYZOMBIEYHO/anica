// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/universal_style.wgsl

// Every preset returns display RGB; universal controls never depend on its implementation.
// Work on straight colour and restore coverage exactly once, including transparent islands.
fn preview_screen_space_ao(uv: vec2<f32>) -> f32 {
    if (lighting.render_compat.y < 0.5 || lighting.environment2.z <= 0.0) {
        return 1.0;
    }
    let dimensions = vec2<f32>(textureDimensions(scene_depth));
    let pixel = vec2<i32>(clamp(uv * dimensions, vec2<f32>(0.0), dimensions - 1.0));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    if (center_depth <= 0.000001) { return 1.0; }
    let center_distance = view_distance(center_depth);
    let radius_px = clamp(lighting.environment2.w * 14.0, 2.0, 18.0);
    let offsets = array<vec2<f32>, 8>(
        vec2<f32>(1.0,0.0), vec2<f32>(-1.0,0.0),
        vec2<f32>(0.0,1.0), vec2<f32>(0.0,-1.0),
        vec2<f32>(0.707,0.707), vec2<f32>(-0.707,0.707),
        vec2<f32>(0.707,-0.707), vec2<f32>(-0.707,-0.707)
    );
    var occlusion = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sample_uv = clamp(uv + offsets[i] * radius_px / dimensions,
            vec2<f32>(0.0001), vec2<f32>(0.9999));
        let sample_pixel = vec2<i32>(sample_uv * dimensions);
        let sample_depth = textureLoad(scene_depth, sample_pixel, 0);
        if (sample_depth > 0.000001) {
            let sample_distance = view_distance(sample_depth);
            let delta = center_distance - sample_distance;
            let range_weight = exp(-abs(delta) / max(center_distance * 0.08, 0.02));
            occlusion += select(0.0, range_weight, delta > max(center_distance * 0.002, 0.001));
        }
    }
    return 1.0 - clamp(occlusion / 8.0 * lighting.environment2.z * 0.65, 0.0, 0.65);
}

fn preview_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_depth));
    let distance = view_distance(depth);
    // Remove the current projection jitter before reconstructing the ray.
    let stable_uv = uv - lighting.preview1.xy / dimensions;
    let ndc = vec2<f32>(stable_uv.x * 2.0 - 1.0, 1.0 - stable_uv.y * 2.0);
    let view_x = ndc.x * distance * dimensions.x / max(2.0 * lighting.camera0.w, 0.000001);
    let view_y = ndc.y * distance * dimensions.y / max(2.0 * lighting.camera0.w, 0.000001);
    return lighting.camera0.xyz + lighting.camera3.xyz * distance
        + lighting.camera1.xyz * view_x + lighting.camera2.xyz * view_y;
}

fn preview_previous_uv(world: vec3<f32>) -> vec2<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_depth));
    let relative = world - lighting.previous_camera0.xyz;
    let view_x = dot(relative, lighting.previous_camera1.xyz);
    let view_y = dot(relative, lighting.previous_camera2.xyz);
    let view_z = dot(relative, lighting.previous_camera3.xyz);
    if (view_z <= max(lighting.previous_camera1.w, 0.0001)) {
        return vec2<f32>(-1.0);
    }
    var uv = vec2<f32>(
        0.5 + view_x * lighting.previous_camera0.w / (view_z * dimensions.x),
        0.5 - view_y * lighting.previous_camera0.w / (view_z * dimensions.y)
    );
    uv += lighting.preview1.zw / dimensions;
    return uv;
}

// Screen velocity in normalized UV units, reconstructed from depth and the
// previous camera. It is shared by TAA and motion blur.
fn preview_velocity(uv: vec2<f32>) -> vec2<f32> {
    if (lighting.preview0.w < 0.5) { return vec2<f32>(0.0); }
    let dimensions = vec2<f32>(textureDimensions(scene_depth));
    let pixel = vec2<i32>(clamp(uv * dimensions, vec2<f32>(0.0), dimensions - 1.0));
    return textureLoad(preview_gbuffer, pixel, 0).zw;
}

fn preview_decode_normal(encoded: vec2<f32>) -> vec3<f32> {
    let value = encoded * 2.0 - 1.0;
    var normal = vec3<f32>(value, 1.0 - abs(value.x) - abs(value.y));
    if (normal.z < 0.0) {
        normal = vec3<f32>((vec2<f32>(1.0) - abs(normal.yx)) * sign(normal.xy), normal.z);
    }
    return normalize(normal);
}

fn preview_physical_velocity(uv: vec2<f32>) -> vec2<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_depth));
    // Projection jitter belongs to TAA reprojection, not the camera shutter.
    return preview_velocity(uv) - (lighting.preview1.xy - lighting.preview1.zw) / dimensions;
}

fn preview_motion_blur(radiance: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    if (lighting.preview0.z < 0.5 || lighting.preview0.w < 0.5) { return radiance; }
    let dimensions = vec2<f32>(textureDimensions(scene_color));
    var velocity = preview_physical_velocity(uv);
    let velocity_pixels = length(velocity * dimensions);
    if (velocity_pixels < 0.35) { return radiance; }
    velocity *= min(1.0, 24.0 / max(velocity_pixels, 0.001));
    var accumulated = radiance;
    var weight = 1.0;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let phase = (f32(i) + 0.5) / 6.0 - 0.5;
        let sample_uv = clamp(uv - velocity * phase, vec2<f32>(0.0001), vec2<f32>(0.9999));
        accumulated += textureSampleLevel(scene_color, scene_sampler, sample_uv, 0.0);
        weight += 1.0;
    }
    let blurred = accumulated / weight;
    let shutter_weight = smoothstep(0.35, 10.0, velocity_pixels) * 0.72;
    return mix(radiance, blurred, shutter_weight);
}

fn preview_screen_space_reflection(radiance: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    if (lighting.preview0.y < 0.5 || radiance.a < 0.00001) { return radiance; }
    let dimensions_i = textureDimensions(scene_depth);
    let dimensions = vec2<f32>(dimensions_i);
    let pixel = vec2<i32>(clamp(uv * dimensions, vec2<f32>(1.0), dimensions - 2.0));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    if (center_depth <= 0.000001) { return radiance; }
    let material = textureLoad(preview_material, pixel, 0);
    let roughness = material.r;
    let metallic = material.g;
    if (roughness > 0.92) { return radiance; }

    let world = preview_world_position(uv, center_depth);
    var normal = preview_decode_normal(textureLoad(preview_gbuffer, pixel, 0).xy);
    let incident = normalize(world - lighting.camera0.xyz);
    if (dot(normal, incident) > 0.0) { normal = -normal; }
    let reflected = normalize(reflect(incident, normal));
    let grazing = 1.0 - clamp(abs(dot(normal, -incident)), 0.0, 1.0);
    if (grazing < 0.08) { return radiance; }

    let center_distance = view_distance(center_depth);
    let stride = max(center_distance * 0.035, 0.025);
    var hit = vec4<f32>(0.0);
    var confidence = 0.0;
    for (var step = 1u; step <= 14u; step = step + 1u) {
        let ray_world = world + reflected * stride * f32(step);
        let relative = ray_world - lighting.camera0.xyz;
        let ray_z = dot(relative, lighting.camera3.xyz);
        if (ray_z <= lighting.camera1.w) { continue; }
        let ray_uv = vec2<f32>(
            0.5 + dot(relative, lighting.camera1.xyz) * lighting.camera0.w / (ray_z * dimensions.x),
            0.5 - dot(relative, lighting.camera2.xyz) * lighting.camera0.w / (ray_z * dimensions.y)
        ) + lighting.preview1.xy / dimensions;
        if (any(ray_uv <= vec2<f32>(0.002)) || any(ray_uv >= vec2<f32>(0.998))) { break; }
        let ray_pixel = vec2<i32>(ray_uv * dimensions);
        let scene_sample_depth = textureLoad(scene_depth, ray_pixel, 0);
        if (scene_sample_depth > 0.000001) {
            let scene_distance = view_distance(scene_sample_depth);
            let thickness = max(stride * 1.8, scene_distance * 0.008);
            if (ray_z >= scene_distance - thickness && ray_z <= scene_distance + thickness) {
                hit = textureSampleLevel(scene_color, scene_sampler, ray_uv, 0.0);
                confidence = (1.0 - f32(step) / 15.0)
                    * (1.0 - smoothstep(0.82, 1.0, max(abs(ray_uv.x - 0.5), abs(ray_uv.y - 0.5)) * 2.0));
                break;
            }
        }
    }
    let surface_response = pow(1.0 - roughness, 2.0) * mix(0.10, 0.32, metallic);
    let reflection_weight = confidence * grazing * surface_response;
    return mix(radiance, hit, reflection_weight);
}

fn preview_temporal_resolve(current: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    if (lighting.preview0.x < 0.5 || lighting.preview0.w < 0.5 || current.a < 0.00001) {
        return current;
    }
    let previous_uv = uv - preview_velocity(uv);
    if (any(previous_uv <= vec2<f32>(0.001)) || any(previous_uv >= vec2<f32>(0.999))) {
        return current;
    }
    let history_sample = textureSampleLevel(history_color, scene_sampler, previous_uv, 0.0);
    // A luminance-aware neighbourhood proxy rejects disocclusion ghosts while
    // preserving enough history to converge sub-pixel edges.
    let allowance = vec3<f32>(0.035 + 0.18 * length(current.rgb));
    let history = vec4<f32>(
        clamp(history_sample.rgb, current.rgb - allowance, current.rgb + allowance),
        clamp(history_sample.a, current.a - 0.08, current.a + 0.08)
    );
    let dimensions = vec2<f32>(textureDimensions(scene_depth));
    // History lookup needs jittered velocity, but confidence must only react to
    // physical motion. Otherwise the Halton sequence changes the blend weight
    // of a completely static thin surface every frame.
    let speed_pixels = length(preview_physical_velocity(uv) * dimensions);
    let dimensions_i = textureDimensions(preview_material);
    let pixel = vec2<i32>(clamp(uv * vec2<f32>(dimensions_i), vec2<f32>(0.0), vec2<f32>(dimensions_i) - 1.0));
    let reactive = textureLoad(preview_material, pixel, 0).a;
    let history_weight = mix(0.94, 0.56, smoothstep(0.0, 12.0, speed_pixels))
        * (1.0 - reactive * 0.75);
    return mix(current, history, history_weight);
}

fn finish_render_style(radiance: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
    let enhanced = preview_screen_space_reflection(preview_motion_blur(radiance, uv), uv);
    let resolved = resolve_display(enhanced);
    let alpha = resolved.a;
    if (alpha < 0.00001) { return vec4<f32>(0.0); }
    var rgb = resolved.rgb / alpha;
    rgb *= preview_screen_space_ao(uv);
    if (lighting.universal_highlight.w > 0.5) {
        rgb = ink_wash_soft_v1(rgb, uv);
    }
    // Skip neutral arithmetic exactly; gamma round trips can move an 8-bit boundary.
    let changes_color = any(lighting.universal_tone.xyz != vec3<f32>(1.0)) ||
        lighting.universal_shadow.w > 0.0 || lighting.universal_color.w > 0.0;
    if (lighting.universal_tone.w > 0.5 && changes_color) {
        // Exposure is a linear-light multiplier, not EV. Contrast pivots at display 0.5.
        rgb = pow(pow(max(rgb, vec3<f32>(0.0)), vec3<f32>(2.2)) * lighting.universal_tone.x, vec3<f32>(1.0 / 2.2));
        rgb = clamp((rgb - 0.5) * lighting.universal_tone.y + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
        let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let tones = mix(lighting.universal_shadow.rgb, lighting.universal_highlight.rgb, luminance);
        rgb = mix(rgb, tones, lighting.universal_shadow.w);
        rgb = mix(rgb, rgb * lighting.universal_color.rgb, lighting.universal_color.w);
        let gray = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        rgb = mix(vec3<f32>(gray), rgb, lighting.universal_tone.z);
    }
    let current = vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)) * alpha, alpha);
    return preview_temporal_resolve(current, uv);
}
