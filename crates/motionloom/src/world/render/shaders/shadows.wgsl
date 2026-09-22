// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/shadows.wgsl

fn world_to_shadow(world: vec3<f32>) -> vec3<f32> {
    let relative = world - lighting.shadow3.xyz;
    return vec3<f32>(
        dot(relative, lighting.shadow0.xyz) / lighting.shadow0.w * 0.5 + 0.5,
        0.5 - dot(relative, lighting.shadow1.xyz) / lighting.shadow1.w * 0.5,
        dot(relative, lighting.shadow2.xyz) / lighting.shadow2.w + 0.5
    );
}
fn sample_shadow(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (lighting.color1.w <= 0.0) {
        return 1.0;
    }
    let coordinate = world_to_shadow(world);
    if (any(coordinate.xy < vec2<f32>(0.0)) || any(coordinate.xy > vec2<f32>(1.0)) ||
        coordinate.z < 0.0 || coordinate.z > 1.0) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_texture));
    let light_facing = abs(dot(normalize(normal), normalize(lighting.shadow2.xyz)));
    let bias = lighting.shadow3.w * (1.0 + 2.0 * (1.0 - light_facing));
    if (lighting.surface3.y > 0.5) {
        return mix(1.0, textureSampleCompareLevel(shadow_texture, shadow_sampler, coordinate.xy, coordinate.z - bias), lighting.color1.w);
    }
    // Rotated Poisson PCF avoids the axis-aligned 3x3 stair pattern while
    // retaining explicit level-zero comparison for browser WebGPU.
    let poisson = array<vec2<f32>, 12>(
        vec2<f32>(-0.326,-0.406), vec2<f32>(-0.840,-0.074),
        vec2<f32>(-0.696,0.457), vec2<f32>(-0.203,0.621),
        vec2<f32>(0.962,-0.195), vec2<f32>(0.473,-0.480),
        vec2<f32>(0.519,0.767), vec2<f32>(0.185,-0.893),
        vec2<f32>(0.507,0.064), vec2<f32>(0.896,0.412),
        vec2<f32>(-0.322,-0.933), vec2<f32>(-0.792,-0.598)
    );
    let angle = dot(floor(world.xz * 31.0), vec2<f32>(0.06711056, 0.00583715)) * 6.2831853;
    let rotation = mat2x2<f32>(cos(angle), -sin(angle), sin(angle), cos(angle));
    let radius = mix(1.25, 2.75, clamp(lighting.color1.z, 0.0, 1.0));
    var visibility = 0.0;
    for (var i = 0u; i < 12u; i = i + 1u) {
        visibility += textureSampleCompareLevel(
            shadow_texture,
            shadow_sampler,
            coordinate.xy + rotation * poisson[i] * texel * radius,
            coordinate.z - bias
        );
    }
    return mix(1.0, visibility / 12.0, lighting.color1.w);
}

@vertex
fn vs_shadow(input: VertexIn, @builtin(instance_index) instance_id: u32) -> @builtin(position) vec4<f32> {
    params = instance_params[instance_id];
    let weight_sum = input.weights.x + input.weights.y + input.weights.z + input.weights.w;
    var skinned = vegetation_deform(input.position);
    if (weight_sum > 0.000001) {
        skinned =
            bone_transform(input.joints.x, input.position) * (input.weights.x / weight_sum) +
            bone_transform(input.joints.y, input.position) * (input.weights.y / weight_sum) +
            bone_transform(input.joints.z, input.position) * (input.weights.z / weight_sum) +
            bone_transform(input.joints.w, input.position) * (input.weights.w / weight_sum);
    }
    let local = vec3<f32>(
        skinned.x - params.model.x,
        skinned.y - params.model.y,
        skinned.z - params.model.z,
    ) * params.model.w;
    let world = params.actor.xyz + actor_rotate(local);
    let shadow = world_to_shadow(world);
    return vec4<f32>(shadow.x * 2.0 - 1.0, 1.0 - shadow.y * 2.0, shadow.z, 1.0);
}
