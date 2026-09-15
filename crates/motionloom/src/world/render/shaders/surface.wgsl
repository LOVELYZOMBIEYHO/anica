// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/surface.wgsl

fn authored_light_radiance(light: Light, world_position: vec3<f32>) -> vec4<f32> {
    let kind = light.position_kind.w;
    var direction = normalize(-light.direction_range.xyz);
    var attenuation = 1.0;
    if (kind > 0.5) {
        let delta = light.position_kind.xyz - world_position;
        let distance = max(length(delta), 0.001);
        direction = delta / distance;
        let normalized_distance = distance / max(light.direction_range.w, 0.001);
        attenuation = pow(clamp(1.0 - pow(normalized_distance, 4.0), 0.0, 1.0), 2.0) /
            max(distance * distance, 0.25);
        if (kind > 1.5 && kind < 2.5) {
            let cone = dot(normalize(-direction), normalize(light.direction_range.xyz));
            attenuation *= smoothstep(light.spot_area.y, light.spot_area.x, cone);
        }
    }
    return vec4<f32>(direction, attenuation);
}

// Four deterministic emitter samples give Immediate Preview a stable,
// physically-scaled rectangular source without temporal noise.
fn authored_area_light_radiance(light: Light, world_position: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let emitter_normal = normalize(light.direction_range.xyz);
    let reference = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(emitter_normal.y) > 0.95);
    let right = normalize(cross(reference, emitter_normal));
    let up = normalize(cross(emitter_normal, right));
    let x = select(-0.288675, 0.288675, (sample_index & 1u) != 0u);
    let y = select(-0.288675, 0.288675, (sample_index & 2u) != 0u);
    let width = max(light.spot_area.z, 0.001);
    let height = max(light.spot_area.w, 0.001);
    let sample_position = light.position_kind.xyz + right * x * width + up * y * height;
    let delta = sample_position - world_position;
    let distance = max(length(delta), 0.001);
    let direction = delta / distance;
    let facing = max(dot(emitter_normal, -direction), 0.0);
    var attenuation = width * height * facing / (distance * distance);
    if (light.direction_range.w > 0.001) {
        let normalized_distance = distance / light.direction_range.w;
        attenuation *= pow(clamp(1.0 - pow(normalized_distance, 4.0), 0.0, 1.0), 2.0);
    }
    return vec4<f32>(direction, attenuation * 0.25);
}

fn shade_surface(input: VertexOut) -> vec4<f32> {
    if (input.hidden_weight > 0.01) {
        discard;
    }
    let uv = input.uv;
    let sampled = textureSample(actor_texture, actor_sampler, uv);
    let alpha = sampled.a * input.color.a * params.material4.a * params.style.x;
    if (alpha <= 0.001 || (params.material7.w > 0.0 && alpha < params.material7.w)) {
        discard;
    }
    // The sRGB texture format decodes before filtering. Undo it only for
    // display-locked materials; PBR uses the sampled linear radiance directly.
    let encoded_sample = select(1.055 * pow(max(sampled.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055,
        sampled.rgb * 12.92, sampled.rgb <= vec3<f32>(0.0031308));
    let base_srgb = clamp(
        encoded_sample * input.color.rgb * params.material4.rgb,
        vec3<f32>(0.0), vec3<f32>(1.0)
    );
    var base_color = sampled.rgb * pow(clamp(input.color.rgb * params.material4.rgb,
        vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
    if (lighting.surface0.x < -0.5) {
        // glTF factors and vertex colors are linear, unlike legacy display factors.
        base_color = sampled.rgb * input.color.rgb * params.material4.rgb;
    }
    if (lighting.surface3.x != 1.0) {
        base_color = max(mix(vec3<f32>(dot(base_color, vec3<f32>(0.2126,0.7152,0.0722))), base_color, lighting.surface3.x), vec3<f32>(0.0));
    }
    let mr_sample = textureSample(metallic_roughness_texture, actor_sampler, uv);
    var metallic = clamp(params.material0.x * mr_sample.b, 0.0, 1.0);
    var roughness = clamp(params.material0.y * mr_sample.g + lighting.surface1.z, 0.045, 1.0);
    if (lighting.surface0.x > 2.5 && lighting.surface0.x < 3.5) {
        base_color = clay_base_color();
        metallic = clay_metallic();
        roughness = clay_roughness();
    }

    let geometric_normal = normalize(input.normal);
    let tangent = normalize(input.tangent - geometric_normal * dot(input.tangent, geometric_normal));
    let bitangent_sign = select(-1.0, 1.0, dot(cross(geometric_normal, tangent), input.bitangent) >= 0.0);
    let bitangent = normalize(cross(geometric_normal, tangent)) * bitangent_sign;
    let sampled_normal = textureSample(normal_texture, actor_sampler, uv).xyz * 2.0 - 1.0;
    let tangent_normal = normalize(vec3<f32>(
        sampled_normal.x * params.material0.z,
        sampled_normal.y * params.material0.z,
        sampled_normal.z,
    ));
    let normal = normalize(
        tangent * tangent_normal.x + bitangent * tangent_normal.y + geometric_normal * tangent_normal.z
    );

    let view = normalize(params.camera0.xyz - input.world_position);
    let authored_ior = clamp(params.material6.y, 1.0, 3.0);
    let ior_ratio = (authored_ior - 1.0) / (authored_ior + 1.0);
    let dielectric_f0 = vec3<f32>(ior_ratio * ior_ratio) *
        params.material0.w * params.material2.rgb;
    let f0 = mix(dielectric_f0, base_color, metallic);
    var lit = vec3<f32>(0.0);
    let light_count = u32(lighting.environment2.y + 0.5);
    var face_band = -1.0;
    if (lighting.surface0.x > 3.5 && params.cel_material0.y > 2.5) {
        var key = normalize(vec3<f32>(-0.42, 0.78, 0.47));
        if (light_count > 0u) { key = authored_light_radiance(lighting.lights[0], input.world_position).xyz; }
        let forward = normalize(input.face_forward);
        let right = normalize(input.face_right);
        let lateral = dot(key, right);
        let frontal = dot(key, forward);
        let planar = max(length(vec2<f32>(lateral, frontal)), 0.0001);
        let threshold = 1.0 - clamp(frontal / planar, 0.0, 1.0);
        let face_uv = vec2<f32>(select(uv.x, 1.0 - uv.x, lateral < 0.0), uv.y);
        let sdf = textureSampleLevel(cel_texture, actor_sampler, face_uv, 0.0).g;
        face_band = smoothstep(threshold - lighting.cel0.y, threshold + lighting.cel0.y, sdf);
        if (frontal <= 0.0) { face_band = 0.0; }
    }
    for (var light_index = 0u; light_index < 8u; light_index = light_index + 1u) {
        if (light_index < light_count) {
            let authored = lighting.lights[light_index];
            let direction_attenuation = authored_light_radiance(authored, input.world_position);
            var radiance = authored.color_intensity.rgb * authored.color_intensity.w * direction_attenuation.w;
            if (lighting.surface0.x < -0.5) {
                // The selected shadow owner's index is packed independently of kind.
                if (f32(light_index) == lighting.render_compat.x) { radiance *= sample_shadow(input.world_position, normal); }
            }
            if (authored.position_kind.w > 2.5) {
                for (var area_sample = 0u; area_sample < 4u; area_sample = area_sample + 1u) {
                    let area_direction_attenuation = authored_area_light_radiance(authored, input.world_position, area_sample);
                    var area_radiance = authored.color_intensity.rgb * authored.color_intensity.w * area_direction_attenuation.w;
                    if (lighting.surface0.x < -0.5 && f32(light_index) == lighting.render_compat.x) {
                        area_radiance *= sample_shadow(input.world_position, normal);
                    }
                    lit += direct_pbr(
                        normal, view, area_direction_attenuation.xyz, area_radiance,
                        base_color, metallic, roughness, f0, face_band
                    );
                }
            } else if (lighting.surface0.x > 3.5 && light_index > 0u) {
                // Only the first authored light shapes cel bands; others provide soft fill.
                lit += base_color * radiance * max(dot(normal, direction_attenuation.xyz), 0.0) * 0.15 / 3.14159265;
            } else {
                lit += direct_pbr(
                    normal, view, direction_attenuation.xyz, radiance,
                    base_color, metallic, roughness, f0, face_band
                );
            }
        }
    }
    // Scenes without authored lighting retain the earlier studio setup.
    if (light_count == 0u && lighting.environment0.w < 0.5) {
        lit += direct_pbr(
            normal, view, normalize(vec3<f32>(-0.42, 0.78, 0.47)), vec3<f32>(4.2, 4.0, 3.75),
            base_color, metallic, roughness, f0, face_band
        );
        lit += direct_pbr(
            normal, view, normalize(vec3<f32>(0.68, 0.28, 0.51)), vec3<f32>(1.25, 1.45, 1.75),
            base_color, metallic, roughness, f0, face_band
        );
    }
    if (lighting.surface0.x >= -0.5) { lit *= sample_shadow(input.world_position, normal); }
    let n_dot_v = max(dot(normal, view), 0.0);
    let environment_fresnel = fresnel_schlick(n_dot_v, f0);
    let diffuse_environment = sample_environment(normal, lighting.environment0.z) *
        lighting.environment0.x * lighting.environment1.z;
    let reflected = reflect(-view, normal);
    let specular_environment = sample_environment(reflected, roughness * lighting.environment0.z) *
        lighting.environment0.x * lighting.environment1.w;
    let ao = clamp(1.0 - lighting.environment2.z * (1.0 - max(normal.y, 0.0)) * 0.35, 0.15, 1.0);
    let contact = 1.0 - lighting.color1.x *
        (1.0 - smoothstep(0.0, max(lighting.color1.y, 0.001), max(input.world_position.y, 0.0))) *
        (0.45 + 0.55 * (1.0 - lighting.color1.z));
    var diffuse_ambient = base_color * (1.0 - metallic) * diffuse_environment * lighting.surface2.rgb * lighting.surface2.w;
    if (lighting.surface0.x < -0.5) {
        // Environment irradiance remains available when hemisphere fill is zero.
        diffuse_ambient = base_color * (1.0 - metallic) * diffuse_environment;
    }
    let specular_ambient = environment_fresnel * specular_environment * lighting.surface1.y;
    let material_ao = textureSample(occlusion_texture, actor_sampler, uv).r;
    lit += (diffuse_ambient + specular_ambient) * ao * contact * material_ao;
    lit += base_color * lighting.surface0.w * pow(1.0 - n_dot_v, lighting.surface1.x);
    // Tangent-aligned hair highlights are optional and do not change PBR materials.
    if (lighting.surface0.x > 3.5 && params.cel_material0.y > 1.5 && params.cel_material0.y < 2.5) {
        let key = select(normalize(vec3<f32>(-0.42,0.78,0.47)), authored_light_radiance(lighting.lights[0], input.world_position).xyz, light_count > 0u);
        let half_vector = normalize(view + key);
        let strand = sqrt(max(0.0, 1.0 - pow(dot(tangent, half_vector), 2.0)));
        lit += base_color * smoothstep(0.96, 0.99, strand) * params.cel_material0.z
            * textureSampleLevel(cel_texture, actor_sampler, uv, 0.0).b;
    }

    let emissive_sample = textureSample(emissive_texture, actor_sampler, uv).rgb;
    lit += emissive_sample * params.material1.rgb;
    let force_unlit = max(params.material1.w, params.material2.w);
    let lighting_mix = params.style.y * (1.0 - force_unlit);
    let shaded = mix(base_color, lit, lighting_mix);
    let surface_exposure = mix(params.style.z, 1.0, params.material2.w);
    let exposed = shaded * surface_exposure;
    var display = display_transform(exposed);
    if (params.material2.w > 0.5) { display = inverse_display_curve(base_srgb); }
    let fog_amount = atmosphere_fog_amount(input.world_position);
    let fog_radiance = lighting.fog1.rgb * (1.0 + lighting.fog2.y * 0.35);
    display = mix(display, display_transform(fog_radiance), fog_amount);
    var output_alpha = alpha;
    let transmission = clamp(params.material6.x, 0.0, 1.0);
    if (transmission > 0.001) {
        // Beer-Lambert attenuation gives thick glass stronger colour without
        // treating transmission as missing surface coverage.
        let optical_distance = params.material6.z / max(params.material6.w, 0.0001);
        let attenuation = pow(
            clamp(params.material7.rgb, vec3<f32>(0.0001), vec3<f32>(1.0)),
            vec3<f32>(optical_distance)
        );
        let fresnel_strength = max(
            environment_fresnel.r,
            max(environment_fresnel.g, environment_fresnel.b)
        );
        display = mix(display * attenuation, display, fresnel_strength);
        output_alpha *= clamp(
            (1.0 - transmission) + fresnel_strength + roughness * 0.08,
            0.015,
            1.0
        );
    }
    return vec4<f32>(display, output_alpha);
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    params = instance_params[input.instance_id];
    return shade_surface(input);
}

struct SurfaceGbufferOutput {
    @location(0) color: vec4<f32>,
    @location(1) geometry: vec4<f32>,
    @location(2) material: vec4<f32>,
};

fn encode_gbuffer_normal(value: vec3<f32>) -> vec2<f32> {
    var normal = value / max(abs(value.x) + abs(value.y) + abs(value.z), 0.000001);
    if (normal.z < 0.0) {
        normal = vec3<f32>((vec2<f32>(1.0) - abs(normal.yx)) * sign(normal.xy), normal.z);
    }
    return normal.xy * 0.5 + 0.5;
}

@fragment
fn fs_main_gbuffer(input: VertexOut) -> SurfaceGbufferOutput {
    params = instance_params[input.instance_id];
    let color = shade_surface(input);
    let geometric_normal = normalize(input.normal);
    var normal = geometric_normal;
    var metallic = 0.0;
    var roughness = 1.0;
    var ao = 1.0;
    // Balanced does not run SSR, so avoid repeating material texture reads that
    // the main surface shader has already performed. Cinematic pays for the
    // richer G-buffer only when its material-aware reflection actually uses it.
    if (lighting.preview0.y > 0.5) {
        let tangent = normalize(input.tangent
            - geometric_normal * dot(input.tangent, geometric_normal));
        let bitangent_sign = select(-1.0, 1.0,
            dot(cross(geometric_normal, tangent), input.bitangent) >= 0.0);
        let bitangent = normalize(cross(geometric_normal, tangent)) * bitangent_sign;
        let sampled_normal = textureSample(normal_texture, actor_sampler, input.uv).xyz * 2.0 - 1.0;
        let tangent_normal = normalize(vec3<f32>(
            sampled_normal.x * params.material0.z,
            sampled_normal.y * params.material0.z,
            sampled_normal.z,
        ));
        normal = normalize(tangent * tangent_normal.x
            + bitangent * tangent_normal.y + geometric_normal * tangent_normal.z);
        let mr = textureSample(metallic_roughness_texture, actor_sampler, input.uv);
        metallic = clamp(params.material0.x * mr.b, 0.0, 1.0);
        roughness = clamp(params.material0.y * mr.g + lighting.surface1.z, 0.045, 1.0);
        ao = mix(1.0, textureSample(occlusion_texture, actor_sampler, input.uv).r,
            clamp(params.material2.w, 0.0, 1.0));
    }

    var velocity = vec2<f32>(0.0);
    if (params.motion0.y > 0.5
        && input.current_clip.w > 0.0001
        && input.previous_clip.w > 0.0001) {
        let current_ndc = input.current_clip.xy / input.current_clip.w;
        let current_uv = vec2<f32>(current_ndc.x * 0.5 + 0.5, 0.5 - current_ndc.y * 0.5);
        let previous_ndc = input.previous_clip.xy / input.previous_clip.w;
        let previous_uv = vec2<f32>(previous_ndc.x * 0.5 + 0.5, 0.5 - previous_ndc.y * 0.5);
        if (all(previous_uv >= vec2<f32>(-0.05)) && all(previous_uv <= vec2<f32>(1.05))) {
            // Store physical motion on the stable output grid. Projection
            // jitter is a sampling pattern, never object or shutter motion.
            velocity = current_uv - previous_uv
                - (lighting.preview1.xy - lighting.preview1.zw) / params.canvas.xy;
        }
    }

    let emissive_factor = length(params.material1.rgb) * params.material1.w;
    let reactive = clamp(max(params.material6.x, emissive_factor * 0.25), 0.0, 1.0);
    return SurfaceGbufferOutput(
        color,
        vec4<f32>(encode_gbuffer_normal(normal), velocity),
        vec4<f32>(roughness, metallic, ao, reactive),
    );
}

@fragment
fn fs_transmissive(input: VertexOut) -> @location(0) vec4<f32> {
    params = instance_params[input.instance_id];
    if (input.hidden_weight > 0.01) {
        discard;
    }
    let surface = shade_surface(input);
    let transmission = clamp(params.material6.x, 0.0, 1.0);
    let ior = clamp(params.material6.y, 1.0, 3.0);
    let optical_thickness = max(params.material6.z, 0.0);
    let screen_uv = input.pos.xy / params.canvas.xy;
    let view_normal = normalize(input.normal);
    let refractive_scale = (1.0 - 1.0 / ior) *
        (optical_thickness / (1.0 + optical_thickness)) * 0.08;
    let refracted_uv = clamp(
        screen_uv + view_normal.xy * refractive_scale,
        vec2<f32>(0.001),
        vec2<f32>(0.999)
    );
    let scene_center = textureSample(
        opaque_scene_texture,
        opaque_scene_sampler,
        refracted_uv
    ).rgb;
    let blur_radius = clamp(params.material0.y, 0.0, 1.0) * 0.006;
    let scene_color = scene_center * 0.4 +
        textureSample(opaque_scene_texture, opaque_scene_sampler,
            refracted_uv + vec2<f32>(blur_radius, 0.0)).rgb * 0.15 +
        textureSample(opaque_scene_texture, opaque_scene_sampler,
            refracted_uv - vec2<f32>(blur_radius, 0.0)).rgb * 0.15 +
        textureSample(opaque_scene_texture, opaque_scene_sampler,
            refracted_uv + vec2<f32>(0.0, blur_radius)).rgb * 0.15 +
        textureSample(opaque_scene_texture, opaque_scene_sampler,
            refracted_uv - vec2<f32>(0.0, blur_radius)).rgb * 0.15;
    let optical_distance = optical_thickness / max(params.material6.w, 0.0001);
    let attenuation = pow(
        clamp(params.material7.rgb, vec3<f32>(0.0001), vec3<f32>(1.0)),
        vec3<f32>(optical_distance)
    );
    let view = normalize(params.camera0.xyz - input.world_position);
    let ior_ratio = (ior - 1.0) / (ior + 1.0);
    let f0 = ior_ratio * ior_ratio;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - abs(dot(view_normal, view)), 5.0);
    let reflected_weight = clamp((1.0 - transmission) + fresnel, 0.0, 1.0);
    let transmitted_color = scene_color * attenuation;
    let glass_color = mix(transmitted_color, surface.rgb, reflected_weight);
    // The sampled opaque scene is already inside glass_color, so full coverage
    // avoids blending the same background into the result a second time.
    return vec4<f32>(glass_color, params.style.x);
}

struct BackgroundVertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) index: u32) -> BackgroundVertexOut {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: BackgroundVertexOut;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_background(input: BackgroundVertexOut) -> @location(0) vec4<f32> {
    if (lighting.environment2.x < 0.5 || lighting.environment0.w < 0.5) {
        discard;
    }
    let ndc = input.uv * 2.0 - vec2<f32>(1.0);
    let aspect = max(lighting.camera3.w, 0.001);
    let direction = normalize(
        lighting.camera3.xyz +
        lighting.camera1.xyz * ndc.x * aspect +
        lighting.camera2.xyz * -ndc.y
    );
    let lod = lighting.environment1.y * lighting.environment0.z;
    let color = sample_environment(direction, lod) * lighting.environment1.x;
    var display = display_transform(color);
    if (lighting.fog2.z > 0.5 && lighting.fog2.w > 0.5) {
        let horizon = pow(clamp(1.0 - abs(direction.y), 0.0, 1.0), 3.0);
        var sky_fog = horizon * clamp(lighting.fog0.y * 8.0 + lighting.fog2.y * 0.15, 0.0, 0.75);
        if (lighting.fog3.w > 0.5) {
            let volume_sample = atmosphere_fog_ray_sample(direction, 100000.0, lighting.fog1.w);
            let volume_distance = max(volume_sample.x - lighting.fog0.z, 0.0);
            sky_fog = clamp(
                (1.0 - exp(-lighting.fog0.y * volume_distance)) * volume_sample.y,
                0.0,
                0.75,
            );
        }
        display = mix(display, display_transform(lighting.fog1.rgb), sky_fog);
    }
    return vec4<f32>(display, 1.0);
}
