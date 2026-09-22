// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/dof.wgsl

struct Light {
    position_kind: vec4<f32>,
    direction_range: vec4<f32>,
    color_intensity: vec4<f32>,
    spot_area: vec4<f32>,
};

struct Lighting {
    environment0: vec4<f32>,
    environment1: vec4<f32>,
    environment2: vec4<f32>,
    color0: vec4<f32>,
    color1: vec4<f32>,
    fog0: vec4<f32>,
    fog1: vec4<f32>,
    fog2: vec4<f32>,
    fog3: vec4<f32>,
    fog4: vec4<f32>,
    optics0: vec4<f32>,
    dof_style: vec4<f32>,
    render_compat: vec4<f32>,
    camera0: vec4<f32>,
    camera1: vec4<f32>,
    camera2: vec4<f32>,
    camera3: vec4<f32>,
    previous_camera0: vec4<f32>,
    previous_camera1: vec4<f32>,
    previous_camera2: vec4<f32>,
    previous_camera3: vec4<f32>,
    preview0: vec4<f32>,
    preview1: vec4<f32>,
    preview2: vec4<f32>,
    shadow0: vec4<f32>,
    shadow1: vec4<f32>,
    shadow2: vec4<f32>,
    shadow3: vec4<f32>,
    surface0: vec4<f32>,
    surface1: vec4<f32>,
    surface2: vec4<f32>,
    surface3: vec4<f32>,
    cel0: vec4<f32>,
    cel1: vec4<f32>,
    cel2: vec4<f32>,
    universal_color: vec4<f32>,
    universal_tone: vec4<f32>,
    universal_shadow: vec4<f32>,
    universal_highlight: vec4<f32>,
    lights: array<Light, 8>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> lighting: Lighting;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var scene_depth: texture_depth_2d;
@group(0) @binding(4) var history_color: texture_2d<f32>;
@group(0) @binding(5) var preview_gbuffer: texture_2d<f32>;
@group(0) @binding(6) var preview_material: texture_2d<f32>;
@group(0) @binding(7) var history_depth: texture_depth_2d;
@group(0) @binding(8) var history_gbuffer: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: VertexOut;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// Color arrives premultiplied from hardware blending. Map straight radiance
// and restore coverage so transparent silhouettes do not gain a dark fringe.
fn resolve_display(color: vec4<f32>) -> vec4<f32> {
    let alpha = clamp(color.a, 0.0, 1.0);
    if (alpha < 0.00001) { return vec4<f32>(0.0); }
    let linear = max(color.rgb / alpha, vec3<f32>(0.0));
    var mapped = linear;
    if (lighting.color0.w > 2.5) {
        // Filmic ACES transform; exposure is already applied upstream.
        let input_matrix = mat3x3<f32>(vec3<f32>(0.59719,0.07600,0.02840),
            vec3<f32>(0.35458,0.90834,0.13383),vec3<f32>(0.04823,0.01566,0.83777));
        let output_matrix = mat3x3<f32>(vec3<f32>(1.60475,-0.10208,-0.00327),
            vec3<f32>(-0.53108,1.10813,-0.07276),vec3<f32>(-0.07367,-0.00605,1.07602));
        let v = input_matrix * (linear / 0.6);
        mapped = clamp(output_matrix * ((v * (v + 0.0245786) - 0.000090537) /
            (v * (0.983729 * v + 0.4329510) + 0.238081)), vec3<f32>(0.0), vec3<f32>(1.0));
        let srgb = select(1.055 * pow(mapped, vec3<f32>(1.0 / 2.4)) - 0.055,
            mapped * 12.92, mapped <= vec3<f32>(0.0031308));
        return vec4<f32>(srgb * alpha, alpha);
    } else if (lighting.color0.w > 1.5) {
        mapped = clamp((linear * (2.51 * linear + 0.03)) /
            (linear * (2.43 * linear + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (lighting.color0.w > 0.5) { mapped = linear / (vec3<f32>(1.0) + linear); }
    return vec4<f32>(pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)) * alpha, alpha);
}

fn view_distance(depth: f32) -> f32 {
    let near = lighting.camera1.w;
    let far = max(lighting.camera2.w, near + 0.001);
    return near * far / max(near + depth * (far - near), 0.000001);
}

fn circle_of_confusion(distance: f32, image_height: f32) -> f32 {
    if (lighting.optics0.x <= 0.0 || lighting.optics0.w <= 0.0) { return 0.0; }
    let focus = max(lighting.optics0.x, 0.05);
    let focal = clamp(lighting.optics0.y * 0.001, 0.001, focus * 0.95);
    let aperture = focal / max(lighting.optics0.z, 0.7);
    let sensor_coc = abs(aperture * focal * (focus - distance) /
        max(distance * (focus - focal), 0.000001));
    return clamp(sensor_coc / 0.024 * image_height, 0.0, lighting.optics0.w);
}

// FXAA resolves high-contrast edges without temporal history or extra buffers.
fn fxaa_color(uv: vec2<f32>) -> vec4<f32> {
    let center = textureSampleLevel(scene_color, scene_sampler, uv, 0.0);
    let texel = 1.0 / vec2<f32>(textureDimensions(scene_color));
    let quality = clamp(lighting.render_compat.z, 0.0, 3.0);
    let nw = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(-1.0,-1.0) * texel, 0.0).rgb;
    let ne = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(1.0,-1.0) * texel, 0.0).rgb;
    let sw = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(-1.0,1.0) * texel, 0.0).rgb;
    let se = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(1.0,1.0) * texel, 0.0).rgb;
    let luma = vec3<f32>(0.299,0.587,0.114);
    let a = dot(nw,luma); let b = dot(ne,luma); let c = dot(sw,luma); let d = dot(se,luma); let m = dot(center.rgb,luma);
    let lo = min(m,min(min(a,b),min(c,d))); let hi = max(m,max(max(a,b),max(c,d)));
    var edge_floor = 0.055;
    var relative_threshold = 0.18;
    var span = 4.0;
    if (quality > 0.5) {
        edge_floor = 0.035;
        relative_threshold = 0.125;
        span = 8.0;
    }
    if (quality > 1.5) {
        edge_floor = 0.022;
        relative_threshold = 0.08;
        span = 12.0;
    }
    if (hi - lo < max(edge_floor, hi * relative_threshold)) { return center; }
    var direction = vec2<f32>(-((a+b)-(c+d)), (a+c)-(b+d));
    let reduce = max((a+b+c+d)*0.03125,0.0078125);
    direction = clamp(direction / (min(abs(direction.x),abs(direction.y))+reduce),vec2<f32>(-span),vec2<f32>(span))*texel;
    let rgb_a = 0.5*(textureSampleLevel(scene_color,scene_sampler,uv-direction/6.0,0.0).rgb + textureSampleLevel(scene_color,scene_sampler,uv+direction/6.0,0.0).rgb);
    let rgb_b = rgb_a*0.5+0.25*(textureSampleLevel(scene_color,scene_sampler,uv-direction*0.5,0.0).rgb+textureSampleLevel(scene_color,scene_sampler,uv+direction*0.5,0.0).rgb);
    let lb = dot(rgb_b,luma);
    return vec4<f32>(select(rgb_b,rgb_a,lb<lo || lb>hi),center.a);
}

// A compact morphology-based edge search gives SMAA-like spatial stability
// without temporal history and remains practical on browser GPUs.
fn smaa_color(uv: vec2<f32>) -> vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(scene_color));
    let quality = clamp(lighting.render_compat.z, 0.0, 3.0);
    let center = textureSampleLevel(scene_color, scene_sampler, uv, 0.0);
    let left = textureSampleLevel(scene_color, scene_sampler, uv - vec2<f32>(texel.x, 0.0), 0.0);
    let right = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(texel.x, 0.0), 0.0);
    let up = textureSampleLevel(scene_color, scene_sampler, uv - vec2<f32>(0.0, texel.y), 0.0);
    let down = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(0.0, texel.y), 0.0);
    let luma = vec3<f32>(0.299, 0.587, 0.114);
    let horizontal = abs(dot(left.rgb - right.rgb, luma));
    let vertical = abs(dot(up.rgb - down.rgb, luma));
    let contrast = max(horizontal, vertical);
    var threshold = 0.055;
    var strength = 1.8;
    var blend_limit = 0.45;
    if (quality > 0.5) {
        threshold = 0.035;
        strength = 2.5;
        blend_limit = 0.65;
    }
    if (quality > 1.5) {
        threshold = 0.02;
        strength = 3.2;
        blend_limit = 0.75;
    }
    if (contrast < threshold) { return center; }
    var neighbours = select((left + right) * 0.5, (up + down) * 0.5, horizontal < vertical);
    // High and Ultra pay for a wider edge search. They intentionally share
    // the same spatial kernel, matching the public quality table.
    if (quality > 1.5) {
        let left2 = textureSampleLevel(scene_color, scene_sampler, uv - vec2<f32>(2.0 * texel.x, 0.0), 0.0);
        let right2 = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(2.0 * texel.x, 0.0), 0.0);
        let up2 = textureSampleLevel(scene_color, scene_sampler, uv - vec2<f32>(0.0, 2.0 * texel.y), 0.0);
        let down2 = textureSampleLevel(scene_color, scene_sampler, uv + vec2<f32>(0.0, 2.0 * texel.y), 0.0);
        let wide = select((left2 + right2) * 0.5, (up2 + down2) * 0.5, horizontal < vertical);
        neighbours = mix(neighbours, wide, 0.35);
    }
    let blend = clamp((contrast - threshold) * strength, 0.0, blend_limit);
    return mix(center, neighbours, blend);
}

fn antialiased_color(uv: vec2<f32>) -> vec4<f32> {
    let method = lighting.surface3.w;
    if (method < 0.5) { return textureSampleLevel(scene_color, scene_sampler, uv, 0.0); }
    if (method < 1.5) { return fxaa_color(uv); }
    if (method < 2.5) { return smaa_color(uv); }
    return smaa_color(uv);
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(scene_color));
    // Raster jitter moves geometry in the source buffers. Sampling at the
    // inverse location places every TAA result back on a stable output grid;
    // only sub-pixel coverage is allowed to vary between phases.
    var jitter_uv = select(
        vec2<f32>(0.0),
        lighting.preview1.xy / dimensions,
        lighting.preview0.x > 0.5
    );
    let jittered_uv = clamp(input.uv + jitter_uv, vec2<f32>(0.0001), vec2<f32>(0.9999));
    let jittered_pixel = vec2<i32>(clamp(
        jittered_uv * dimensions,
        vec2<f32>(0.0),
        dimensions - 1.0
    ));
    let stable_pixel = vec2<i32>(clamp(
        input.uv * dimensions,
        vec2<f32>(0.0),
        dimensions - 1.0
    ));
    // The CPU/LDR background is not rasterized with the projection jitter.
    // Keep its interior on the stable grid while retaining jitter at geometry
    // coverage boundaries where either depth sample contains a surface.
    let jittered_depth = textureLoad(scene_depth, jittered_pixel, 0);
    let stable_depth = textureLoad(scene_depth, stable_pixel, 0);
    if (jittered_depth <= 0.000001 && stable_depth <= 0.000001) {
        jitter_uv = vec2<f32>(0.0);
    }
    let sample_uv = clamp(input.uv + jitter_uv, vec2<f32>(0.0001), vec2<f32>(0.9999));
    let pixel = vec2<i32>(clamp(sample_uv * dimensions, vec2<f32>(0.0), dimensions - 1.0));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    let center_distance = view_distance(center_depth);
    let radius_px = circle_of_confusion(center_distance, dimensions.y);
    // Explicit LOD keeps the sample legal inside the depth-dependent branch
    // below. Browser WebGPU enforces derivative-uniformity more strictly than
    // native Metal; implicit `textureSample` there can invalidate the DoF pass
    // even though the preceding 3D render completed successfully.
    let center = antialiased_color(sample_uv);
    if (lighting.dof_style.x > 1.5 && lighting.optics0.x > 0.0) {
        return finish_render_style(
            filmic_bokeh(sample_uv, center_distance),
            sample_uv,
            input.uv
        );
    }
    if (lighting.dof_style.x > 0.5 && lighting.optics0.x > 0.0 && lighting.optics0.w > 0.0) {
        return finish_render_style(
            cinematic_bokeh(sample_uv, center, center_distance, radius_px),
            sample_uv,
            input.uv
        );
    }
    if (radius_px < 0.35) {
        return finish_render_style(center, sample_uv, input.uv);
    }

    let offsets = array<vec2<f32>, 12>(
        vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0),
        vec2<f32>(0.707, 0.707), vec2<f32>(-0.707, 0.707),
        vec2<f32>(0.707, -0.707), vec2<f32>(-0.707, -0.707),
        vec2<f32>(0.383, 0.924), vec2<f32>(-0.924, 0.383),
        vec2<f32>(0.924, -0.383), vec2<f32>(-0.383, -0.924)
    );
    var accumulated = center;
    var total_weight = 1.0;
    for (var i = 0u; i < 12u; i = i + 1u) {
        let sample_uv = clamp(
            sample_uv + offsets[i] * radius_px / dimensions,
            vec2<f32>(0.0001),
            vec2<f32>(0.9999)
        );
        let sample_pixel = vec2<i32>(sample_uv * dimensions);
        let sample_distance = view_distance(textureLoad(scene_depth, sample_pixel, 0));
        let sample_coc = circle_of_confusion(sample_distance, dimensions.y);
        // Depth-aware weights keep foreground silhouettes from bleeding into a focused subject.
        let separation = abs(sample_distance - center_distance) /
            max(min(sample_distance, center_distance), 0.1);
        let depth_weight = exp(-separation * 6.0);
        let coc_weight = smoothstep(0.0, 1.0, sample_coc + radius_px);
        let weight = max(depth_weight * coc_weight, 0.001);
        accumulated += textureSampleLevel(scene_color, scene_sampler, sample_uv, 0.0) * weight;
        total_weight += weight;
    }
    return finish_render_style(accumulated / total_weight, sample_uv, input.uv);
}

// Motion blur runs after temporal resolve so shutter samples never feed back
// into the next frame's clean TAA history.
@fragment
fn fs_motion_blur(input: VertexOut) -> @location(0) vec4<f32> {
    let radiance = antialiased_color(input.uv);
    return preview_sharpen(preview_motion_blur(radiance, input.uv), input.uv);
}
