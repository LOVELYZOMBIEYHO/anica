// =========================================
// =========================================
// crates/motionloom/src/scene/backend/gpu/shaders/spectral_energy.wgsl

struct SpectralEnergyParams {
    canvas: vec4<f32>,
    field: vec4<f32>,
    timing: vec4<f32>,
};

@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: SpectralEnergyParams;

fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    return fract(q.x * q.y * (q.x + q.y + 45.32));
}

fn particle_hash(position: vec2<f32>, salt: f32) -> f32 {
    // A floating-point permutation is used instead of a u32 overflow hash.
    // This stays portable across browser WebGPU implementations while the
    // irrational coordinate mix prevents the visible row/column lattice of
    // sampling a simple float hash at exact pixel coordinates.
    var p = fract(
        position * vec2<f32>(0.1031, 0.1030)
            + vec2<f32>(salt * 0.000173, salt * 0.000119)
    );
    p = p + dot(p, p.yx + vec2<f32>(33.33, 17.17));
    return fract((p.x + p.y) * (p.x * 1.731 + p.y * 2.137));
}

fn value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let fraction = fract(p);
    let smooth_fraction = fraction * fraction * (vec2<f32>(3.0) - 2.0 * fraction);
    let a = hash21(cell);
    let b = hash21(cell + vec2<f32>(1.0, 0.0));
    let c = hash21(cell + vec2<f32>(0.0, 1.0));
    let d = hash21(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, smooth_fraction.x), mix(c, d, smooth_fraction.x), smooth_fraction.y);
}

fn fbm(position: vec2<f32>) -> f32 {
    var p = position;
    var amplitude = 0.54;
    var result = 0.0;
    for (var octave = 0; octave < 5; octave = octave + 1) {
        result = result + value_noise(p) * amplitude;
        p = p * 2.03 + vec2<f32>(13.1, 7.7);
        amplitude = amplitude * 0.49;
    }
    return result;
}

fn gaussian_band(y: f32, center: f32, width: f32) -> f32 {
    let distance = abs(y - center) / max(width, 0.0001);
    return exp(-(distance * distance));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let width = params.canvas.x;
    let height = params.canvas.y;
    if (x >= u32(width) || y >= u32(height)) {
        return;
    }

    let pixel = vec2<i32>(i32(x), i32(y));
    let base = textureLoad(base_tex, pixel, 0);
    let effect_mix = clamp(params.canvas.z, 0.0, 1.0);
    if (effect_mix <= 0.0001) {
        textureStore(out_tex, pixel, base);
        return;
    }

    let intensity = max(params.canvas.w, 0.0);
    let density = max(params.field.x, 0.05);
    let drift = params.field.y;
    let seed = params.field.z;
    let time = params.field.w;
    let mode = params.timing.x;
    let uv = (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5))
        / max(vec2<f32>(width, height), vec2<f32>(1.0));
    let aspect = width / max(height, 1.0);
    let p = vec2<f32>(uv.x * aspect, uv.y);

    let slow_motion = drift + time * 0.055;
    let upper_curve = 0.245
        + 0.105 * sin(uv.x * 5.7 + slow_motion * 0.8)
        + 0.030 * sin(uv.x * 14.0 - 0.4);
    let center_curve = 0.495
        + 0.057 * sin(uv.x * 8.8 - 0.9 + slow_motion)
        + 0.022 * sin(uv.x * 21.0);
    let lower_curve = 0.735
        + 0.095 * sin(uv.x * 4.6 + 1.7 - slow_motion * 0.5)
        + 0.035 * sin(uv.x * 13.2);

    let cyan_upper = gaussian_band(uv.y, upper_curve, 0.205);
    let cyan_lower = gaussian_band(uv.y, lower_curve, 0.235);
    let magenta_center = gaussian_band(uv.y, center_curve, 0.090);
    let magenta_lower = gaussian_band(uv.y, lower_curve - 0.055, 0.090);

    let broad_noise = fbm(
        p * vec2<f32>(2.4, 4.4) * density
            + vec2<f32>(seed * 0.017 + slow_motion, seed * 0.011)
    );
    let flow_noise = fbm(
        p * vec2<f32>(7.3, 9.1) * density
            + vec2<f32>(seed * 0.031 - slow_motion * 0.6, seed * 0.023)
    );
    let pixel_noise = hash21(
        vec2<f32>(f32(x), f32(y)) * density
            + vec2<f32>(seed * 31.7, seed * 17.3)
    );
    let sparkle = pixel_noise * pixel_noise * pixel_noise;
    let grain = 0.15 + broad_noise * 0.72 + flow_noise * 0.42 + sparkle * 1.65;

    let cyan = vec3<f32>(0.00, 0.82, 1.00);
    let ice = vec3<f32>(0.30, 0.98, 1.00);
    let magenta = vec3<f32>(1.00, 0.00, 0.53);
    let electric_blue = vec3<f32>(0.00, 0.25, 0.72);

    var field = vec3<f32>(0.001, 0.006, 0.014);
    field = field
        + cyan_upper * mix(electric_blue, cyan, broad_noise) * grain
        + cyan_lower * mix(cyan, ice, flow_noise) * grain * 1.15
        + magenta_center * magenta * (0.24 + grain * 1.10)
        + magenta_lower * magenta * (0.12 + grain * 0.82);

    let center_burst = gaussian_band(uv.y, 0.69 + 0.03 * sin(uv.x * 9.0), 0.16)
        * gaussian_band(uv.x, 0.47, 0.29);
    field = field + center_burst * ice * (0.42 + grain * 0.78);

    let upper_void = gaussian_band(uv.x, 0.66, 0.31)
        * gaussian_band(uv.y, 0.15, 0.18);
    field = field * (1.0 - upper_void * 0.86);

    // Granular light-field mode keeps the same Effect and pipeline, but changes
    // the field synthesis from smooth waveform bands to a directional cloud of
    // light. `drift` is interpreted as a normalized right-to-left sweep while
    // the default mode intentionally retains the legacy waveform output.
    if (mode >= 0.5) {
        let signed_broad = broad_noise * 2.0 - 1.0;
        let signed_flow = flow_noise * 2.0 - 1.0;
        let sweep = clamp(drift, 0.0, 1.0);
        let emitter_position = vec2<f32>(1.035, 0.455);
        let distance_from_emitter = clamp(emitter_position.x - uv.x, 0.0, 1.20);

        // The advancing edge starts just outside the right side of the frame
        // and travels past the left edge. Noise breaks up the edge into dust
        // rather than exposing a rectangular wipe.
        let front_x = 1.08 - sweep * 1.31;
        let edge_noise = signed_broad * 0.080 + signed_flow * 0.030;
        let swept_region = smoothstep(
            front_x - 0.075 + edge_noise,
            front_x + 0.045 + edge_noise,
            uv.x,
        );
        let advancing_front = gaussian_band(
            uv.x,
            front_x - edge_noise * 0.55,
            0.090 + distance_from_emitter * 0.040,
        );

        // Rays originate on the right and expand towards the left. Their axes
        // are linear rather than periodic, so there is no waveform silhouette.
        let ray_warp = signed_broad * (0.012 + distance_from_emitter * 0.036)
            + signed_flow * 0.010;
        let upper_axis = emitter_position.y
            - distance_from_emitter * 0.205
            + ray_warp;
        let center_axis = emitter_position.y
            + distance_from_emitter * 0.018
            + ray_warp * 0.44;
        let lower_axis = emitter_position.y
            + distance_from_emitter * 0.245
            + ray_warp * 1.18;
        // The source is compact on the right, while every lobe opens into a
        // broad cone towards the left. This is deliberately much wider than
        // the legacy waveform bands above.
        let upper_width = 0.038 + distance_from_emitter * 0.255;
        let center_width = 0.026 + distance_from_emitter * 0.125;
        let lower_width = 0.045 + distance_from_emitter * 0.285;
        let upper_ray = gaussian_band(uv.y, upper_axis, upper_width);
        let center_ray = gaussian_band(uv.y, center_axis, center_width);
        let lower_ray = gaussian_band(uv.y, lower_axis, lower_width);

        // Independent full-resolution samples form an irregular photographic
        // particle distribution without creating a visible waveform lattice.
        let frame_seed = floor(max(time, 0.0) * 12.0) * 13.0
            + max(seed, 0.0) * 4093.0;
        let pixel_position = vec2<f32>(f32(x), f32(y));
        let fine_a = particle_hash(pixel_position, frame_seed + 17.0);
        let fine_b = particle_hash(pixel_position.yx + vec2<f32>(19.0, 47.0), frame_seed + 113.0);
        let particle = pow(clamp(fine_a * 0.68 + fine_b * 0.48, 0.0, 1.0), 2.15);
        let sparse_particle = smoothstep(0.31, 0.94, particle);
        let dust = 0.20 + particle * 1.52 + sparse_particle * 1.32;
        let cloud = 0.34 + broad_noise * 0.74 + flow_noise * 0.24;
        let front_dust = 0.60 + advancing_front * (1.12 + particle * 1.25);
        let emitter = gaussian_band(uv.x, emitter_position.x, 0.13);
        let directional_energy = swept_region
            * (0.72 + distance_from_emitter * 0.46 + emitter * 0.72)
            * front_dust;

        var granular_field = vec3<f32>(0.001, 0.007, 0.017);
        granular_field = granular_field
            + upper_ray
                * mix(electric_blue, cyan, clamp(broad_noise * 1.25, 0.0, 1.0))
                * dust
                * cloud
                * directional_energy
                * 1.18
            + center_ray
                * magenta
                * dust
                * (0.46 + flow_noise * 1.18)
                * directional_energy
                * 1.20
            + lower_ray
                * mix(cyan, ice, flow_noise)
                * dust
                * cloud
                * directional_energy
                * 1.30;

        // The plume widens continuously from the right emitter. The moving
        // front receives extra light, producing a readable right-to-left sweep.
        let plume_y = emitter_position.y
            + distance_from_emitter * 0.055
            + signed_broad * (0.035 + distance_from_emitter * 0.070);
        let broad_plume = gaussian_band(
            uv.y,
            plume_y,
            0.080 + distance_from_emitter * 0.335,
        );
        let front_plume = broad_plume * advancing_front;

        // These continuous lobes provide the photographic body of the light
        // field. Fine particles are added on top instead of being responsible
        // for all illumination, preventing the result from reading as a dark
        // waveform with grain.
        let upper_volume = gaussian_band(
            uv.y,
            emitter_position.y - distance_from_emitter * 0.215 + ray_warp * 0.55,
            0.055 + distance_from_emitter * 0.285,
        );
        let magenta_volume = gaussian_band(
            uv.y,
            emitter_position.y + distance_from_emitter * 0.060 + ray_warp * 0.35,
            0.042 + distance_from_emitter * 0.145,
        );
        let lower_volume = gaussian_band(
            uv.y,
            emitter_position.y + distance_from_emitter * 0.285 + ray_warp * 0.72,
            0.070 + distance_from_emitter * 0.315,
        );
        let volume_noise = 0.48 + broad_noise * 0.52 + particle * 0.42;
        let volume_energy = swept_region
            * (0.62 + distance_from_emitter * 0.72)
            * (0.68 + advancing_front * 1.05);
        granular_field = granular_field
            + upper_volume
                * mix(electric_blue, cyan, clamp(broad_noise * 1.12, 0.0, 1.0))
                * volume_noise
                * volume_energy
                * 1.10
            + magenta_volume
                * magenta
                * (0.42 + flow_noise * 0.68 + particle * 0.34)
                * volume_energy
                * 1.20
            + lower_volume
                * mix(cyan, ice, clamp(flow_noise * 1.08, 0.0, 1.0))
                * volume_noise
                * volume_energy
                * 1.15;

        granular_field = granular_field
            + broad_plume
                * mix(magenta, ice, clamp(uv.y * 1.08, 0.0, 1.0))
                * dust
                * cloud
                * directional_energy
                * 0.88
            + front_plume
                * mix(cyan, ice, clamp(uv.y * 1.12, 0.0, 1.0))
                * dust
                * (0.52 + cloud * 0.96)
                * 1.35;

        let hot_core = gaussian_band(uv.y, lower_axis - 0.025, lower_width * 0.66)
            * advancing_front
            * broad_plume;
        granular_field = granular_field
            + hot_core * ice * dust * (0.62 + broad_noise * 0.88);

        // Preserve a dark upper pocket behind the expanding field. This keeps
        // the result dimensional instead of filling the frame like a gradient.
        let upper_pocket = gaussian_band(uv.x, 0.50, 0.36)
            * gaussian_band(uv.y, 0.12, 0.23)
            * swept_region;
        field = granular_field * (1.0 - upper_pocket * 0.88);
    }
    field = min(field * intensity, vec3<f32>(1.45));

    // Dark source pixels become the generated energy plate. Existing bright
    // title and logo pixels stay crisp and are composited above the field.
    let foreground_luma = max(base.r, max(base.g, base.b));
    let foreground = smoothstep(0.095, 0.30, foreground_luma);
    let composed = mix(field, base.rgb, foreground);
    let rgb = mix(base.rgb, composed, effect_mix);
    // The Graph background may be composited only after the Scene texture is
    // produced. Generated energy is visible content and therefore needs to
    // contribute coverage instead of inheriting a transparent Scene alpha.
    let output_alpha = max(base.a, effect_mix);
    textureStore(
        out_tex,
        pixel,
        vec4<f32>(
            clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
            output_alpha
        )
    );
}
