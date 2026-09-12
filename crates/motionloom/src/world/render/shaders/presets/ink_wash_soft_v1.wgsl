// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/presets/ink_wash_soft_v1.wgsl

// Soft coloured-ink recipe: depth-aware washes, five ink densities, broken contours,
// world-anchored pigment and stationary paper grain. No temporal history is required.
fn ink_hash(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn ink_noise(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(ink_hash(cell), ink_hash(cell + vec3<f32>(1,0,0)), u.x),
                   mix(ink_hash(cell + vec3<f32>(0,1,0)), ink_hash(cell + vec3<f32>(1,1,0)), u.x), u.y),
               mix(mix(ink_hash(cell + vec3<f32>(0,0,1)), ink_hash(cell + vec3<f32>(1,0,1)), u.x),
                   mix(ink_hash(cell + vec3<f32>(0,1,1)), ink_hash(cell + vec3<f32>(1,1,1)), u.x), u.y), u.z);
}

fn ink_wash_soft_v1(center: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(scene_color));
    let px = vec2<i32>(clamp(uv * dims, vec2<f32>(0), dims - 1.0));
    let depth = textureLoad(scene_depth, px, 0);
    let distance = view_distance(depth);
    let offsets = array<vec2<f32>, 4>(vec2<f32>(-1,0),vec2<f32>(1,0),vec2<f32>(0,-1),vec2<f32>(0,1));
    var neighbors: array<f32, 4>;
    var shades: array<f32, 4>;
    let weights = vec3<f32>(0.2126,0.7152,0.0722);
    var wash = center;
    var weight_sum = 1.0;
    for (var i = 0u; i < 8u; i++) {
        let sample_uv = clamp(uv + offsets[i] * 1.6 / dims, vec2<f32>(0), vec2<f32>(0.99999));
        let pixel = vec2<i32>(clamp(sample_uv * dims, vec2<f32>(0), dims - 1.0));
        let d = view_distance(textureLoad(scene_depth, pixel, 0));
        neighbors[i] = d;
        let sample = resolve_display(textureSampleLevel(scene_color, scene_sampler, sample_uv, 0.0));
        let color = sample.rgb / max(sample.a, 0.00001);
        shades[i] = dot(color, weights);
        let weight = exp(-abs(d - distance) / max(distance, 0.1) * 70.0) * sample.a * 0.6;
        wash += color * weight;
        weight_sum += weight;
    }
    wash /= weight_sum;
    // Reconstruct position with the same pixel focal length as the scene camera.
    let screen = (uv - 0.5) * dims;
    let world = lighting.camera0.xyz + lighting.camera3.xyz * distance +
        (lighting.camera1.xyz * screen.x - lighting.camera2.xyz * screen.y) * distance / max(lighting.camera0.w, 1.0);
    let pigment = ink_noise(world * 3.5);
    let paper = ink_hash(vec3<f32>(floor(uv * dims), 19.0));
    let luma = clamp(dot(wash, weights), 0.0, 1.0);
    let density = clamp(1.0 - luma + (pigment - 0.5) * 0.022 * (1.0 - luma), 0.0, 1.0);
    let bands = density * 5.0;
    let stepped = (floor(bands) + smoothstep(0.25, 0.8, fract(bands))) / 5.0;
    let depth_edge = (abs(neighbors[0] + neighbors[1] - 2.0 * distance) +
        abs(neighbors[2] + neighbors[3] - 2.0 * distance)) / max(distance, 0.1);
    let color_edge = max(abs(shades[0] - shades[1]), abs(shades[2] - shades[3]));
    // Deposit on the darker/nearer side only: bilateral outlines otherwise look embossed.
    let nearer_side = smoothstep(0.002, 0.04,
        (max(max(neighbors[0],neighbors[1]),max(neighbors[2],neighbors[3])) - distance) / max(distance, 0.1));
    let darker_side = smoothstep(0.0, 0.07,
        (shades[0]+shades[1]+shades[2]+shades[3])*0.25 - dot(center, weights));
    let contour = max(smoothstep(0.018, 0.14, depth_edge) * nearer_side,
        smoothstep(0.08, 0.32, color_edge) * darker_side);
    // Softer far contours and irregular pigment avoid a uniform cartoon outline.
    let edge = contour * (0.035 + pigment * 0.075) *
        mix(1.0, 0.08, smoothstep(16.0, 65.0, distance)) * (1.0 - 0.6 * luma);
    // Ink shapes luminance rather than replacing every material with one ink hue.
    // Preserve warm timber, golden lamps, blue distance and pink blossom pigments.
    let ink = clamp(mix(density, stepped, 0.30) + edge, 0.0, 1.0);
    let painted_luma = 1.0 - ink;
    let colored_wash = vec3<f32>(painted_luma) + (wash - vec3<f32>(luma)) * 1.08;
    // Paper shows mainly through thin, bright washes; dark forms keep their depth.
    let paper_color = vec3<f32>(0.995, 0.985, 0.962);
    let base = mix(colored_wash, paper_color, 0.065 * smoothstep(0.35, 0.95, luma));
    return clamp(base + (paper - 0.5) * 0.004, vec3<f32>(0.0), vec3<f32>(1.0));
}
