// =========================================
// =========================================
// crates/motionloom/src/weaver/backend/wgpu/shaders/denoise.wgsl

// GPU denoiser for the Weaver film buffer. `denoise_seed` converts the accumulated
// film into radiance, normal, albedo and variance planes; `denoise_wavelet` runs an
// edge-avoiding a-trous wavelet pass guided by those planes. All inputs come
// from the film AOVs that the path tracer already accumulates, so no host
// library is required.

struct Film {
    sum: vec4<f32>,
    stats: vec4<f32>,
    albedo: vec4<f32>,
    normal: vec4<f32>,
    aov: vec4<f32>,
}
struct Params { v: array<vec4<f32>, 4> }
@group(0) @binding(0) var<uniform> p: Params;

@group(1) @binding(0) var<storage, read> film: array<Film>;
@group(1) @binding(1) var<storage, read_write> seed_color: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> seed_normal: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> seed_albedo: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read> src_color: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> src_normal: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> src_albedo: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> dst_color: array<vec4<f32>>;

fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

@compute @workgroup_size(8,8)
fn denoise_seed(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<u32>(p.v[0].xy);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let index = id.y * dims.x + id.x;
    let f = film[index];
    let n = max(f.sum.w, 1.0);
    let color = f.sum.xyz / n;
    // Variance of the mean, matching the convergence estimator.
    let variance = select(
        f.stats.y / max((n - 1.0) * n, 1.0),
        0.0,
        n < 2.0,
    );
    let normal = normalize(select(f.normal.xyz / n, vec3<f32>(0.0, 0.0, 1.0), dot(f.normal.xyz, f.normal.xyz) < 1e-12));
    let depth = f.normal.w / n;
    seed_color[index] = vec4<f32>(color, variance);
    seed_normal[index] = vec4<f32>(normal, depth);
    seed_albedo[index] = vec4<f32>(f.albedo.xyz / n, 0.0);
}

@compute @workgroup_size(8,8)
fn denoise_wavelet(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = vec2<i32>(p.v[0].xy);
    let step = i32(p.v[0].z);
    let phi_normal = p.v[1].x;
    let phi_depth = p.v[1].y;
    let phi_albedo = p.v[1].z;
    let phi_color = p.v[1].w;
    let coord = vec2<i32>(id.xy);
    if coord.x >= dims.x || coord.y >= dims.y { return; }
    let index = u32(coord.y) * u32(dims.x) + u32(coord.x);
    let center = src_color[index];
    let normal = src_normal[index];
    let albedo = src_albedo[index];
    var sum = center.xyz;
    var weight = 1.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let sample_coord = clamp(coord + vec2<i32>(dx, dy) * step, vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
            let sample_index = u32(sample_coord.y) * u32(dims.x) + u32(sample_coord.x);
            let c = src_color[sample_index];
            let n = src_normal[sample_index];
            let a = src_albedo[sample_index];
            let spatial = exp(-0.5 * f32(dx * dx + dy * dy));
            let normal_weight = pow(max(dot(normal.xyz, n.xyz), 0.0), phi_normal);
            let depth_scale = max(phi_depth * abs(normal.w), 0.001);
            let depth_weight = exp(-abs(normal.w - n.w) / depth_scale);
            let albedo_weight = exp(-abs(luminance(albedo.xyz) - luminance(a.xyz)) / phi_albedo);
            let variance = max(center.w + c.w, 1e-6);
            let color_delta = center.xyz - c.xyz;
            let color_weight = exp(-dot(color_delta, color_delta) / (phi_color * variance));
            let w = spatial * normal_weight * depth_weight * albedo_weight * color_weight;
            sum += c.xyz * w;
            weight += w;
        }
    }
    dst_color[index] = vec4<f32>(sum / max(weight, 1e-5), center.w);
}
