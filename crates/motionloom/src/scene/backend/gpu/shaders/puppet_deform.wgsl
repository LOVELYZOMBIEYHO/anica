struct PuppetDeformParams {
    canvas: vec4<f32>,
};

struct PuppetDeformTriangle {
    src0: vec4<f32>,
    src1: vec4<f32>,
    src2: vec4<f32>,
    dst0: vec4<f32>,
    dst1: vec4<f32>,
    dst2: vec4<f32>,
};

struct PuppetDeformTriangleBuffer {
    items: array<PuppetDeformTriangle>,
};

@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: PuppetDeformParams;
@group(0) @binding(4) var<storage, read> triangle_buffer: PuppetDeformTriangleBuffer;

fn barycentric(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let den = v0.x * v1.y - v1.x * v0.y;
    if (abs(den) <= 0.000001) {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let v = (v2.x * v1.y - v1.x * v2.y) / den;
    let w = (v0.x * v2.y - v2.x * v0.y) / den;
    let u = 1.0 - v - w;
    return vec3<f32>(u, v, w);
}

fn inside_tri(b: vec3<f32>) -> bool {
    return b.x >= -0.0005 && b.y >= -0.0005 && b.z >= -0.0005;
}

fn segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let length_squared = dot(ab, ab);
    if (length_squared <= 0.000001) {
        return distance(p, a);
    }
    let t = clamp(dot(p - a, ab) / length_squared, 0.0, 1.0);
    return distance(p, a + ab * t);
}

fn inside_source_tri_with_fringe(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
    clear_fringe_px: f32,
) -> bool {
    if (clear_fringe_px <= 0.0) {
        return false;
    }
    if (inside_tri(barycentric(p, a, b, c))) {
        return true;
    }
    // Match the CPU renderer and remove antialiased source outlines that sit
    // just outside a local Puppet mesh.
    return min(
        segment_distance(p, a, b),
        min(segment_distance(p, b, c), segment_distance(p, c, a)),
    ) <= clear_fringe_px;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(params.canvas.x);
    let height = u32(params.canvas.y);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    var out_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let tri_count = u32(params.canvas.z);
    var painted_destination = false;
    for (var ix = 0u; ix < tri_count; ix = ix + 1u) {
        let tri = triangle_buffer.items[ix];
        let dst0 = tri.dst0.xy;
        let dst1 = tri.dst1.xy;
        let dst2 = tri.dst2.xy;
        let b = barycentric(pixel, dst0, dst1, dst2);
        if (!inside_tri(b)) {
            continue;
        }

        let src = tri.src0.xy * b.x + tri.src1.xy * b.y + tri.src2.xy * b.z;
        if (src.x >= 0.0 && src.y >= 0.0 && src.x < params.canvas.x && src.y < params.canvas.y) {
            let uv = src / max(params.canvas.xy, vec2<f32>(1.0, 1.0));
            let sampled = textureSampleLevel(source_tex, source_sampler, uv, 0.0);
            // Match the CPU painter: transparent texels never erase an
            // already-painted overlap cap or preserved artwork.
            if (sampled.a > 0.0001) {
                out_color = sampled;
                painted_destination = true;
            }
        }
        // Match the CPU painter: later triangles may deliberately cover an
        // earlier seam patch at the same destination pixel.
    }

    if (!painted_destination && params.canvas.w >= 0.5) {
        let uv = pixel / max(params.canvas.xy, vec2<f32>(1.0, 1.0));
        out_color = textureSampleLevel(source_tex, source_sampler, uv, 0.0);
        for (var ix = 0u; ix < tri_count; ix = ix + 1u) {
            let tri = triangle_buffer.items[ix];
            let bind0 = tri.src0.zw;
            let bind1 = tri.src1.zw;
            let bind2 = tri.src2.zw;
            let moved = distance(bind0, tri.dst0.xy) > 0.01
                || distance(bind1, tri.dst1.xy) > 0.01
                || distance(bind2, tri.dst2.xy) > 0.01;
            if (inside_source_tri_with_fringe(
                pixel,
                bind0,
                bind1,
                bind2,
                select(0.0, 2.0, moved),
            )) {
                out_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                break;
            }
        }
    }

    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), out_color);
}
