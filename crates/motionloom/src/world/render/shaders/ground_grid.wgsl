// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/ground_grid.wgsl

struct GridParams {
    canvas: vec4<f32>,
    camera0: vec4<f32>,
    camera1: vec4<f32>,
    camera2: vec4<f32>,
    camera3: vec4<f32>,
    options: vec4<f32>,
    _pad0: vec4<f32>,
    _pad1: vec4<f32>,
};

struct VertexIn {
    @location(0) offset: vec3<f32>,
};

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: GridParams;

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let center = vec3<f32>(params.camera0.x, 0.0, params.camera0.z);
    let world = center + input.offset;
    let right = params.camera1.xyz;
    let up = params.camera2.xyz;
    let forward = params.camera3.xyz;
    let rel = world - params.camera0.xyz;
    let view_x = dot(rel, right);
    let view_y = dot(rel, up);
    let view_z = dot(rel, forward);
    let near = params.camera1.w;
    let far = max(params.camera2.w, near + 0.001);

    var out: VertexOut;
    out.world_pos = world;
    let clip_x = (2.0 * params.canvas.z / params.canvas.x - 1.0) * view_z + 2.0 * view_x * params.camera0.w / params.canvas.x;
    let clip_y = (1.0 - 2.0 * params.canvas.w / params.canvas.y) * view_z + 2.0 * view_y * params.camera0.w / params.canvas.y;
    let clip_z = near * (far - view_z) / (far - near);
    out.pos = vec4<f32>(clip_x, clip_y, clip_z, view_z);
    return out;
}

fn grid_alpha(coord: vec2<f32>, scale: f32) -> f32 {
    let scaled = coord / scale;
    let derivative = max(fwidth(scaled), vec2<f32>(0.000001, 0.000001));
    let grid = abs(fract(scaled - 0.5) - 0.5) / derivative;
    let line_val = min(grid.x, grid.y);
    return 1.0 - min(line_val, 1.0);
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let grid_size = max(params.options.w, 0.0001);
    let coord = input.world_pos.xz / grid_size;
    let debug_mode = params.options.y < 0.0;

    var fine_weight: f32 = 0.45;
    var coarse_weight: f32 = 1.00;
    var axis_width: f32 = grid_size * 0.04;
    var opacity: f32 = params.options.x;
    var fade: f32 = 1.0;
    if (!debug_mode) {
        // Must be per-fragment distance (not interpolated vertex distance),
        // otherwise the whole grid fades out when plane vertices are far away.
        let dist = distance(input.world_pos, params.camera0.xyz);
        fade = 1.0 - smoothstep(params.options.y, params.options.z, dist);
    } else {
        // Debug grid mode: strong, thick, high-contrast lines with no fade.
        fine_weight = 1.10;
        coarse_weight = 1.25;
        axis_width = grid_size * 0.10;
        opacity = 1.0;
        fade = 1.0;
    }

    let fine = grid_alpha(coord, 1.0) * fine_weight;
    let coarse = grid_alpha(coord, 10.0) * coarse_weight;
    let axis_x = 1.0 - smoothstep(0.0, axis_width, abs(input.world_pos.z));
    let axis_z = 1.0 - smoothstep(0.0, axis_width, abs(input.world_pos.x));
    let line_alpha = max(max(fine, coarse), max(axis_x, axis_z));

    let alpha = min(line_alpha, 1.0) * fade * opacity;
    if (alpha <= 0.001) {
        discard;
    }

    var base_color = mix(vec3<f32>(0.50, 0.54, 0.60), vec3<f32>(0.86, 0.89, 0.94), coarse);
    if (debug_mode) {
        base_color = mix(vec3<f32>(0.10, 0.12, 0.18), vec3<f32>(0.98, 0.98, 1.00), coarse);
    }
    let x_axis_color = vec3<f32>(0.95, 0.28, 0.28);
    let z_axis_color = vec3<f32>(0.30, 0.86, 0.42);
    var color = base_color;
    color = mix(color, x_axis_color, axis_x);
    color = mix(color, z_axis_color, axis_z);
    return vec4<f32>(color, alpha);
}
