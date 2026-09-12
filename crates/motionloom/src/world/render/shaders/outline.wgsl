// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/outline.wgsl

@vertex
fn vs_main(input: VertexIn, @builtin(instance_index) instance_id: u32) -> VertexOut {
    return cel_shared_vertex(input, instance_id);
}
// Reuse the exact skinned vertex path, expanding only in projected screen space.
@vertex
fn vs_outline(input: VertexIn, @builtin(instance_index) instance_id: u32) -> VertexOut {
    var smooth_input = input;
    smooth_input.normal = input.outline_normal;
    var out = cel_shared_vertex(smooth_input, instance_id);
    let projected = vec2<f32>(dot(out.normal, params.camera1.xyz), dot(out.normal, params.camera2.xyz));
    let magnitude = max(length(projected), 0.0001);
    let dims = textureDimensions(cel_texture);
    let coord = clamp(vec2<i32>(out.uv * vec2<f32>(dims)), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let mask = textureLoad(cel_texture, coord, 0).r;
    let width = select(lighting.cel0.z, params.cel_material0.x, params.cel_material0.x >= 0.0) * lighting.cel0.w * mask;
    out.pos.x += projected.x / magnitude * width * 2.0 / params.canvas.x * out.pos.w;
    out.pos.y += projected.y / magnitude * width * 2.0 / params.canvas.y * out.pos.w;
    return out;
}

@fragment
fn fs_outline(input: VertexOut) -> @location(0) vec4<f32> {
    params = instance_params[input.instance_id];
    let alpha = textureSample(actor_texture, actor_sampler, input.uv).a * input.color.a * params.material4.a * params.style.x;
    // Opaque materials ignore texture alpha, including legacy solid-color fallbacks.
    // Masked materials retain cutouts; actor fades still suppress solid outlines.
    let masked_out = params.material7.w > 0.0 && alpha < max(params.material7.w, 0.99);
    if (input.hidden_weight > 0.5 || masked_out || params.style.x < 0.99
        || dot(input.world_position - params.camera0.xyz, params.camera3.xyz) <= params.camera1.w) { discard; }
    return vec4<f32>(inverse_display_curve(lighting.cel2.rgb), 1.0);
}
