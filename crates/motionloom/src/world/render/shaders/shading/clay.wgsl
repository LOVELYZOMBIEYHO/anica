// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/shading/clay.wgsl

fn clay_base_color() -> vec3<f32> {
    return vec3<f32>(0.55, 0.48, 0.40);
}

fn clay_metallic() -> f32 {
    return 0.0;
}

fn clay_roughness() -> f32 {
    return 0.85;
}
