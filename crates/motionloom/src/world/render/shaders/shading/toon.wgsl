// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/shading/toon.wgsl

fn toon_intensity(intensity: f32) -> f32 {
    let steps = max(lighting.surface0.y, 2.0);
    return floor(intensity * (steps - 1.0) + 0.5) / (steps - 1.0);
}
