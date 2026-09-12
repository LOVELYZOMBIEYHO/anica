// =========================================
// =========================================
// crates/motionloom/src/world/render/tests/shader_validation.rs

//! Validates the assembled shader and the presence of each shading module.

#[test]
fn assembled_world_shader_contains_every_shading_mode() {
    let source = super::super::WGPU_WORLD_SHADER.as_str();
    for function in [
        "fn shade_physical",
        "fn shade_stylized",
        "fn toon_intensity",
        "fn shade_cel",
        "fn clay_base_color",
    ] {
        assert!(
            source.contains(function),
            "missing shader function {function}"
        );
    }
    let module = wgpu::naga::front::wgsl::parse_str(source).expect("assembled WGSL must parse");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("assembled WGSL must validate");
}
