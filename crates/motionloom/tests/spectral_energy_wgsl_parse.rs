#[test]
fn spectral_energy_wgsl_parses() {
    let source = include_str!("../src/scene/backend/gpu/shaders/spectral_energy.wgsl");
    let module =
        wgpu::naga::front::wgsl::parse_str(source).expect("spectral_energy.wgsl must parse");
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("spectral_energy.wgsl must validate");
}
