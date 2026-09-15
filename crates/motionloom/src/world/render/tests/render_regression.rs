// =========================================
// =========================================
// crates/motionloom/src/world/render/tests/render_regression.rs

//! Locks authored shading names to the numeric selector consumed by WGSL.

#[test]
fn every_render_style_keeps_its_shader_selector() {
    let expected = [
        ("physical", 0.0),
        ("stylized", 1.0),
        ("toon", 2.0),
        ("clay", 3.0),
        ("cel", 4.0),
    ];
    for (name, selector) in expected {
        let lighting = crate::world::WorldLighting {
            render_style: Some(crate::render_style::ResolvedSceneRenderStyle {
                anti_aliasing: None,
                depth_of_field: None,
                universal: Default::default(),
                cel: Default::default(),
                scene_id: "test".into(),
                style_id: None,
                shading: name.into(),
                shading_steps: 3,
                diffuse_wrap: 0.0,
                rim_light: 0.0,
                rim_power: 3.0,
                specular: 1.0,
                roughness_bias: 0.0,
                surface_saturation: 1.0,
                ambient_intensity: 1.0,
                ambient_color: [1.0; 3],
                hard_shadows: false,
                lighting_preset: None,
                post: Default::default(),
                overrides: Vec::new(),
            }),
            ..Default::default()
        };
        let params = super::super::GpuWorldLightingParams::from_world(
            &lighting,
            super::super::PerspectiveCameraView {
                eye: [0.0; 3],
                right: [1.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                forward: [0.0, 0.0, -1.0],
                focal_px: 1.0,
                near: 0.01,
                far: 100.0,
                aspect: 1.0,
                optics: [0.0; 4],
            },
            false,
            1,
        );
        assert_eq!(params.surface0[0], selector, "selector for {name}");
    }
}
