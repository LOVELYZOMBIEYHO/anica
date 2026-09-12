// =========================================
// =========================================
// crates/motionloom/src/world/render/lighting.rs

//! Resolve authored lights, environment controls, color management and style selectors.

use super::*;

impl GpuWorldLighting {
    pub(super) fn fallback(camera: PerspectiveCameraView) -> Self {
        let mut mip = Vec::with_capacity(8);
        for value in [0.18, 0.19, 0.22, 1.0] {
            mip.extend_from_slice(&f16::from_f32(value).to_bits().to_ne_bytes());
        }
        Self {
            params: GpuWorldLightingParams::from_world(&WorldLighting::default(), camera, false, 1),
            environment: Arc::new(WorldEnvironmentImage {
                width: 1,
                height: 1,
                mip_bytes: vec![mip],
                signature: 0,
            }),
            frame_index: 0,
            temporal_jitter: false,
        }
    }
}

impl GpuWorldLightingParams {
    pub(super) fn from_world(
        lighting: &WorldLighting,
        camera: PerspectiveCameraView,
        has_environment: bool,
        mip_count: usize,
    ) -> Self {
        let environment = lighting.environment.as_ref();
        let tone_mapping = match lighting.color_management.tone_mapping.as_str() {
            "none" => 0.0,
            "reinhard" => 1.0,
            "filmic_aces_v1" => 3.0,
            _ => 2.0,
        };
        let fog = lighting.atmosphere_fog.as_ref();
        let fog_mode = fog.map_or(0.0, |fog| match fog.mode.as_str() {
            "linear" => 1.0,
            "exp" => 2.0,
            "height" => 3.0,
            _ => 0.0,
        });
        let mut lights = [[0.0; 16]; 8];
        for (output, light) in lights.iter_mut().zip(lighting.lights.iter().take(8)) {
            let kind = match light.kind {
                WorldLightKind::Directional => 0.0,
                WorldLightKind::Point => 1.0,
                WorldLightKind::Spot => 2.0,
                WorldLightKind::RectArea => 3.0,
            };
            output[0..4].copy_from_slice(&[
                light.position[0],
                light.position[1],
                light.position[2],
                kind,
            ]);
            output[4..8].copy_from_slice(&[
                light.direction[0],
                light.direction[1],
                light.direction[2],
                light.range,
            ]);
            output[8..12].copy_from_slice(&[
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ]);
            output[12..16].copy_from_slice(&[
                light.inner_cone_degrees.to_radians().cos(),
                light.outer_cone_degrees.to_radians().cos(),
                light.width,
                light.height,
            ]);
        }
        let shadow_light = lighting.lights.iter().find(|light| light.cast_shadow);
        let (shadow0, shadow1, shadow2, shadow3, shadow_strength) =
            if let Some(light) = shadow_light {
                let forward = normalize3(light.direction);
                let reference_up = if forward[1].abs() > 0.95 {
                    [0.0, 0.0, 1.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let right = normalize3(cross3(reference_up, forward));
                let up = normalize3(cross3(forward, right));
                (
                    [right[0], right[1], right[2], 14.0],
                    [up[0], up[1], up[2], 14.0],
                    [forward[0], forward[1], forward[2], 28.0],
                    [0.0, 2.0, 0.0, 0.0018],
                    light.shadow_strength,
                )
            } else {
                (
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0; 4],
                    0.0,
                )
            };
        let universal = lighting
            .render_style
            .as_ref()
            .map(|s| s.universal.clone())
            .unwrap_or_default();
        Self {
            universal_color: [
                universal.tint[0],
                universal.tint[1],
                universal.tint[2],
                universal.tint_strength,
            ],
            universal_tone: [
                universal.exposure,
                universal.contrast,
                universal.saturation,
                universal.enabled as u8 as f32,
            ],
            universal_shadow: [
                universal.shadow_color[0],
                universal.shadow_color[1],
                universal.shadow_color[2],
                universal.tone_strength,
            ],
            // The preset selector is separate from legacy surface selectors.
            universal_highlight: [
                universal.highlight_color[0],
                universal.highlight_color[1],
                universal.highlight_color[2],
                lighting
                    .render_style
                    .as_ref()
                    .is_some_and(|s| s.shading == "ink_wash_soft_v1") as u8 as f32,
            ],
            // Disabled styles use exact legacy-neutral multipliers.
            cel0: lighting
                .render_style
                .as_ref()
                .map_or([0.5, 0.025, 0.0, 0.0], |s| {
                    [
                        s.cel.shadow_threshold,
                        s.cel.shadow_feather,
                        s.cel.outline_width,
                        1.0,
                    ]
                }),
            cel1: lighting
                .render_style
                .as_ref()
                .map_or([0.4, 0.37, 0.5, 0.0], |s| {
                    [
                        s.cel.shadow_color[0],
                        s.cel.shadow_color[1],
                        s.cel.shadow_color[2],
                        0.0,
                    ]
                }),
            cel2: lighting.render_style.as_ref().map_or([0.0; 4], |s| {
                [
                    s.cel.outline_color[0],
                    s.cel.outline_color[1],
                    s.cel.outline_color[2],
                    0.0,
                ]
            }),
            surface0: lighting
                .render_style
                .as_ref()
                .map_or([0.0, 3.0, 0.0, 0.0], |s| {
                    [
                        match s.shading.as_str() {
                            "stylized" => 1.0,
                            "toon" => 2.0,
                            "clay" => 3.0,
                            "filmic_physical_v1" => -1.0,
                            "cel" => 4.0,
                            _ => 0.0,
                        },
                        s.shading_steps as f32,
                        s.diffuse_wrap,
                        s.rim_light,
                    ]
                }),
            surface1: lighting
                .render_style
                .as_ref()
                .map_or([3.0, 1.0, 0.0, 1.0], |s| {
                    [
                        s.rim_power,
                        s.specular,
                        s.roughness_bias,
                        s.post.saturation.unwrap_or(1.0),
                    ]
                }),
            surface2: lighting.render_style.as_ref().map_or([1.0; 4], |s| {
                [
                    s.ambient_color[0],
                    s.ambient_color[1],
                    s.ambient_color[2],
                    s.ambient_intensity,
                ]
            }),
            surface3: lighting
                .render_style
                .as_ref()
                .map_or([1.0, 0.0, 1536.0, 0.0], |s| {
                    [
                        s.surface_saturation,
                        s.hard_shadows as u8 as f32,
                        1536.0,
                        0.0,
                    ]
                }),
            // x intensity, y rotation, z mip count, w environment present.
            environment0: [
                environment.map_or(1.0, |env| env.intensity),
                environment.map_or(0.0, |env| env.rotation_y_degrees.to_radians()),
                mip_count.saturating_sub(1) as f32,
                has_environment as u8 as f32,
            ],
            // x background, y blur, z diffuse, w specular.
            environment1: [
                environment.map_or(0.0, |env| env.background_intensity),
                environment.map_or(0.0, |env| env.background_blur),
                environment.map_or(1.0, |env| env.diffuse_intensity),
                environment.map_or(1.0, |env| env.specular_intensity),
            ],
            // x visible, y light count, z AO, w AO radius.
            environment2: [
                environment.is_some_and(|env| env.visible) as u8 as f32,
                lighting.lights.len().min(8) as f32,
                lighting.ao_intensity,
                lighting.ao_radius,
            ],
            // x exposure, y white balance, z contrast, w tone mapper.
            color0: [
                lighting.color_management.exposure,
                lighting.color_management.white_balance_kelvin,
                lighting.color_management.contrast,
                tone_mapping,
            ],
            color1: [
                lighting.contact_shadow_intensity,
                lighting.contact_shadow_distance,
                lighting.contact_shadow_softness,
                shadow_strength,
            ],
            fog0: [
                fog_mode,
                fog.map_or(0.0, |value| value.density),
                fog.map_or(0.0, |value| value.start),
                fog.map_or(100.0, |value| value.end),
            ],
            fog1: [
                fog.map_or(1.0, |value| value.color[0]),
                fog.map_or(1.0, |value| value.color[1]),
                fog.map_or(1.0, |value| value.color[2]),
                fog.map_or(0.0, |value| value.base_height),
            ],
            fog2: [
                fog.map_or(0.0, |value| value.height_falloff),
                fog.map_or(0.0, |value| value.scattering),
                fog.is_some_and(|value| value.affect_sky) as u8 as f32,
                fog.is_some() as u8 as f32,
            ],
            fog3: [
                fog.and_then(|value| value.bounds_min)
                    .map_or(0.0, |value| value[0]),
                fog.and_then(|value| value.bounds_min)
                    .map_or(0.0, |value| value[1]),
                fog.and_then(|value| value.bounds_min)
                    .map_or(0.0, |value| value[2]),
                fog.is_some_and(|value| value.bounds_min.is_some() && value.bounds_max.is_some())
                    as u8 as f32,
            ],
            fog4: [
                fog.and_then(|value| value.bounds_max)
                    .map_or(0.0, |value| value[0]),
                fog.and_then(|value| value.bounds_max)
                    .map_or(0.0, |value| value[1]),
                fog.and_then(|value| value.bounds_max)
                    .map_or(0.0, |value| value[2]),
                fog.map_or(0.0, |value| value.edge_feather),
            ],
            optics0: camera.optics,
            render_compat: [
                lighting
                    .lights
                    .iter()
                    .position(|l| l.cast_shadow)
                    .map_or(-1.0, |i| i as f32),
                0.0,
                0.0,
                0.0,
            ],
            dof_style: lighting
                .render_style
                .as_ref()
                .and_then(|s| s.depth_of_field.as_ref())
                .map(|d| {
                    let samples = match d.quality.as_deref().unwrap_or("balanced") {
                        "preview" => 32.0,
                        "high" => 192.0,
                        _ => 96.0,
                    };
                    if d.preset == "filmic_bokeh_v1" {
                        [
                            2.0,
                            41.0,
                            d.aperture.unwrap_or(0.00032),
                            d.max_blur.unwrap_or(0.0025),
                        ]
                    } else {
                        [1.0, samples, 0.0, 0.0]
                    }
                })
                .unwrap_or([0.0; 4]),
            camera0: [camera.eye[0], camera.eye[1], camera.eye[2], camera.focal_px],
            camera1: [
                camera.right[0],
                camera.right[1],
                camera.right[2],
                camera.near,
            ],
            camera2: [camera.up[0], camera.up[1], camera.up[2], camera.far],
            camera3: [
                camera.forward[0],
                camera.forward[1],
                camera.forward[2],
                camera.aspect,
            ],
            previous_camera0: [camera.eye[0], camera.eye[1], camera.eye[2], camera.focal_px],
            previous_camera1: [
                camera.right[0],
                camera.right[1],
                camera.right[2],
                camera.near,
            ],
            previous_camera2: [camera.up[0], camera.up[1], camera.up[2], camera.far],
            previous_camera3: [
                camera.forward[0],
                camera.forward[1],
                camera.forward[2],
                camera.aspect,
            ],
            preview0: [0.0; 4],
            preview1: [0.0; 4],
            shadow0,
            shadow1,
            shadow2,
            shadow3,
            lights,
        }
    }
}
