// =========================================
// =========================================
// crates/motionloom/tests/render_style.rs

use motionloom::api::{parse_graph_script, resolve_scene_render_style};

#[test]
fn filmic_style_controls_roundtrip_and_reject_invalid_values() {
    let resource = "<RenderStyle id=\"t\">\n<SurfaceStyle shading=\"filmic_physical_v1\" />\n<DepthOfFieldStyle preset=\"filmic_bokeh_v1\" aperture=\"0.00032\" maxBlur=\"0.0025\" />\n<PostStyle toneMapping=\"filmic_aces_v1\" />\n</RenderStyle>";
    let source = script(resource, "renderStyle=\"t\"");
    let graph = parse_graph_script(&source).unwrap();
    let style = resolve_scene_render_style(&graph, "main").unwrap();
    assert_eq!(
        style.depth_of_field.as_ref().unwrap().aperture,
        Some(0.00032)
    );
    assert_eq!(
        style.depth_of_field.as_ref().unwrap().max_blur,
        Some(0.0025)
    );
    let restored = serde_json::from_str(&serde_json::to_string(&graph).unwrap()).unwrap();
    assert_eq!(
        style,
        resolve_scene_render_style(&restored, "main").unwrap()
    );
    for bad in [
        source.replace("0.00032", "-1"),
        source.replace("0.0025", "0.5"),
        source.replace("filmic_bokeh_v1", "cinematic_bokeh_v1"),
    ] {
        assert!(parse_graph_script(&bad).is_err());
    }
    let report: serde_json::Value = serde_json::from_str(
        &motionloom::api::motionloom_analyze_script_for_target_json(&source, "native-webgpu"),
    )
    .unwrap();
    assert_eq!(report["summary"]["errors"], 0);
    assert_eq!(report["summary"]["ignoredAttributes"], 0);
}

// Zero aperture must preserve pixels; a defocused subject must actually change.
#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires a native GPU adapter"]
fn filmic_bokeh_gpu_zero_aperture_and_disabled_camera_are_neutral() {
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let make = |aperture: f32, enabled: bool| {
            parse_graph_script(&script(
            &format!("<RenderStyle id=\"t\">\n<DepthOfFieldStyle preset=\"filmic_bokeh_v1\" aperture=\"{aperture}\" maxBlur=\"0.04\" />\n</RenderStyle>"),
            "renderStyle=\"t\"").replace("fov=\"45\"", &format!("fov=\"45\" depthOfField=\"{enabled}\" focusDistance=\"1\""))).unwrap()
        };
        let off = renderer
            .render_frame_gpu_readback(&make(0.01, false), 0)
            .await
            .unwrap();
        let zero = renderer
            .render_frame_gpu_readback(&make(0.0, true), 0)
            .await
            .unwrap();
        let blurred = renderer
            .render_frame_gpu_readback(&make(0.01, true), 0)
            .await
            .unwrap();
        let changed = |a: &image::RgbaImage, b: &image::RgbaImage| {
            a.pixels()
                .zip(b.pixels())
                .filter(|(a, b)| a.0.iter().zip(b.0).any(|(x, y)| x.abs_diff(y) > 2))
                .count()
        };
        assert_eq!(changed(&off, &zero), 0, "zero aperture changed the source");
        assert!(
            changed(&off, &blurred) > 100,
            "defocus had no meaningful effect"
        );
    });
}

// Universal controls must remain accepted and effective regardless of the preset.
#[test]
fn universal_style_roundtrip_and_legacy_defaults() {
    for mode in [
        "physical",
        "stylized",
        "toon",
        "clay",
        "cel",
        "ink_wash_soft_v1",
    ] {
        let resource = format!(
            "<RenderStyle id=\"u\">\n<SurfaceStyle shading=\"{mode}\" />\n<ColorStyle tint=\"#DDE5D7\" tintStrength=\"0.15\" saturation=\"0.75\" />\n<ToneStyle exposure=\"1\" contrast=\"0.95\" shadowColor=\"#293B38\" highlightColor=\"#F1EBDD\" toneStrength=\"0.65\" />\n</RenderStyle>"
        );
        let graph = parse_graph_script(&script(&resource, "renderStyle=\"u\"")).unwrap();
        let style = resolve_scene_render_style(&graph, "main").unwrap();
        assert!(style.universal.enabled);
        assert_eq!(style.universal.tint_strength, 0.15);
        assert_eq!(style.universal.tone_strength, 0.65);
        let restored = serde_json::from_str(&serde_json::to_string(&graph).unwrap()).unwrap();
        assert_eq!(
            style,
            resolve_scene_render_style(&restored, "main").unwrap()
        );
        let mut old = serde_json::to_value(&style).unwrap();
        old.as_object_mut().unwrap().remove("universal");
        let old: motionloom::render_style::ResolvedSceneRenderStyle =
            serde_json::from_value(old).unwrap();
        assert!(!old.universal.enabled);
        assert_eq!(old.universal.exposure, 1.0);
    }
}

#[test]
fn universal_style_rejects_invalid_controls() {
    for child in [
        "<ColorStyle tint=\"red\" />",
        "<ColorStyle tintStrength=\"1.1\" />",
        "<ColorStyle saturation=\"-1\" />",
        "<ColorStyle unknown=\"1\" />",
        "<ToneStyle exposure=\"NaN\" />",
        "<ToneStyle contrast=\"4\" />",
        "<ToneStyle toneStrength=\"-0.1\" />",
        "<ToneStyle highlightColor=\"#GGGGGG\" />",
        "<ColorStyle />\n<ColorStyle />",
        "<ToneStyle />\n<ToneStyle />",
        "<SurfaceStyle shading=\"ink_typo\" />",
    ] {
        let resource = format!("<RenderStyle id=\"u\">\n{child}\n</RenderStyle>");
        assert!(
            parse_graph_script(&script(&resource, "renderStyle=\"u\"")).is_err(),
            "accepted {child}"
        );
    }
}

#[test]
fn ink_template_has_strict_authoring_coverage() {
    let source = include_str!("../examples/ink_wash.motionloom");
    let graph = parse_graph_script(source).unwrap();
    assert_eq!(
        resolve_scene_render_style(&graph, "ink_study")
            .unwrap()
            .shading,
        "ink_wash_soft_v1"
    );
    let report: serde_json::Value = serde_json::from_str(
        &motionloom::api::motionloom_analyze_script_for_target_json(source, "native-webgpu"),
    )
    .unwrap();
    assert_eq!(report["summary"]["errors"], 0);
    assert_eq!(report["summary"]["unknownTags"], 0);
    assert_eq!(report["summary"]["ignoredAttributes"], 0);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires a native GPU adapter"]
fn universal_controls_affect_every_preset_on_gpu() {
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let mut physical = None;
        for mode in [
            "physical",
            "stylized",
            "toon",
            "clay",
            "cel",
            "ink_wash_soft_v1",
        ] {
            let source = |controls: &str| {
                parse_graph_script(&script(
                &format!("<RenderStyle id=\"u\">\n<SurfaceStyle shading=\"{mode}\" />\n{controls}\n</RenderStyle>"),
                "renderStyle=\"u\"")).unwrap()
            };
            let base = renderer
                .render_frame_gpu_readback(&source(""), 0)
                .await
                .unwrap();
            let neutral = renderer
                .render_frame_gpu_readback(&source("<ColorStyle />\n<ToneStyle />"), 0)
                .await
                .unwrap();
            assert!(base == neutral, "neutral controls changed {mode}");
            let red = renderer
                .render_frame_gpu_readback(
                    &source("<ColorStyle tint=\"#FF0000\" tintStrength=\"1\" />"),
                    0,
                )
                .await
                .unwrap();
            let gray = renderer
                .render_frame_gpu_readback(&source("<ColorStyle saturation=\"0\" />"), 0)
                .await
                .unwrap();
            let black = renderer
                .render_frame_gpu_readback(&source("<ToneStyle exposure=\"0\" />"), 0)
                .await
                .unwrap();
            let blue = renderer.render_frame_gpu_readback(&source("<ToneStyle shadowColor=\"#0000FF\" highlightColor=\"#0000FF\" toneStrength=\"1\" />"), 0).await.unwrap();
            let p = red.get_pixel(128, 96);
            assert!(
                p[0] > 10 && p[1] < 3 && p[2] < 3,
                "tint failed for {mode}: {p:?}"
            );
            let p = gray.get_pixel(128, 96);
            assert!(
                p[0].abs_diff(p[1]) <= 1 && p[1].abs_diff(p[2]) <= 1,
                "saturation failed for {mode}"
            );
            let p = black.get_pixel(128, 96);
            assert!(
                p[0] < 2 && p[1] < 2 && p[2] < 2,
                "exposure failed for {mode}"
            );
            let p = blue.get_pixel(128, 96);
            assert!(
                p[0] < 3 && p[1] < 3 && p[2] > 240,
                "tone map failed for {mode}"
            );
            if mode == "physical" {
                physical = Some(base.clone());
            }
            if mode == "ink_wash_soft_v1" {
                // A coloured material must not collapse into the old grey-green wash.
                let original = physical.as_ref().unwrap().get_pixel(128, 96);
                let painted = base.get_pixel(128, 96);
                let chroma = |p: &image::Rgba<u8>| {
                    *p.0[..3].iter().max().unwrap() - *p.0[..3].iter().min().unwrap()
                };
                assert!(
                    chroma(painted) as f32 >= chroma(original) as f32 * 0.65,
                    "ink must retain material colour: {original:?} -> {painted:?}"
                );
                let difference: usize = base
                    .pixels()
                    .zip(physical.as_ref().unwrap().pixels())
                    .map(|(a, b)| (0..3).map(|i| a[i].abs_diff(b[i]) as usize).sum::<usize>())
                    .sum();
                assert!(
                    difference > 100_000,
                    "ink preset must differ visibly from physical"
                );
            }
        }
    });
}

// Exercise the solid-color GLB path, not just generated primitive textures.
#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires GPU and sibling Character1 CC0 asset"]
fn character1_opaque_glb_has_visible_outline() {
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    let asset = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../../motionloom-example/assets/sample_assets/characters/character1/character1.glb",
    );
    assert!(asset.is_file());
    let source = script(
        "<RenderStyle id=\"c\">\n<SurfaceStyle shading=\"cel\" />\n<OutlineStyle width=\"3\" />\n</RenderStyle>",
        "renderStyle=\"c\"",
    ).replace("<PrimitiveAsset id=\"ball\" shape=\"sphere\" radius=\"1\" material=\"paint\" />",
        &format!("<ModelAsset id=\"ball\" src=\"{}\" />", asset.display()))
     .replace("<Model id=\"hero\" asset=\"ball\" />", "<Model id=\"hero\" asset=\"ball\" scaleMode=\"normalize_height\" scale=\"2\" />")
     .replace("target={[0,0,0]}", "target={[0,1,0]}");
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let on = renderer
            .render_frame(&parse_graph_script(&source).unwrap(), 0)
            .await
            .unwrap();
        let off_source = source.replace("width=\"3\"", "enabled=\"false\"");
        let off = renderer
            .render_frame(&parse_graph_script(&off_source).unwrap(), 0)
            .await
            .unwrap();
        let black_added = on
            .pixels()
            .zip(off.pixels())
            .filter(|(a, b)| {
                a[3] > 240
                    && a[0] < 10
                    && a[1] < 10
                    && a[2] < 10
                    && (b[3] < 10 || b[0] > 30 || b[1] > 30 || b[2] > 30)
            })
            .count();
        assert!(
            black_added > 50,
            "expected visible GLB outline; added {black_added} black pixels"
        );
    });
}

// Keep camera, lights and assets identical across visual comparisons.
fn script(resource: &str, reference: &str) -> String {
    format!(
        r##"<Graph fps="30" duration="1s" size={{[256,192]}}>
{resource}
<Assets>
<MaterialAsset id="paint" baseColor="#38ACB8" roughness="0.32" />
<PrimitiveAsset id="ball" shape="sphere" radius="1" material="paint" />
</Assets>
<Scene id="main" {reference}>
<Timeline>
<Track id="stage" space="3d">
<Sequence from="0s" duration="1s">
<CompositeGroup id="island" space="3d" depth="true" format="rgba16f">
<Camera3D id="cam" position={{[0,1,4]}} target={{[0,0,0]}} fov="45" />
<DirectionalLight direction={{[-0.4,-1,-0.4]}} intensity="3" />
<Model id="hero" asset="ball" />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="main" />
</Graph>"##
    )
}

#[test]
fn style_reference_and_legacy_defaults() {
    let plain = parse_graph_script(&script("", "")).unwrap();
    assert!(plain.render_styles.is_empty());
    let legacy = resolve_scene_render_style(&plain, "main").unwrap();
    assert_eq!(legacy.shading, "physical");
    assert_eq!(legacy.specular, 1.0);
    let styled = parse_graph_script(&script("<RenderStyle id=\"toon\">\n<SurfaceStyle shading=\"toon\" shadingSteps=\"4\" />\n</RenderStyle>", "renderStyle=\"toon\"")).unwrap();
    let r = resolve_scene_render_style(&styled, "main").unwrap();
    assert_eq!(r.shading_steps, 4);
    assert_eq!(r.shading, "toon");
    assert!(styled.raw_script.unwrap().contains("renderStyle=\"toon\""));
}

#[test]
fn cel_defaults_and_outline_validation() {
    let cel = "<RenderStyle id=\"c\">\n<SurfaceStyle shading=\"cel\" />\n</RenderStyle>";
    let graph = parse_graph_script(&script(cel, "renderStyle=\"c\"")).unwrap();
    let r = resolve_scene_render_style(&graph, "main").unwrap();
    assert_eq!(r.cel.outline_width, 1.5);
    assert_eq!(r.cel.outline_color, [0.0; 3]);
    let disabled = cel.replace(
        "</RenderStyle>",
        "<OutlineStyle enabled=\"false\" />\n</RenderStyle>",
    );
    let graph = parse_graph_script(&script(&disabled, "renderStyle=\"c\"")).unwrap();
    assert_eq!(
        resolve_scene_render_style(&graph, "main")
            .unwrap()
            .cel
            .outline_width,
        0.0
    );
    for child in [
        "<OutlineStyle width=\"-1\" />",
        "<OutlineStyle method=\"screen\" />",
        "<OutlineStyle distanceMode=\"world\" />",
        "<SurfaceStyle shadowFeather=\"0\" />",
    ] {
        let bad = format!("<RenderStyle id=\"c\">\n{child}\n</RenderStyle>");
        assert!(parse_graph_script(&script(&bad, "renderStyle=\"c\"")).is_err());
    }
}

#[test]
fn invalid_styles_are_rejected_before_render() {
    for resource in [
        "<RenderStyle id=\"t\">\n<SurfaceStyle shading=\"typo\" />\n</RenderStyle>",
        "<RenderStyle id=\"t\">\n<SurfaceStyle shadingSteps=\"0\" />\n</RenderStyle>",
        "<RenderStyle id=\"t\">\n<SurfaceStyle typo=\"1\" />\n</RenderStyle>",
        "<RenderStyle id=\"t\">\n<SurfaceStyle />\n<SurfaceStyle />\n</RenderStyle>",
        "<RenderStyle id=\"t\">\n<PostStyle exposure=\"NaN\" />\n</RenderStyle>",
        "<RenderStyle id=\"t\" unknown=\"x\">\n</RenderStyle>",
    ] {
        assert!(
            parse_graph_script(&script(resource, "renderStyle=\"t\"")).is_err(),
            "{resource}"
        );
    }
    assert!(parse_graph_script(&script("", "renderStyle=\"missing\"")).is_err());
    let missing_key = script(
        "<RenderStyle id=\"t\">\n<SurfaceStyle shading=\"toon\" />\n</RenderStyle>",
        "renderStyle=\"t\"",
    )
    .replace(
        "<Present",
        "<AnimationTarget node=\"main\" property=\"renderStyle\">\n<Key time=\"0s\" value=\"missing\" />\n</AnimationTarget>\n<Present",
    );
    assert!(parse_graph_script(&missing_key).is_err());
}

#[test]
fn removed_quality_syntax_is_rejected() {
    assert!(parse_graph_script(&script("<RenderQuality id=\"q\">\n</RenderQuality>", "")).is_err());
    assert!(parse_graph_script(&script("", "renderQuality=\"q\"")).is_err());

    let graph = parse_graph_script(&script("", "")).unwrap();
    let mut graph_json = serde_json::to_value(&graph).unwrap();
    graph_json
        .as_object_mut()
        .unwrap()
        .insert("renderQualities".into(), serde_json::json!([]));
    assert!(serde_json::from_value::<motionloom::GraphScript>(graph_json).is_err());

    let mut scene_json = serde_json::to_value(&graph).unwrap();
    scene_json["scenes"][0]
        .as_object_mut()
        .unwrap()
        .insert("renderQuality".into(), serde_json::json!("q"));
    assert!(serde_json::from_value::<motionloom::GraphScript>(scene_json).is_err());
}

#[test]
fn bloom_uses_existing_process_and_survives_json_roundtrip() {
    let g = parse_graph_script(&script(
        "<RenderStyle id=\"t\">\n<PostStyle bloomIntensity=\"0.1\" />\n</RenderStyle>",
        "renderStyle=\"t\"",
    ))
    .unwrap();
    assert_eq!(g.processes.len(), 1);
    assert_eq!(g.passes[0].effect, "glow_bloom");
    let copy: motionloom::GraphScript =
        serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
    assert_eq!(g.scenes, copy.scenes);
}

#[test]
fn style_never_rewrites_authored_materials() {
    let plain = parse_graph_script(&script("", "")).unwrap();
    let g = parse_graph_script(&script(
        "<RenderStyle id=\"t\">\n<SurfaceStyle shading=\"clay\" />\n</RenderStyle>",
        "renderStyle=\"t\"",
    ))
    .unwrap();
    assert_eq!(plain.material_assets, g.material_assets);
    assert_eq!(plain.assets, g.assets);
}

#[test]
fn explicit_nodes_are_reported() {
    let text = script("<RenderStyle id=\"t\">\n<LightingStyle preset=\"night\" />\n<PostStyle exposure=\"2\" />\n</RenderStyle>", "renderStyle=\"t\"")
        .replace("<Model id=\"hero\"", "<ColorManagement id=\"grade\" exposure=\"1.2\" />\n<Model id=\"hero\"");
    let g = parse_graph_script(&text).unwrap();
    let r = resolve_scene_render_style(&g, "main").unwrap();
    assert!(r.overrides.iter().any(|v| v.property == "lighting.preset"));
    let exposure = r
        .overrides
        .iter()
        .find(|v| v.property == "post.exposure")
        .unwrap();
    assert_eq!(exposure.final_expression, "1.2");
    assert_eq!(exposure.style_value, serde_json::json!(2.0));
}

#[test]
fn styles_are_scene_local_and_serialized_graphs_roundtrip() {
    let text = script(
        "<RenderStyle id=\"t\">\n<SurfaceStyle shading=\"toon\" />\n</RenderStyle>",
        "renderStyle=\"t\"",
    )
    .replace("<Present", "<Scene id=\"plain\">\n</Scene>\n<Present");
    let g = parse_graph_script(&text).unwrap();
    assert_eq!(
        resolve_scene_render_style(&g, "plain").unwrap().shading,
        "physical"
    );
    assert_eq!(
        resolve_scene_render_style(&g, "main").unwrap().shading,
        "toon"
    );
    let encoded = serde_json::to_value(&g).unwrap();
    let loaded: motionloom::GraphScript = serde_json::from_value(encoded).unwrap();
    assert_eq!(loaded.render_styles, g.render_styles);
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires the sibling motionloom-example checkout"]
fn existing_showcases_remain_parseable() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../motionloom-example/showcase");
    for number in 1..=79 {
        let file = root.join(format!("s-{number:06}/main.motionloom"));
        let text = std::fs::read_to_string(&file).unwrap();
        parse_graph_script(&text).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires the sibling motionloom-example checkout"]
fn showcases_80_to_88_remain_parseable() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../motionloom-example/showcase");
    for number in 80..=88 {
        let file = root.join(format!("s-{number:06}/main.motionloom"));
        let text = std::fs::read_to_string(&file).unwrap();
        parse_graph_script(&text).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires a GPU; run explicitly on native Metal/WebGPU host"]
fn rendered_modes_differ_and_preserve_output_size() {
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let mut images = Vec::new();
        for mode in ["physical", "stylized", "toon", "clay", "cel"] {
            let g = parse_graph_script(&script(
                &format!(
                    "<RenderStyle id=\"t\">\n<SurfaceStyle shading=\"{mode}\" />\n</RenderStyle>"
                ),
                "renderStyle=\"t\"",
            ))
            .unwrap();
            let image = renderer.render_frame(&g, 0).await.unwrap();
            assert_eq!(image.dimensions(), (256, 192));
            assert!(image.as_raw().iter().any(|v| *v > 0));
            images.push(image);
        }
        for pair in images.windows(2) {
            assert_ne!(pair[0].as_raw(), pair[1].as_raw());
        }
        let plain = parse_graph_script(&script("", "")).unwrap();
        assert_eq!(renderer.render_frame(&plain, 0).await.unwrap(), images[0]);
    });
}

#[test]
fn cel_material_settings_validate_and_roundtrip() {
    let source = script("", "").replace(
        "<Model id=\"hero\" asset=\"ball\" />",
        "<Model id=\"hero\" asset=\"ball\">\n<MaterialBinding material=\"*\" celRole=\"hair\" outlineWidth=\"0.5\" celShadowColor=\"#684C71\" hairHighlight=\"0.3\" />\n</Model>",
    );
    let graph = parse_graph_script(&source).unwrap();
    let json = serde_json::to_value(&graph).unwrap();
    let restored: motionloom::GraphScript = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), json);
    for (old, new) in [
        ("outlineWidth=\"0.5\"", "outlineWidth=\"-1\""),
        ("celRole=\"hair\"", "celRole=\"unknown\""),
        ("hairHighlight=\"0.3\"", "hairHighlight=\"NaN\""),
        ("#684C71", "bad-color"),
    ] {
        assert!(parse_graph_script(&source.replace(old, new)).is_err());
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires a GPU"]
fn cel_outline_material_override_and_missing_slot() {
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let source = script(
            "<RenderStyle id=\"c\">\n<SurfaceStyle shading=\"cel\" />\n<OutlineStyle width=\"3\" />\n</RenderStyle>",
            "renderStyle=\"c\"",
        );
        let normal = parse_graph_script(&source).unwrap();
        let outlined = renderer.render_frame(&normal, 0).await.unwrap();
        let disabled_source = source.replace(
            "<OutlineStyle width=\"3\" />",
            "<OutlineStyle enabled=\"false\" />",
        );
        let disabled = renderer
            .render_frame(&parse_graph_script(&disabled_source).unwrap(), 0)
            .await
            .unwrap();
        assert_ne!(outlined, disabled);
        let override_source = source.replace("<Model id=\"hero\" asset=\"ball\" />",
            "<Model id=\"hero\" asset=\"ball\">\n<MaterialBinding material=\"*\" outlineWidth=\"0\" />\n</Model>");
        let override_image = renderer
            .render_frame(&parse_graph_script(&override_source).unwrap(), 0)
            .await
            .unwrap();
        assert_eq!(override_image, disabled);
        let bad = override_source.replace("material=\"*\"", "material=\"missing_slot\"");
        let error = renderer
            .render_frame(&parse_graph_script(&bad).unwrap(), 0)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("missing_slot"), "{error}");
    });
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires a GPU"]
fn cel_control_texture_changes_outline_and_face_shadow() {
    use base64::Engine;
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    fn source(pixel: [u8; 4], role: &str) -> String {
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(2, 2, image::Rgba(pixel))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let data = base64::engine::general_purpose::STANDARD.encode(png.into_inner());
        script("<RenderStyle id=\"c\">\n<SurfaceStyle shading=\"cel\" />\n<OutlineStyle width=\"3\" />\n</RenderStyle>", "renderStyle=\"c\"")
            .replace("<Assets>", &format!("<Assets>\n<ImageAsset id=\"control\" src=\"data:image/png;base64,{data}\" colorSpace=\"linear-srgb\" />"))
            .replace("<Model id=\"hero\" asset=\"ball\" />", &format!("<Model id=\"hero\" asset=\"ball\">\n<MaterialBinding material=\"*\" celRole=\"{role}\" celControlMap=\"control\" />\n</Model>"))
    }
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let white = source([255; 4], "body");
        let black = source([0, 255, 255, 255], "body");
        let a = renderer
            .render_frame(&parse_graph_script(&white).unwrap(), 0)
            .await
            .unwrap();
        let b = renderer
            .render_frame(&parse_graph_script(&black).unwrap(), 0)
            .await
            .unwrap();
        assert_ne!(a, b);
        let off = black.replace(
            "<OutlineStyle width=\"3\" />",
            "<OutlineStyle enabled=\"false\" />",
        );
        assert_eq!(
            b,
            renderer
                .render_frame(&parse_graph_script(&off).unwrap(), 0)
                .await
                .unwrap()
        );
        let face_lit = source([0, 255, 255, 255], "face");
        let face_shadow = source([0, 0, 255, 255], "face");
        let a = renderer
            .render_frame(&parse_graph_script(&face_lit).unwrap(), 0)
            .await
            .unwrap();
        let b = renderer
            .render_frame(&parse_graph_script(&face_shadow).unwrap(), 0)
            .await
            .unwrap();
        assert!(a != b, "face SDF channels should produce different pixels");
        assert!(parse_graph_script(&face_lit.replace(" celControlMap=\"control\"", "")).is_err());
        let bad = face_lit.replace("celControlMap=\"control\"", "celControlMap=\"missing\"");
        assert!(
            renderer
                .render_frame(&parse_graph_script(&bad).unwrap(), 0)
                .await
                .unwrap_err()
                .to_string()
                .contains("missing")
        );
    });
}
