// =========================================
// =========================================
// crates/motionloom/tests/render_style.rs

use motionloom::api::{parse_graph_script, resolve_scene_render_style};

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
fn quality_is_separate_and_reports_real_fallbacks() {
    let g = parse_graph_script(&script("<RenderQuality id=\"q\">\n<Resolution scale=\"0.5\" />\n<Shadows resolution=\"1024\" filtering=\"pcf\" />\n<AntiAliasing mode=\"fxaa\" />\n<AmbientOcclusion quality=\"high\" />\n</RenderQuality>","renderQuality=\"q\"")).unwrap();
    let r = resolve_scene_render_style(&g, "main").unwrap();
    assert_eq!(r.render_scale, 0.5);
    assert_eq!(r.shadow_resolution, 1024);
    assert_eq!(r.anti_aliasing, "fxaa");
    assert_eq!(r.fallbacks.len(), 1);
    assert!(r.fallbacks[0].contains("not SSAO"));
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
fn explicit_nodes_are_reported_and_quality_children_override_presets() {
    let text = script("<RenderStyle id=\"t\">\n<LightingStyle preset=\"night\" />\n<PostStyle exposure=\"2\" />\n</RenderStyle>\n<RenderQuality id=\"q\" preset=\"cinematic\">\n<Resolution scale=\"0.75\" />\n</RenderQuality>", "renderStyle=\"t\" renderQuality=\"q\"")
        .replace("<Model id=\"hero\"", "<ColorManagement id=\"grade\" exposure=\"1.2\" />\n<Model id=\"hero\"");
    let g = parse_graph_script(&text).unwrap();
    let r = resolve_scene_render_style(&g, "main").unwrap();
    assert_eq!(r.render_scale, 0.75);
    assert_eq!(r.shadow_resolution, 4096);
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
fn styles_are_scene_local_and_old_serialized_graphs_load() {
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
    let plain = parse_graph_script(&script("", "")).unwrap();
    let mut old = serde_json::to_value(&plain).unwrap();
    old.as_object_mut().unwrap().remove("renderStyles");
    old.as_object_mut().unwrap().remove("renderQualities");
    for s in old["scenes"].as_array_mut().unwrap() {
        s.as_object_mut().unwrap().remove("renderStyle");
        s.as_object_mut().unwrap().remove("renderQuality");
    }
    let loaded: motionloom::GraphScript = serde_json::from_value(old).unwrap();
    assert!(loaded.render_styles.is_empty());
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
#[ignore = "requires a GPU; run explicitly on native Metal/WebGPU host"]
fn rendered_modes_differ_and_quality_preserves_output_size() {
    use motionloom::api::{SceneRenderProfile, SceneRenderer};
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let mut images = Vec::new();
        for mode in ["physical", "stylized", "toon", "clay"] {
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
        let g = parse_graph_script(&script("<RenderQuality id=\"q\">\n<Resolution scale=\"0.5\" />\n<AntiAliasing mode=\"fxaa\" />\n<Shadows resolution=\"512\" />\n</RenderQuality>","renderQuality=\"q\"")).unwrap();
        assert_eq!(
            renderer.render_frame(&g, 0).await.unwrap().dimensions(),
            (256, 192)
        );
    });
}
