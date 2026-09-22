// =========================================
// =========================================
// crates/motionloom/tests/immediate_materials.rs

#![cfg(not(target_arch = "wasm32"))]

use base64::Engine;
use motionloom::{
    ImmediatePreviewProfile, ImmediatePreviewSettings, SceneRenderProfile, SceneRenderer,
    parse_graph_script,
};

// An inline texture keeps the regression independent of downloads and asset roots.
fn black_ao() -> String {
    let image = image::RgbaImage::from_pixel(8, 8, image::Rgba([0, 0, 0, 255]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())
    )
}

fn fixture(ambient: f32, strength: f32, emission: f32, optics: &str) -> String {
    // Disable environment specular so ambient=0 isolates authored direct light.
    let ao = black_ao();
    format!(
        r##"<Graph fps="24" duration="1s" size={{[128,128]}}>
<RenderStyle id="physical">
<SurfaceStyle shading="physical" specular="0" />
<LightingStyle ambientIntensity="{ambient}" />
</RenderStyle>
<Assets>
<ImageAsset id="ao" src="{ao}" colorSpace="linear-srgb" />
<MaterialAsset id="paint" baseColor="#B08050" metallic="0" roughness="0.5" occlusionTexture="ao" occlusionStrength="{strength}" emissive="#FFFFFF" emissiveStrength="{emission}" />
<PrimitiveAsset id="ball" shape="sphere" radius="0.7" material="paint" />
</Assets>
<Scene id="main" renderStyle="physical">
<Timeline>
<Track space="3d">
<Sequence duration="1s">
<CompositeGroup space="3d" depth="true" format="rgba16f">
<Camera3D position={{[0,0,4]}} target={{[0,0,0]}} {optics} />
<DirectionalLight direction={{[-0.4,-1,-0.4]}} intensity="2" castShadow="false" />
<Model asset="ball" />
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
#[ignore = "requires a real native GPU"]
fn material_ao_only_attenuates_indirect_light() {
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let mut render = async |ambient, strength| {
            renderer
                .render_frame_gpu_readback(
                    &parse_graph_script(&fixture(ambient, strength, 0.0, "")).unwrap(),
                    0,
                )
                .await
                .unwrap()
        };
        let direct = render(0.0, 0.0).await;
        let direct_ao = render(0.0, 1.0).await;
        assert!(direct == direct_ao, "AO must not stain direct-lit albedo");
        let indirect = render(1.0, 0.0).await;
        let indirect_ao = render(1.0, 1.0).await;
        let energy = |image: &image::RgbaImage| {
            image
                .pixels()
                .map(|p| u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]))
                .sum::<u64>()
        };
        assert!(energy(&indirect) > energy(&indirect_ao) + 1000);
        assert!(
            direct == indirect_ao,
            "black AO removes the indirect contribution only"
        );
    });
}

#[test]
#[ignore = "requires a real native GPU"]
fn hdr_optics_preserve_highlight_energy_and_frame_size() {
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let optics = "depthOfField=\"true\" focusDistance=\"1\" focalLength=\"150\" fStop=\"0.7\" maxBlur=\"12\"";
        let moderate = renderer
            .render_frame_gpu_readback(
                &parse_graph_script(&fixture(0.0, 0.0, 1.0, optics)).unwrap(),
                0,
            )
            .await
            .unwrap();
        let bright = renderer
            .render_frame_gpu_readback(
                &parse_graph_script(&fixture(0.0, 0.0, 16.0, optics)).unwrap(),
                0,
            )
            .await
            .unwrap();
        assert_eq!(bright.dimensions(), (128, 128));
        let brighter = bright
            .pixels()
            .zip(moderate.pixels())
            .filter(|(a, b)| a[0] > b[0].saturating_add(8))
            .count();
        assert!(
            brighter > 100,
            "HDR highlights must survive optics: {brighter}"
        );
        assert!(
            bright.pixels().any(|p| p[3] > 0 && p[3] < 255),
            "blur must preserve partial coverage"
        );
    });
}

#[test]
#[ignore = "requires a real native GPU"]
fn saturated_highlights_keep_energy_when_blurred_over_dark_surfaces() {
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let optics = "depthOfField=\"true\" focusDistance=\"1\" focalLength=\"150\" fStop=\"0.7\" maxBlur=\"12\"";
        // Both source intensities saturate the display curve; only HDR blur
        // can retain their energy difference against an opaque black backdrop.
        let source = |emission| {
            fixture(0.0, 0.0, emission, optics)
            .replace("</Assets>", "<PrimitiveAsset id=\"backdrop\" shape=\"box\" size={[10,10,0.1]} color=\"#000000\" />\n</Assets>")
            .replace("<Model asset=\"ball\" />", "<Model asset=\"backdrop\" position={[0,0,-1]} />\n<Model asset=\"ball\" />")
        };
        let low = renderer
            .render_frame_gpu_readback(&parse_graph_script(&source(4.0)).unwrap(), 0)
            .await
            .unwrap();
        let high = renderer
            .render_frame_gpu_readback(&parse_graph_script(&source(64.0)).unwrap(), 0)
            .await
            .unwrap();
        let brighter = high
            .pixels()
            .zip(low.pixels())
            .filter(|(a, b)| a[0] > b[0].saturating_add(12))
            .count();
        assert!(
            brighter > 30,
            "tone mapping must follow HDR blur: {brighter}"
        );
    });
}

#[test]
#[ignore = "requires a real native GPU"]
fn cinematic_screen_space_lighting_executes_across_history() {
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        renderer.set_immediate_preview_settings(ImmediatePreviewSettings {
            profile: ImmediatePreviewProfile::Cinematic,
            dynamic_resolution: false,
            ..Default::default()
        });
        let graph = parse_graph_script(&fixture(0.7, 0.4, 0.0, "")).unwrap();
        let first = renderer.render_frame_gpu_readback(&graph, 0).await.unwrap();
        let second = renderer.render_frame_gpu_readback(&graph, 1).await.unwrap();
        assert_eq!(first.dimensions(), (128, 128));
        assert_eq!(second.dimensions(), (128, 128));
        assert!(second.pixels().any(|pixel| pixel[3] > 0));
    });
}
