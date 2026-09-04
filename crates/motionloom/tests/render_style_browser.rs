// =========================================
// =========================================
// crates/motionloom/tests/render_style_browser.rs

// A real WebGPU browser must validate the shared style shader and optics pass.
#![cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn style_modes_render_on_browser_webgpu() {
    use motionloom::wasm_api::WasmSceneRenderer;
    let mut outputs = Vec::new();
    for shading in ["physical", "stylized", "toon", "clay"] {
        let script = format!(
            r##"<Graph fps="30" duration="1s" size={{[96,64]}}>
<RenderStyle id="s">
<SurfaceStyle shading="{shading}" />
</RenderStyle>
<RenderQuality id="q" preset="web_high">
<Resolution scale="0.75" />
</RenderQuality>
<Assets>
<MaterialAsset id="paint" baseColor="#38ACB8" roughness="0.32" />
<PrimitiveAsset id="ball" shape="sphere" radius="1" material="paint" />
</Assets>
<Scene id="main" renderStyle="s" renderQuality="q">
<Timeline>
<Track space="3d">
<Sequence duration="1s">
<CompositeGroup space="3d" depth="true" format="rgba16f">
<Camera3D position={{[0,1,4]}} target={{[0,0,0]}} />
<DirectionalLight direction={{[-0.4,-1,-0.4]}} intensity="3" />
<Model asset="ball" />
</CompositeGroup>
</Sequence>
</Track>
</Timeline>
</Scene>
<Present from="main" />
</Graph>"##
        );
        let mut renderer = WasmSceneRenderer::new(&script, "gpu").unwrap();
        let image = renderer.render_frame(0).await.unwrap();
        assert_eq!(image.len(), 96 * 64 * 4);
        assert!(image.chunks_exact(4).any(|p| p[0] > 20 || p[1] > 20));
        outputs.push(image);
    }
    for pair in outputs.windows(2) {
        assert_ne!(pair[0], pair[1]);
    }
}
