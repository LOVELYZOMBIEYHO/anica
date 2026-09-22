// =========================================
// =========================================
// crates/motionloom/tests/volumetrics.rs

#![cfg(not(target_arch = "wasm32"))]

use motionloom::{SceneRenderProfile, SceneRenderer, parse_graph_script};

fn source(with_volume: bool) -> String {
    let fog = if with_volume {
        r##"<AtmosphereFog density="0.035" anisotropy="0.6" scatteringColor={[0.02,0.07,0.12]}
            boundsMin={[-4,-4,-8]} boundsMax={[4,4,3]}>
          <VolumetricScattering lightRef="sun" shaftStrength="1.2" maxDistance="12" shadowed="true" />
          <WaterCaustics intensity="0.2" scale="0.4" speed="0.2"
              attenuation="0.3" color="#BFE9FF" volumeTerm="true" surfaceTerm="true" />
        </AtmosphereFog>"##
    } else {
        r##"<AtmosphereFog density="0.035" scatteringColor="#16384A" />"##
    };
    format!(
        r##"<Graph fps="24" duration="1s" size={{[96,64]}}>
<Assets>
  <MaterialAsset id="paint" baseColor="#8095A0" roughness="0.45" />
  <PrimitiveAsset id="ball" shape="sphere" radius="0.8" material="paint" />
</Assets>
<Scene id="main">
<Timeline>
<Track>
<Sequence duration="1s">
<CompositeGroup space="3d" depth="true">
<Camera3D position={{[0,0,4]}} target={{[0,0,0]}} />
<DirectionalLight id="sun" direction={{[-0.4,-1,-0.3]}} intensity="3" castShadow="true" />
{fog}
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
fn volumetric_scene_parses_without_gpu_side_effects() {
    parse_graph_script(&source(true)).expect("volumetric graph should parse on CPU hosts");
    parse_graph_script(&source(false)).expect("simple atmosphere graph should remain valid");
}

#[test]
fn shadowed_volume_rejects_a_non_shadow_casting_light() {
    let invalid = source(true).replace("castShadow=\"true\"", "castShadow=\"false\"");
    let error = parse_graph_script(&invalid)
        .expect_err("common DSL validation must enforce volumetric shadow ownership");
    assert!(
        error
            .to_string()
            .contains("requires castShadow=true on its light")
    );
}

#[test]
#[ignore = "requires a real native GPU"]
fn froxel_volume_changes_the_rendered_radiance() {
    pollster::block_on(async {
        let mut renderer = SceneRenderer::new(SceneRenderProfile::Gpu).await.unwrap();
        let simple = renderer
            .render_frame_gpu_readback(&parse_graph_script(&source(false)).unwrap(), 0)
            .await
            .unwrap();
        let volume = renderer
            .render_frame_gpu_readback(&parse_graph_script(&source(true)).unwrap(), 0)
            .await
            .unwrap();
        assert_eq!(simple.dimensions(), volume.dimensions());
        assert_ne!(simple.as_raw(), volume.as_raw());
    });
}
