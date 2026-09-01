#![cfg(not(target_arch = "wasm32"))]

use motionloom::{SceneRenderProfile, parse_graph_script, render_scene_graph_frame};

#[test]
fn atmosphere_fog_and_camera_dof_are_optional_compatible_scene_features() {
    let graph = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[32,24]}>
  <Background color="#101820" />
  <Scene id="optics_scene">
    <Timeline>
      <Track id="world" space="3d">
        <Sequence duration="1s">
          <CompositeGroup space="3d" depth="true">
            <AtmosphereFog id="mist" mode="height" color="#BFD9C8" density="0.02" start="2" end="30" baseHeight="0.4" heightFalloff="0.2" scattering="0.1" affectSky="true" boundsMin={[-4,0,-8]} boundsMax={[4,6,-1]} edgeFeather="0.75" />
            <Camera3D id="portrait" position={[0,1,5]} target={[0,1,0]} fov="35" depthOfField="true" focusDistance="5" focalLength="50" fStop="2.8" maxBlur="8" />
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="optics_scene" />
</Graph>
"##,
    )
    .expect("parse optional atmosphere and camera optics");

    assert_eq!(graph.scenes.len(), 1);
}

#[test]
fn atmosphere_fog_rejects_unknown_modes() {
    let error = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[32,24]}>
  <Background color="#101820" />
  <Scene id="bad_fog">
    <Timeline>
      <Track id="world" space="3d">
        <Sequence duration="1s">
          <CompositeGroup space="3d">
            <AtmosphereFog mode="volumetric" />
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="bad_fog" />
</Graph>
"##,
    )
    .expect_err("unknown fog mode must fail explicitly");

    assert!(error.message.contains("linear, exp, or height"));
}

#[test]
fn atmosphere_fog_requires_complete_local_bounds() {
    let error = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[32,24]}>
  <Scene id="bad_local_fog">
    <Timeline>
      <Track>
        <Sequence duration="1s">
          <CompositeGroup space="3d">
            <AtmosphereFog density="0.02" boundsMin={[-4,0,-8]} />
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="bad_local_fog" />
</Graph>
"##,
    )
    .expect_err("local fog requires both bounds");

    assert!(
        error.message.contains("boundsMin and boundsMax"),
        "unexpected parse error: {}",
        error.message
    );
}

#[test]
fn public_scene_render_api_draws_cpu_frame() {
    let graph = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[32,24]}>
  <Background color="#000000" />

  <Scene id="api_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Rect x="4" y="6" width="10" height="8" color="#ff0000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="api_scene" />
</Graph>
"##,
    )
    .expect("parse scene graph");

    let frame = pollster::block_on(render_scene_graph_frame(&graph, 0, SceneRenderProfile::Cpu))
        .expect("render frame");
    assert_eq!(frame.width(), 32);
    assert_eq!(frame.height(), 24);

    let red = frame.get_pixel(8, 10);
    assert!(red[0] > 200 && red[1] < 40 && red[2] < 40, "got {red:?}");
}

#[test]
fn limb_envelope_builds_an_exact_layer_bone_mesh() {
    let graph = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[96,64]}>
  <Background color="#101820" />
  <Scene id="limb_envelope_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Rect x="8" y="22" width="76" height="20" color="#F3C9A8" />
            <PuppetWarp id="arm_rig" target="@layer" solver="bones"
              preserveOutside="true" jointSoftness="8">
              <LimbEnvelope id="arm_area"
                d="M 12 24 L 80 24 L 84 32 L 80 40 L 12 40 Z"
                alphaClip="true" handFrom="wrist" />
              <PuppetPin id="shoulder" role="anchor"
                x="16" y="32" fixed="true" />
              <PuppetPin id="elbow" role="joint" x="48" y="32" />
              <PuppetPin id="wrist" role="control"
                x="76" y="32" targetX="64" targetY="12" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="limb_envelope_scene" />
</Graph>
"##,
    )
    .expect("parse LimbEnvelope scene");

    let frame = pollster::block_on(render_scene_graph_frame(
        &graph,
        15,
        SceneRenderProfile::Cpu,
    ))
    .expect("render LimbEnvelope frame");
    assert_eq!(frame.width(), 96);
    assert_eq!(frame.height(), 64);
    // preserveOutside keeps source artwork outside the exact envelope intact.
    let preserved = frame.get_pixel(9, 32);
    assert!(
        preserved[0] > 200 && preserved[1] > 150,
        "outside artwork was not preserved: {preserved:?}"
    );
    // The moved wrist produces skin-colored pixels above its bind pose.
    let moved = frame.get_pixel(64, 13);
    assert!(
        moved[0] > 150 && moved[1] > 100,
        "envelope-controlled wrist did not move: {moved:?}"
    );
}

#[test]
fn limb_regions_bind_anchor_joint_and_control_areas() {
    let graph = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[96,64]}>
  <Background color="#101820" />
  <Scene id="limb_regions_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Rect x="8" y="22" width="76" height="20" color="#F3C9A8" />
            <PuppetWarp id="arm_rig" target="@layer" solver="bones"
              preserveOutside="true" jointSoftness="8">
              <LimbRegion id="upper_area" role="anchor"
                d="M 12 24 L 40 24 L 40 40 L 12 40 Z" />
              <LimbRegion id="elbow_area" role="joint"
                d="M 36 24 L 56 24 L 56 40 L 36 40 Z" />
              <LimbRegion id="lower_area" role="control"
                d="M 52 24 L 84 24 L 84 40 L 52 40 Z" />
              <PuppetPin id="shoulder" role="anchor"
                x="16" y="32" fixed="true" />
              <PuppetPin id="elbow" role="joint" x="48" y="32" />
              <PuppetPin id="wrist" role="control"
                x="76" y="32" targetX="64" targetY="12" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="limb_regions_scene" />
</Graph>
"##,
    )
    .expect("parse LimbRegion scene");

    let frame = pollster::block_on(render_scene_graph_frame(
        &graph,
        15,
        SceneRenderProfile::Cpu,
    ))
    .expect("render LimbRegion frame");
    assert_eq!(frame.width(), 96);
    assert_eq!(frame.height(), 64);
    let preserved = frame.get_pixel(9, 32);
    assert!(
        preserved[0] > 200 && preserved[1] > 150,
        "outside artwork was not preserved: {preserved:?}"
    );
    let moved = (58..71).any(|x| {
        (6..21).any(|y| {
            let pixel = frame.get_pixel(x, y);
            pixel[0] > 150 && pixel[1] > 100
        })
    });
    assert!(moved, "control region did not follow the wrist: {moved:?}");
}

#[test]
fn incomplete_limb_regions_keep_the_bind_pose() {
    let graph = parse_graph_script(
        r##"
<Graph fps={30} duration="1s" size={[96,64]}>
  <Background color="#101820" />
  <Scene id="partial_limb_regions_scene">
    <Timeline>
      <Track id="main" z="0">
        <Sequence duration="1s">
          <Layer>
            <Rect x="8" y="22" width="76" height="20" color="#F3C9A8" />
            <PuppetWarp id="arm_rig" target="@layer" solver="bones"
              preserveOutside="true" jointSoftness="8">
              <LimbRegion id="hand_only" role="control"
                d="M 62 24 L 84 24 L 84 40 L 62 40 Z" />
              <PuppetPin id="shoulder" role="anchor"
                x="16" y="32" fixed="true" />
              <PuppetPin id="elbow" role="joint" x="48" y="32" />
              <PuppetPin id="wrist" role="control"
                x="76" y="32" targetX="64" targetY="12" />
            </PuppetWarp>
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="partial_limb_regions_scene" />
</Graph>
"##,
    )
    .expect("parse partial LimbRegion scene");

    let frame = pollster::block_on(render_scene_graph_frame(
        &graph,
        15,
        SceneRenderProfile::Cpu,
    ))
    .expect("render partial LimbRegion frame");
    let original_hand = frame.get_pixel(76, 32);
    assert!(
        original_hand[0] > 150 && original_hand[1] > 100,
        "partial regions detached the original hand: {original_hand:?}"
    );
    let premature_move = frame.get_pixel(64, 13);
    assert!(
        premature_move[0] < 100 && premature_move[1] < 100,
        "partial regions activated before all roles existed: {premature_move:?}"
    );
}
