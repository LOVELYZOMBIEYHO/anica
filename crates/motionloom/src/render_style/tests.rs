// =========================================
// =========================================
// crates/motionloom/src/render_style/tests.rs

//! Module-boundary regression tests for render-style parsing and resolution.

use super::*;

#[test]
fn default_cel_style_is_stable_across_module_boundary() {
    assert_eq!(ResolvedCelStyle::default().shadow_threshold, 0.5);
    assert_eq!(ResolvedCelStyle::default().outline_width, 0.0);
}

#[test]
fn cinematic_dof_is_opt_in_and_strict() {
    let source = r#"<Graph fps={30} duration="1s" size={[320,180]}>
  <RenderStyle id="cinema">
    <DepthOfFieldStyle preset="cinematic_bokeh_v1" quality="high" />
  </RenderStyle>
  <Scene id="styled_scene" renderStyle="cinema">
  </Scene>
  <Present from="styled_scene" />
</Graph>"#;
    let graph = crate::dsl::parse_graph_script(source).unwrap();
    let style = resolve_scene_render_style(&graph, "styled_scene").unwrap();
    assert_eq!(
        style.depth_of_field.unwrap().quality.as_deref(),
        Some("high")
    );
    for bad in [
        source.replace("cinematic_bokeh_v1", "unknown"),
        source.replace("high", "ultra"),
    ] {
        assert!(crate::dsl::parse_graph_script(&bad).is_err());
    }
    let legacy = source.replace(
        "    <DepthOfFieldStyle preset=\"cinematic_bokeh_v1\" quality=\"high\" />",
        "",
    );
    let graph = crate::dsl::parse_graph_script(&legacy).unwrap();
    assert!(
        resolve_scene_render_style(&graph, "styled_scene")
            .unwrap()
            .depth_of_field
            .is_none()
    );
}
