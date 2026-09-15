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

#[test]
fn anti_aliasing_methods_parse_and_resolve_with_stable_defaults() {
    for method in ["auto", "off", "fxaa", "smaa", "msaa", "taa", "ssaa"] {
        let fallback = if method == "off" {
            ""
        } else if method == "auto" {
            " fallback=\"fxaa\""
        } else {
            " fallback=\"auto\""
        };
        let source = format!(
            r#"<Graph fps={{30}} duration="1s" size={{[320,180]}}>
  <RenderStyle id="style">
    <AntiAliasingStyle method="{method}" quality="high"{fallback} sharpness="0.2" />
  </RenderStyle>
  <Scene id="aa_scene" renderStyle="style">
  </Scene>
  <Present from="aa_scene" />
</Graph>"#
        );
        let graph = crate::dsl::parse_graph_script(&source).unwrap();
        let aa = resolve_scene_render_style(&graph, "aa_scene")
            .unwrap()
            .anti_aliasing
            .unwrap();
        assert_eq!(aa.method, method);
        assert_eq!(aa.quality, "high");
        assert_eq!(aa.sharpness, 0.2);
    }
}

#[test]
fn authored_style_without_anti_aliasing_is_off_but_unstyled_scene_uses_host_policy() {
    let source = r#"<Graph fps={30} duration="1s" size={[320,180]}>
  <RenderStyle id="style">
    <SurfaceStyle shading="physical" />
  </RenderStyle>
  <Scene id="styled" renderStyle="style">
  </Scene>
  <Scene id="unstyled">
  </Scene>
  <Present from="styled" />
</Graph>"#;
    let graph = crate::dsl::parse_graph_script(source).unwrap();
    assert_eq!(
        resolve_scene_render_style(&graph, "styled")
            .unwrap()
            .anti_aliasing
            .unwrap()
            .method,
        "off"
    );
    assert!(
        resolve_scene_render_style(&graph, "unstyled")
            .unwrap()
            .anti_aliasing
            .is_none()
    );
}

#[test]
fn anti_aliasing_rejects_unknown_values_and_unsafe_ranges() {
    let source = r#"<Graph fps={30} duration="1s" size={[320,180]}>
  <RenderStyle id="style">
    <AntiAliasingStyle method="taa" quality="high" fallback="fxaa" sharpness="0.2" />
  </RenderStyle>
  <Scene id="scene" renderStyle="style">
  </Scene>
  <Present from="scene" />
</Graph>"#;
    for bad in [
        source.replace("method=\"taa\"", "method=\"txaa\""),
        source.replace("quality=\"high\"", "quality=\"extreme\""),
        source.replace("sharpness=\"0.2\"", "sharpness=\"1.2\""),
        source.replace("fallback=\"fxaa\"", "fallback=\"taa\""),
    ] {
        assert!(crate::dsl::parse_graph_script(&bad).is_err());
    }
}
