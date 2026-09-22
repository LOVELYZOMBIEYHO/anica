// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/collect.rs

use crate::{
    dsl::GraphScript,
    scene::model::{GroupNode, SceneNode, SceneRootNode},
};

use super::{
    AlphaMode, ColorStage, CompositionLayer, CompositionSource, RenderDomain,
    SceneCompositionError, SceneCompositionPlan, SourceColorSpace,
};

/// Compile structural layer ordering after animation evaluation has selected a frame.
pub fn build_scene_composition_plan(
    graph: &GraphScript,
    scene_id: &str,
) -> Result<SceneCompositionPlan, SceneCompositionError> {
    let scene = graph
        .scenes
        .iter()
        .find(|scene| scene.id == scene_id)
        .ok_or_else(|| SceneCompositionError::MissingScene(scene_id.into()))?;
    let logical = scene.size.unwrap_or(graph.size);
    let output = graph.render_size.unwrap_or(graph.size);
    let mut plan = SceneCompositionPlan::new(
        [logical.0.max(1), logical.1.max(1)],
        [output.0.max(1), output.1.max(1)],
    );
    collect_scene(scene, &mut plan);
    plan.effects.extend(
        scene
            .effects
            .iter()
            .chain(&scene.post_effects)
            .map(|effect| effect.process.clone()),
    );
    plan.sort_layers();
    plan.derive_required_aovs();
    plan.validate()?;
    Ok(plan)
}

fn collect_scene(scene: &SceneRootNode, plan: &mut SceneCompositionPlan) {
    for node in &scene.children {
        collect_node(node, 0, "scene", plan);
    }
}

fn collect_node(
    node: &SceneNode,
    inherited_order: i32,
    inherited_id: &str,
    plan: &mut SceneCompositionPlan,
) {
    match node {
        SceneNode::Timeline(value) => {
            let id = value.id.as_deref().unwrap_or(inherited_id);
            for child in &value.children {
                collect_node(child, inherited_order, id, plan);
            }
        }
        SceneNode::Track(value) => {
            let order = value.composite_order.unwrap_or(value.z);
            let id = value.id.as_deref().unwrap_or(inherited_id);
            let before = plan.layers.len();
            for child in &value.children {
                collect_node(child, order, id, plan);
            }
            // A screen track without an explicit 3D group is one retained 2D run.
            if plan.layers.len() == before && value.space != "3d" {
                plan.layers.push(screen_layer(
                    id,
                    order,
                    &value
                        .effects
                        .iter()
                        .map(|e| e.process.clone())
                        .collect::<Vec<_>>(),
                ));
            }
        }
        SceneNode::Sequence(value) => {
            let id = value.id.as_deref().unwrap_or(inherited_id);
            for child in &value.children {
                collect_node(child, inherited_order, id, plan);
            }
        }
        SceneNode::Chain(value) => {
            let id = value.id.as_deref().unwrap_or(inherited_id);
            for child in &value.children {
                collect_node(child, inherited_order, id, plan);
            }
        }
        SceneNode::Group(value) => collect_group(value, inherited_order, inherited_id, plan),
        SceneNode::Layer(value) => {
            let id = value.id.as_deref().unwrap_or(inherited_id);
            plan.layers.push(screen_layer(
                id,
                inherited_order,
                &value
                    .process_effects
                    .iter()
                    .map(|e| e.process.clone())
                    .collect::<Vec<_>>(),
            ));
        }
        SceneNode::Defs(_) => {}
        _ => {}
    }
}

fn collect_group(
    group: &GroupNode,
    inherited_order: i32,
    inherited_id: &str,
    plan: &mut SceneCompositionPlan,
) {
    if let Some(composite) = &group.composite {
        if composite.space == "3d" {
            let order = composite.composite_order.unwrap_or(inherited_order);
            plan.layers.push(CompositionLayer {
                id: group.id.clone().unwrap_or_else(|| inherited_id.into()),
                order,
                domain: RenderDomain::ThreeD,
                source: CompositionSource::ThreeDIsland,
                transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                opacity: 1.0,
                blend: "normal".into(),
                color_stage: ColorStage::SceneLinearPreDisplay,
                source_color_space: SourceColorSpace::Linear,
                source_alpha: AlphaMode::Premultiplied,
                effects: group
                    .process_effects
                    .iter()
                    .map(|effect| effect.process.clone())
                    .collect(),
                required_aovs: super::RequiredAovs {
                    coverage: true,
                    depth: true,
                    motion: true,
                    normal: true,
                    albedo: true,
                },
            });
            return;
        }
    }
    let id = group.id.as_deref().unwrap_or(inherited_id);
    for child in &group.children {
        collect_node(child, inherited_order, id, plan);
    }
}

fn screen_layer(id: &str, order: i32, effects: &[String]) -> CompositionLayer {
    CompositionLayer {
        id: id.into(),
        order,
        domain: RenderDomain::Screen,
        source: CompositionSource::TwoDRun,
        transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        opacity: 1.0,
        blend: "normal".into(),
        color_stage: ColorStage::DisplayLinearPostTransform,
        source_color_space: SourceColorSpace::Srgb,
        source_alpha: AlphaMode::Straight,
        effects: effects.to_vec(),
        required_aovs: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s74_shape_preserves_world_weather_and_title_order() {
        let graph = crate::parse_graph_script(
            r##"
<Graph fps={30} duration="1s" size={[1920,1080]}>
  <Scene id="s74">
    <Timeline>
      <Track id="world" space="3d" compositeOrder="30">
        <Sequence duration="1s">
          <CompositeGroup id="metro" space="3d">
            <Camera3D position={[0,1,5]} target={[0,1,0]} />
          </CompositeGroup>
        </Sequence>
      </Track>
      <Track id="weather" space="screen" compositeOrder="75">
        <Sequence duration="1s">
          <Layer id="rain">
            <Rect x="0" y="0" width="1920" height="1080" color="#ffffff10" />
          </Layer>
        </Sequence>
      </Track>
      <Track id="titles" space="screen" compositeOrder="90">
        <Sequence duration="1s">
          <Layer id="title">
            <Text x="10" y="20" value="TITLE" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="s74" />
</Graph>
"##,
        )
        .unwrap();
        let plan = build_scene_composition_plan(&graph, "s74").unwrap();
        assert_eq!(
            plan.layers
                .iter()
                .map(|layer| (layer.id.as_str(), layer.order, layer.domain))
                .collect::<Vec<_>>(),
            [
                ("metro", 30, RenderDomain::ThreeD),
                ("rain", 75, RenderDomain::Screen),
                ("title", 90, RenderDomain::Screen),
            ]
        );
        assert!(plan.required_aovs.coverage);
        assert!(plan.required_aovs.depth);
        assert!(plan.required_aovs.motion);
    }
}
