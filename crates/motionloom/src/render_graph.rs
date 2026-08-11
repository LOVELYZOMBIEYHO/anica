use crate::dsl::{GraphAssetKind, GraphScript};
use crate::error::GraphParseError;
use crate::scene::model::{
    GroupNode, Scene3DNode, SceneEffectRef, SceneLayerNode, SceneNode, SceneRootNode,
    SceneTrackNode,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderPassSpace {
    Screen,
    World,
    ThreeD,
    Process,
}

impl RenderPassSpace {
    fn parse(value: &str) -> Self {
        match value {
            "screen" => Self::Screen,
            "3d" => Self::ThreeD,
            _ => Self::World,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderEffectScope {
    Scene,
    ScenePost,
    Track,
    CompositeGroup,
    Group,
    Layer,
    Process,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum RenderPassDagKind {
    Scene,
    Track,
    CompositeGroup,
    Group,
    Layer,
    ThreeDIsland,
    Effect {
        process: String,
        scope: RenderEffectScope,
    },
    ProcessPass {
        effect: String,
    },
    Present,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPassDagNode {
    pub id: String,
    pub kind: RenderPassDagKind,
    pub space: RenderPassSpace,
    pub composite_order: i32,
    pub format: String,
    #[serde(default)]
    pub inputs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPassDagEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPassDag {
    pub nodes: Vec<RenderPassDagNode>,
    pub edges: Vec<RenderPassDagEdge>,
    pub output: String,
}

struct DagBuilder<'a> {
    graph: &'a GraphScript,
    nodes: Vec<RenderPassDagNode>,
    ids: HashSet<String>,
    scene_ids: HashSet<String>,
    assets: HashMap<String, GraphAssetKind>,
}

impl<'a> DagBuilder<'a> {
    fn push(&mut self, node: RenderPassDagNode) -> Result<(), GraphParseError> {
        if !self.ids.insert(node.id.clone()) {
            return Err(dag_error(format!(
                "Render Pass DAG contains duplicate node id '{}'.",
                node.id
            )));
        }
        self.nodes.push(node);
        Ok(())
    }

    fn effect_chain(
        &mut self,
        prefix: &str,
        input: String,
        effects: &[SceneEffectRef],
        scope: RenderEffectScope,
        space: RenderPassSpace,
        order: i32,
        format: &str,
    ) -> Result<String, GraphParseError> {
        let mut previous = input;
        for (index, effect) in effects.iter().enumerate() {
            if !self
                .graph
                .processes
                .iter()
                .any(|process| process.id == effect.process)
            {
                return Err(dag_error(format!(
                    "Scene <Effect process=\"{}\"> references an unknown <Process id>.",
                    effect.process
                )));
            }
            let mut param_names = HashSet::new();
            for param in &effect.params {
                if !param_names.insert(param.name.as_str()) {
                    return Err(dag_error(format!(
                        "Scene Effect '{}' contains duplicate Param name '{}'.",
                        effect.process, param.name
                    )));
                }
            }
            let id = effect
                .id
                .clone()
                .map(|id| format!("{prefix}:effect:{id}"))
                .unwrap_or_else(|| format!("{prefix}:effect:{index}:{}", effect.process));
            self.push(RenderPassDagNode {
                id: id.clone(),
                kind: RenderPassDagKind::Effect {
                    process: effect.process.clone(),
                    scope,
                },
                space,
                composite_order: order,
                format: format.to_string(),
                inputs: vec![previous],
            })?;
            previous = id;
        }
        Ok(previous)
    }

    fn build_scene(&mut self, scene: &SceneRootNode) -> Result<String, GraphParseError> {
        let scene_prefix = format!("scene:{}", scene.id);
        let mut track_outputs = Vec::new();
        let mut track_index = 0usize;
        for node in &scene.children {
            match node {
                SceneNode::Timeline(timeline) => {
                    for child in &timeline.children {
                        if let SceneNode::Track(track) = child {
                            track_outputs.push(self.build_track(
                                &scene_prefix,
                                track,
                                track_index,
                            )?);
                            track_index += 1;
                        }
                    }
                }
                SceneNode::Track(track) => {
                    track_outputs.push(self.build_track(&scene_prefix, track, track_index)?);
                    track_index += 1;
                }
                _ => {}
            }
        }
        track_outputs.sort_by_key(|item| item.0);
        let scene_content_id = format!("{scene_prefix}:content");
        self.push(RenderPassDagNode {
            id: scene_content_id.clone(),
            kind: RenderPassDagKind::Scene,
            space: RenderPassSpace::Screen,
            composite_order: 0,
            format: "rgba8unorm".to_string(),
            inputs: track_outputs.into_iter().map(|(_, id)| id).collect(),
        })?;
        let effected = self.effect_chain(
            &scene_prefix,
            scene_content_id,
            &scene.effects,
            RenderEffectScope::Scene,
            RenderPassSpace::Screen,
            0,
            "rgba8unorm",
        )?;
        let posted = self.effect_chain(
            &format!("{scene_prefix}:post"),
            effected,
            &scene.post_effects,
            RenderEffectScope::ScenePost,
            RenderPassSpace::Screen,
            i32::MAX - 1,
            "rgba8unorm",
        )?;
        let scene_output_id = scene_prefix;
        self.push(RenderPassDagNode {
            id: scene_output_id.clone(),
            kind: RenderPassDagKind::Scene,
            space: RenderPassSpace::Screen,
            composite_order: i32::MAX,
            format: "rgba8unorm".to_string(),
            inputs: vec![posted],
        })?;
        Ok(scene_output_id)
    }

    fn build_track(
        &mut self,
        scene_prefix: &str,
        track: &SceneTrackNode,
        index: usize,
    ) -> Result<(i32, String), GraphParseError> {
        let order = track.composite_order.unwrap_or(track.z);
        let id = format!(
            "{scene_prefix}:track:{}",
            track.id.clone().unwrap_or_else(|| format!("track_{index}"))
        );
        let mut inputs = Vec::new();
        self.collect_scene_node_outputs(&id, &track.children, &mut inputs, order)?;
        self.push(RenderPassDagNode {
            id: id.clone(),
            kind: RenderPassDagKind::Track,
            space: RenderPassSpace::parse(&track.space),
            composite_order: order,
            format: "rgba8unorm".to_string(),
            inputs,
        })?;
        let output = self.effect_chain(
            &id,
            id.clone(),
            &track.effects,
            RenderEffectScope::Track,
            RenderPassSpace::parse(&track.space),
            order,
            "rgba8unorm",
        )?;
        Ok((order, output))
    }

    fn collect_scene_node_outputs(
        &mut self,
        prefix: &str,
        nodes: &[SceneNode],
        outputs: &mut Vec<String>,
        order: i32,
    ) -> Result<(), GraphParseError> {
        for (index, node) in nodes.iter().enumerate() {
            match node {
                SceneNode::Sequence(sequence) => {
                    // A Track may contain many temporal Sequences and each
                    // Sequence commonly starts with an anonymous <Layer>.
                    // Include the structural path in the DAG id so those
                    // layers do not all collapse to `layer_0`. This affects
                    // compiler bookkeeping only; Scene timing and rendering
                    // order continue to come from the original AST.
                    let sequence_prefix = format!("{prefix}:sequence:{index}");
                    self.collect_scene_node_outputs(
                        &sequence_prefix,
                        &sequence.children,
                        outputs,
                        order,
                    )?;
                }
                SceneNode::Group(group) => {
                    outputs.push(self.build_group(prefix, group, index, order)?);
                }
                SceneNode::Layer(layer) => {
                    outputs.push(self.build_layer(prefix, layer, index, order)?);
                }
                SceneNode::Precompose(node) => {
                    let precompose_prefix = format!("{prefix}:precompose:{}", node.id);
                    self.collect_scene_node_outputs(
                        &precompose_prefix,
                        &node.children,
                        outputs,
                        order,
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn build_group(
        &mut self,
        prefix: &str,
        group: &GroupNode,
        index: usize,
        inherited_order: i32,
    ) -> Result<String, GraphParseError> {
        let id = format!(
            "{prefix}:group:{}",
            group.id.clone().unwrap_or_else(|| format!("group_{index}"))
        );
        let (space, order, format, kind) = if let Some(composite) = &group.composite {
            (
                RenderPassSpace::parse(&composite.space),
                composite.composite_order.unwrap_or(inherited_order),
                composite.format.as_str(),
                if composite
                    .nodes_3d
                    .iter()
                    .any(|node| matches!(node, Scene3DNode::Model(_)))
                {
                    RenderPassDagKind::ThreeDIsland
                } else {
                    RenderPassDagKind::CompositeGroup
                },
            )
        } else {
            (
                RenderPassSpace::World,
                inherited_order,
                "rgba8unorm",
                RenderPassDagKind::Group,
            )
        };
        let mut inputs = Vec::new();
        self.collect_scene_node_outputs(&id, &group.children, &mut inputs, order)?;
        if let Some(composite) = &group.composite {
            for node in &composite.nodes_3d {
                match node {
                    Scene3DNode::Model(model) => {
                        self.validate_asset_reference(
                            &model.asset,
                            GraphAssetKind::Model,
                            "Model",
                        )?;
                        for binding in &model.material_bindings {
                            if let Some(texture) = &binding.texture
                                && let Some(scene_id) = scene_reference(texture)
                            {
                                if !self.scene_ids.contains(scene_id) {
                                    return Err(dag_error(format!(
                                        "MaterialBinding texture=\"{texture}\" references unknown Scene '{scene_id}'."
                                    )));
                                }
                                inputs.push(format!("scene:{scene_id}"));
                            }
                        }
                    }
                    Scene3DNode::EnvironmentLight(light) => self.validate_asset_reference(
                        &light.asset,
                        GraphAssetKind::Image,
                        "EnvironmentLight",
                    )?,
                    Scene3DNode::Camera(_) | Scene3DNode::Anchor(_) | Scene3DNode::Debug(_) => {}
                }
            }
        }
        self.push(RenderPassDagNode {
            id: id.clone(),
            kind,
            space,
            composite_order: order,
            format: format.to_string(),
            inputs,
        })?;
        self.effect_chain(
            &id,
            id.clone(),
            &group.process_effects,
            if group.composite.is_some() {
                RenderEffectScope::CompositeGroup
            } else {
                RenderEffectScope::Group
            },
            space,
            order,
            format,
        )
    }

    fn build_layer(
        &mut self,
        prefix: &str,
        layer: &SceneLayerNode,
        index: usize,
        order: i32,
    ) -> Result<String, GraphParseError> {
        let id = format!(
            "{prefix}:layer:{}",
            layer.id.clone().unwrap_or_else(|| format!("layer_{index}"))
        );
        let mut inputs = Vec::new();
        if let Some(source) = &layer.source
            && let Some(scene_id) = scene_reference(source)
        {
            if !self.scene_ids.contains(scene_id) {
                return Err(dag_error(format!(
                    "Layer source=\"{source}\" references unknown Scene '{scene_id}'."
                )));
            }
            inputs.push(format!("scene:{scene_id}"));
        }
        self.collect_scene_node_outputs(&id, &layer.children, &mut inputs, order)?;
        let space = layer
            .space
            .as_deref()
            .map(RenderPassSpace::parse)
            .unwrap_or(if layer.is_3d {
                RenderPassSpace::ThreeD
            } else {
                RenderPassSpace::World
            });
        self.push(RenderPassDagNode {
            id: id.clone(),
            kind: RenderPassDagKind::Layer,
            space,
            composite_order: order,
            format: "rgba8unorm".to_string(),
            inputs,
        })?;
        self.effect_chain(
            &id,
            id.clone(),
            &layer.process_effects,
            RenderEffectScope::Layer,
            space,
            order,
            "rgba8unorm",
        )
    }

    fn validate_asset_reference(
        &self,
        asset_id: &str,
        expected: GraphAssetKind,
        node_name: &str,
    ) -> Result<(), GraphParseError> {
        // Existing showcases commonly use direct file paths. Asset ids become
        // strict only when the Graph opts in by declaring an <Assets> block.
        if self.assets.is_empty() {
            return Ok(());
        }
        let Some(kind) = self.assets.get(asset_id) else {
            return Err(dag_error(format!(
                "<{node_name} asset=\"{asset_id}\"> references an unknown Graph asset."
            )));
        };
        if *kind != expected {
            return Err(dag_error(format!(
                "<{node_name} asset=\"{asset_id}\"> references the wrong asset kind."
            )));
        }
        Ok(())
    }

    fn build_process_passes(&mut self) -> Result<(), GraphParseError> {
        let mut resource_producer = HashMap::<String, String>::new();
        for pass in &self.graph.passes {
            let id = format!("process-pass:{}", pass.id);
            let inputs = pass
                .inputs
                .iter()
                .filter_map(|input| resource_producer.get(input.resource_id()).cloned())
                .collect::<Vec<_>>();
            self.push(RenderPassDagNode {
                id: id.clone(),
                kind: RenderPassDagKind::ProcessPass {
                    effect: pass.effect.clone(),
                },
                space: RenderPassSpace::Process,
                composite_order: 0,
                format: "process-defined".to_string(),
                inputs,
            })?;
            for output in &pass.outputs {
                resource_producer.insert(output.resource_id().to_string(), id.clone());
            }
        }
        Ok(())
    }
}

fn scene_reference(value: &str) -> Option<&str> {
    value
        .strip_prefix("@scene:")
        .or_else(|| value.strip_prefix("scene:"))
}

fn dag_error(message: String) -> GraphParseError {
    GraphParseError { line: 0, message }
}

pub fn compile_render_pass_dag(graph: &GraphScript) -> Result<RenderPassDag, GraphParseError> {
    let scene_ids = graph
        .scenes
        .iter()
        .map(|scene| scene.id.clone())
        .collect::<HashSet<_>>();
    if scene_ids.len() != graph.scenes.len() {
        return Err(dag_error(
            "Scene ids must be unique before compiling the Render Pass DAG.".to_string(),
        ));
    }
    let mut assets = HashMap::new();
    for asset in &graph.assets {
        if assets.insert(asset.id.clone(), asset.kind).is_some() {
            return Err(dag_error(format!(
                "Graph asset id '{}' is duplicated.",
                asset.id
            )));
        }
    }
    let mut builder = DagBuilder {
        graph,
        nodes: Vec::new(),
        ids: HashSet::new(),
        scene_ids,
        assets,
    };
    for scene in &graph.scenes {
        builder.build_scene(scene)?;
    }
    builder.build_process_passes()?;
    let present_input = if builder
        .ids
        .contains(&format!("scene:{}", graph.present.from))
    {
        format!("scene:{}", graph.present.from)
    } else {
        graph.present.from.clone()
    };
    builder.push(RenderPassDagNode {
        id: "present".to_string(),
        kind: RenderPassDagKind::Present,
        space: RenderPassSpace::Screen,
        composite_order: i32::MAX,
        format: "present".to_string(),
        inputs: vec![present_input],
    })?;

    let node_ids = builder
        .nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<HashSet<_>>();
    let edges = builder
        .nodes
        .iter()
        .flat_map(|node| {
            node.inputs
                .iter()
                .filter(|input| node_ids.contains(*input))
                .map(|input| RenderPassDagEdge {
                    from: input.clone(),
                    to: node.id.clone(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    validate_acyclic(&builder.nodes, &edges)?;
    Ok(RenderPassDag {
        nodes: builder.nodes,
        edges,
        output: "present".to_string(),
    })
}

fn validate_acyclic(
    nodes: &[RenderPassDagNode],
    edges: &[RenderPassDagEdge],
) -> Result<(), GraphParseError> {
    let mut indegree = nodes
        .iter()
        .map(|node| (node.id.clone(), 0usize))
        .collect::<HashMap<_, _>>();
    let mut outgoing = HashMap::<String, Vec<String>>::new();
    for edge in edges {
        *indegree.entry(edge.to.clone()).or_default() += 1;
        outgoing
            .entry(edge.from.clone())
            .or_default()
            .push(edge.to.clone());
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect::<Vec<_>>();
    let mut visited = 0usize;
    while let Some(id) = ready.pop() {
        visited += 1;
        if let Some(targets) = outgoing.get(&id) {
            for target in targets {
                if let Some(degree) = indegree.get_mut(target) {
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push(target.clone());
                    }
                }
            }
        }
    }
    if visited != nodes.len() {
        return Err(dag_error(
            "Render Pass DAG contains a cycle. Check Scene-to-texture references and Process resources."
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RenderEffectScope, RenderPassDagKind, RenderPassSpace, compile_render_pass_dag};
    use crate::dsl::parse_graph_script;

    fn unified_scene_script(screen_texture: &str) -> String {
        format!(
            r##"
<Graph fps={{30}} duration="4s" size={{[1920,1080]}}>
  <Assets>
    <ModelAsset id="phone_model" src="models/phone.glb" decoder="draco" />
    <ImageAsset id="studio_hdri" src="assets/studio.hdr" colorSpace="linear" />
  </Assets>
  <Process id="fx_grade">
    <Input id="effect_input" type="video" />
    <Tex id="effect_src" fmt="rgba16f" from="input:effect_input" />
    <Tex id="effect_out" fmt="rgba16f" size={{[1920,1080]}} />
    <Pass id="grade_pass" kind="compute" effect="brightness"
          in={{["effect_src"]}} out={{["effect_out"]}}
          params={{{{ brightness: "0.0" }}}} />
  </Process>
  <Scene id="phone_ui">
    <Timeline>
      <Track id="ui" space="screen" compositeOrder="0">
        <Sequence duration="4s">
          <Layer id="ui_layer" space="screen">
            <Rect x="0" y="0" width="1920" height="1080" color="#101827" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Scene id="product_promo">
    <Timeline>
      <Track id="product" space="3d" compositeOrder="20">
        <Sequence duration="4s">
          <CompositeGroup id="phone_island" space="3d" depth="true" format="rgba16f">
            <Camera3D id="camera" position={{[0,0,6]}} target={{[0,0,0]}} fov="35" />
            <EnvironmentLight asset="studio_hdri" intensity="1.2" />
            <Model id="phone" asset="phone_model">
              <MaterialBinding material="screen" texture="{screen_texture}" />
            </Model>
            <Effects>
              <Effect process="fx_grade" id="phone_grade">
                <Param name="brightness" value="0.1" />
              </Effect>
            </Effects>
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
    <PostEffects>
      <Effect process="fx_grade">
        <Param name="brightness" value="0.05" />
      </Effect>
    </PostEffects>
  </Scene>
  <Present from="product_promo" />
</Graph>
"##
        )
    }

    #[test]
    fn compiles_unified_scene_and_process_effect_scopes_to_one_dag() {
        let graph = parse_graph_script(&unified_scene_script("scene:phone_ui"))
            .expect("unified Scene graph should parse");
        let dag = compile_render_pass_dag(&graph).expect("unified DAG should compile");

        let island = dag
            .nodes
            .iter()
            .find(|node| node.id.ends_with(":group:phone_island"))
            .expect("3D island node");
        assert_eq!(island.kind, RenderPassDagKind::ThreeDIsland);
        assert_eq!(island.space, RenderPassSpace::ThreeD);
        assert_eq!(island.composite_order, 20);
        assert_eq!(island.format, "rgba16f");
        assert!(island.inputs.iter().any(|input| input == "scene:phone_ui"));

        assert!(dag.nodes.iter().any(|node| {
            matches!(
                node.kind,
                RenderPassDagKind::Effect {
                    scope: RenderEffectScope::CompositeGroup,
                    ..
                }
            )
        }));
        assert!(dag.nodes.iter().any(|node| {
            matches!(
                node.kind,
                RenderPassDagKind::Effect {
                    scope: RenderEffectScope::ScenePost,
                    ..
                }
            )
        }));
        assert!(
            dag.nodes
                .iter()
                .any(|node| { matches!(node.kind, RenderPassDagKind::ProcessPass { .. }) })
        );
    }

    #[test]
    fn rejects_scene_to_texture_cycles() {
        let mut script = unified_scene_script("scene:phone_ui");
        script = script.replace(
            "<Rect x=\"0\" y=\"0\" width=\"1920\" height=\"1080\" color=\"#101827\" />",
            "<Layer source=\"scene:product_promo\" />",
        );
        let error = parse_graph_script(&script).expect_err("Scene texture cycle must fail");
        assert!(error.message.contains("cycle"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_unknown_declared_asset_reference() {
        let script = unified_scene_script("scene:phone_ui")
            .replace("asset=\"phone_model\"", "asset=\"missing_phone\"");
        let error = parse_graph_script(&script).expect_err("unknown ModelAsset must fail");
        assert!(
            error.message.contains("unknown Graph asset"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn anonymous_layers_in_separate_sequences_receive_unique_dag_ids() {
        let script = r##"
<Graph fps={30} duration="4s" size={[640,360]}>
  <Scene id="chapters">
    <Timeline>
      <Track id="story">
        <Sequence from="0s" duration="2s" out="hide">
          <Layer>
            <Text value="ONE" x="20" y="40" />
          </Layer>
        </Sequence>
        <Sequence from="2s" duration="2s" out="hold">
          <Layer>
            <Text value="TWO" x="20" y="40" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="chapters" />
</Graph>
"##;
        let graph = parse_graph_script(script)
            .expect("anonymous Layers in separate Sequences should not collide");
        let dag = compile_render_pass_dag(&graph).expect("DAG should compile");
        let layer_ids = dag
            .nodes
            .iter()
            .filter(|node| matches!(node.kind, RenderPassDagKind::Layer))
            .map(|node| node.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(layer_ids.len(), 2);
        assert_ne!(layer_ids[0], layer_ids[1]);
        assert!(layer_ids[0].contains(":sequence:0:"));
        assert!(layer_ids[1].contains(":sequence:1:"));
    }
}
