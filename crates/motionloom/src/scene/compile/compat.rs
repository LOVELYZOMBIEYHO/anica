// =========================================
// =========================================
// crates/motionloom/src/scene/compile/compat.rs

use crate::dsl::GraphScript;
use crate::scene::model::SceneNode;

pub(crate) fn scene_nodes_for_present(graph: &GraphScript) -> Option<&[SceneNode]> {
    let present = graph.present.from.as_str();
    if present == "scene" {
        if !graph.scene_nodes.is_empty() {
            return Some(&graph.scene_nodes);
        }
        return graph.scenes.first().map(|scene| scene.children.as_slice());
    }
    let scene_id = present.strip_prefix("scene:").unwrap_or(present);
    graph
        .scenes
        .iter()
        .find(|scene| scene.id == scene_id)
        .map(|scene| scene.children.as_slice())
}

pub(crate) fn scene_nodes_contain_image_or_svg(nodes: &[SceneNode]) -> bool {
    nodes.iter().any(|node| match node {
        SceneNode::Image(_) | SceneNode::Svg(_) => true,
        SceneNode::Timeline(timeline) => scene_nodes_contain_image_or_svg(&timeline.children),
        SceneNode::Track(track) => scene_nodes_contain_image_or_svg(&track.children),
        SceneNode::Sequence(sequence) => scene_nodes_contain_image_or_svg(&sequence.children),
        SceneNode::Chain(chain) => scene_nodes_contain_image_or_svg(&chain.children),
        SceneNode::Group(group) => scene_nodes_contain_image_or_svg(&group.children),
        SceneNode::Puppet(puppet) => scene_nodes_contain_image_or_svg(&puppet.children),
        SceneNode::Part(part) => scene_nodes_contain_image_or_svg(&part.children),
        SceneNode::Repeat(repeat) => scene_nodes_contain_image_or_svg(&repeat.children),
        SceneNode::Camera(camera) => scene_nodes_contain_image_or_svg(&camera.children),
        SceneNode::Character(character) => scene_nodes_contain_image_or_svg(&character.children),
        SceneNode::Mask(mask) => scene_nodes_contain_image_or_svg(&mask.children),
        SceneNode::Precompose(precompose) => scene_nodes_contain_image_or_svg(&precompose.children),
        SceneNode::Layer(layer) => scene_nodes_contain_image_or_svg(&layer.children),
        _ => false,
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn scene_nodes_contain_3d_island(nodes: &[SceneNode]) -> bool {
    nodes.iter().any(|node| match node {
        SceneNode::Timeline(timeline) => scene_nodes_contain_3d_island(&timeline.children),
        SceneNode::Track(track) => scene_nodes_contain_3d_island(&track.children),
        SceneNode::Sequence(sequence) => scene_nodes_contain_3d_island(&sequence.children),
        SceneNode::Chain(chain) => scene_nodes_contain_3d_island(&chain.children),
        SceneNode::Group(group) => {
            group.composite.as_ref().is_some_and(|composite| {
                composite.space.eq_ignore_ascii_case("3d") && !composite.nodes_3d.is_empty()
            }) || scene_nodes_contain_3d_island(&group.children)
        }
        SceneNode::Puppet(puppet) => scene_nodes_contain_3d_island(&puppet.children),
        SceneNode::Part(part) => scene_nodes_contain_3d_island(&part.children),
        SceneNode::Repeat(repeat) => scene_nodes_contain_3d_island(&repeat.children),
        SceneNode::Camera(camera) => scene_nodes_contain_3d_island(&camera.children),
        SceneNode::Character(character) => scene_nodes_contain_3d_island(&character.children),
        SceneNode::Mask(mask) => scene_nodes_contain_3d_island(&mask.children),
        SceneNode::Precompose(precompose) => scene_nodes_contain_3d_island(&precompose.children),
        SceneNode::Layer(layer) => scene_nodes_contain_3d_island(&layer.children),
        _ => false,
    })
}

pub(crate) fn scene_nodes_require_cpu_scene_compositing(nodes: &[SceneNode]) -> bool {
    nodes.iter().any(|node| match node {
        SceneNode::Precompose(_) => true,
        SceneNode::Layer(layer) => {
            layer.is_3d
                || layer.source.is_some()
                || layer.mask.is_some()
                || layer.mask_from.is_some()
                || layer.matte.is_some()
                || layer.matte_from.is_some()
                || layer.effect.is_some()
                || !is_default_scene_number(&layer.opacity)
                || !is_normal_blend_name(&layer.blend)
                || scene_nodes_require_cpu_scene_compositing(&layer.children)
        }
        SceneNode::Timeline(timeline) => {
            scene_nodes_require_cpu_scene_compositing(&timeline.children)
        }
        SceneNode::Track(track) => scene_nodes_require_cpu_scene_compositing(&track.children),
        SceneNode::Sequence(sequence) => {
            scene_nodes_require_cpu_scene_compositing(&sequence.children)
        }
        SceneNode::Chain(chain) => scene_nodes_require_cpu_scene_compositing(&chain.children),
        SceneNode::Group(group) => {
            // A 3D island becomes a texture layer, so the shape-only fast path
            // must not discard its compiler-only Camera3D and Model nodes.
            group.composite.as_ref().is_some_and(|composite| {
                composite.space.eq_ignore_ascii_case("3d") && !composite.nodes_3d.is_empty()
            }) || group.mask.is_some()
                || group.mask_from.is_some()
                || scene_nodes_require_cpu_scene_compositing(&group.children)
        }
        SceneNode::Puppet(_) => true,
        SceneNode::Mask(mask) => {
            mask.feather.trim() != "0" || scene_nodes_require_cpu_scene_compositing(&mask.children)
        }
        SceneNode::Part(part) => scene_nodes_require_cpu_scene_compositing(&part.children),
        SceneNode::Repeat(repeat) => scene_nodes_require_cpu_scene_compositing(&repeat.children),
        SceneNode::Camera(camera) => scene_nodes_require_cpu_scene_compositing(&camera.children),
        SceneNode::Character(character) => {
            scene_nodes_require_cpu_scene_compositing(&character.children)
        }
        _ => false,
    })
}

fn is_default_scene_number(value: &str) -> bool {
    matches!(value.trim(), "1" | "1.0" | "1.00" | "1.000")
}

fn is_normal_blend_name(value: &str) -> bool {
    let value = value.trim();
    value.is_empty() || value.eq_ignore_ascii_case("normal")
}

pub(crate) fn graph_has_rich_scene_tree(graph: &GraphScript) -> bool {
    !graph.scenes.is_empty() || graph.scene_nodes.iter().any(scene_node_is_rich)
}

fn scene_node_is_rich(node: &SceneNode) -> bool {
    match node {
        SceneNode::Defs(_)
        | SceneNode::Palette(_)
        | SceneNode::Text(_)
        | SceneNode::Image(_)
        | SceneNode::Svg(_) => false,
        SceneNode::Rect(_)
        | SceneNode::PixelGrid(_)
        | SceneNode::Circle(_)
        | SceneNode::Ellipse(_)
        | SceneNode::Line(_)
        | SceneNode::Polyline(_)
        | SceneNode::Path(_)
        | SceneNode::FaceJaw(_)
        | SceneNode::Shadow(_)
        | SceneNode::Mask(_)
        | SceneNode::Use(_)
        | SceneNode::Layer(_) => true,
        SceneNode::Precompose(precompose) => precompose.children.iter().any(scene_node_is_rich),
        SceneNode::Timeline(timeline) => timeline.children.iter().any(scene_node_is_rich),
        SceneNode::Track(track) => track.children.iter().any(scene_node_is_rich),
        SceneNode::Sequence(sequence) => sequence.children.iter().any(scene_node_is_rich),
        SceneNode::Chain(chain) => chain.children.iter().any(scene_node_is_rich),
        SceneNode::Group(group) => group.children.iter().any(scene_node_is_rich),
        SceneNode::Puppet(puppet) => puppet.children.iter().any(scene_node_is_rich),
        SceneNode::Part(part) => part.children.iter().any(scene_node_is_rich),
        SceneNode::Repeat(repeat) => repeat.children.iter().any(scene_node_is_rich),
        SceneNode::Pin(_)
        | SceneNode::LimbEnvelope(_)
        | SceneNode::LimbRegion(_)
        | SceneNode::MeshTopology(_)
        | SceneNode::Vertex(_)
        | SceneNode::Triangle(_)
        | SceneNode::Edge(_)
        | SceneNode::Region(_) => false,
        SceneNode::Simulation(_) => false,
        SceneNode::Camera(_) | SceneNode::Character(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::{scene_nodes_for_present, scene_nodes_require_cpu_scene_compositing};
    use crate::dsl::parse_graph_script;

    #[test]
    fn scene_3d_composite_bypasses_shape_only_fast_path() {
        let graph = parse_graph_script(
            r#"<Graph fps={30} duration="1s" size={[320,180]}>
  <Assets>
    <ModelAsset id="phone" src="phone.glb" />
  </Assets>
  <Scene id="stage">
    <Timeline>
      <Track id="model" space="3d">
        <Sequence from="0s" duration="1s" out="hold">
          <CompositeGroup id="phone_island" space="3d" depth="true">
            <Camera3D position={[0,0,6]} target={[0,0,0]} />
            <Model asset="phone" scale="1" />
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="stage" />
</Graph>"#,
        )
        .expect("3D scene graph");
        let nodes = scene_nodes_for_present(&graph).expect("presented scene");

        assert!(scene_nodes_require_cpu_scene_compositing(nodes));
    }
}
