// =========================================
// =========================================
// crates/motionloom/src/simulation/compile.rs

use crate::scene::model::{SceneNode, SceneRootNode};
use crate::simulation::error::SimulationError;
use crate::simulation::model::SpringChainNode;

pub fn validate_scene_simulation(scene: &SceneRootNode) -> Result<(), SimulationError> {
    let mut targets = Vec::new();
    collect_spring_target_ids(&scene.children, &mut targets);
    let mut bindings = Vec::new();
    collect_spring_chains(&scene.children, &mut bindings);
    for binding in bindings {
        if !targets.iter().any(|id| id == &binding.target) {
            return Err(SimulationError::MissingTarget {
                id: binding.target.clone(),
            });
        }
    }
    Ok(())
}

fn collect_spring_target_ids(nodes: &[SceneNode], out: &mut Vec<String>) {
    for node in nodes {
        match node {
            SceneNode::Polyline(node) => {
                if let Some(id) = &node.id {
                    out.push(id.clone());
                }
            }
            SceneNode::Puppet(node) if node.solver.eq_ignore_ascii_case("chain") => {
                if let Some(id) = &node.id {
                    out.push(id.clone());
                }
            }
            SceneNode::Timeline(node) => collect_spring_target_ids(&node.children, out),
            SceneNode::Track(node) => collect_spring_target_ids(&node.children, out),
            SceneNode::Sequence(node) => collect_spring_target_ids(&node.children, out),
            SceneNode::Layer(node) => collect_spring_target_ids(&node.children, out),
            SceneNode::Group(node) => collect_spring_target_ids(&node.children, out),
            SceneNode::Part(node) => collect_spring_target_ids(&node.children, out),
            _ => {}
        }
    }
}

fn collect_spring_chains<'a>(nodes: &'a [SceneNode], out: &mut Vec<&'a SpringChainNode>) {
    for node in nodes {
        match node {
            SceneNode::Simulation(
                crate::simulation::model::SimulationBindingNode::SpringChain(binding),
            ) => out.push(binding),
            SceneNode::Timeline(node) => collect_spring_chains(&node.children, out),
            SceneNode::Track(node) => collect_spring_chains(&node.children, out),
            SceneNode::Sequence(node) => collect_spring_chains(&node.children, out),
            SceneNode::Layer(node) => collect_spring_chains(&node.children, out),
            SceneNode::Group(node) => collect_spring_chains(&node.children, out),
            SceneNode::Part(node) => collect_spring_chains(&node.children, out),
            _ => {}
        }
    }
}
