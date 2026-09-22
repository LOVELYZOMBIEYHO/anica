// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/capabilities.rs

use std::collections::HashSet;

use super::{RenderDomain, RequiredAovs, SceneCompositionError, SceneCompositionPlan};

/// An executor advertises support before any expensive renderer starts.
#[derive(Debug, Clone)]
pub struct CompositorCapabilities {
    pub domains: HashSet<RenderDomain>,
    pub effects: HashSet<String>,
    pub aovs: RequiredAovs,
    pub multiple_islands: bool,
}

impl CompositorCapabilities {
    pub fn validate(&self, plan: &SceneCompositionPlan) -> Result<(), SceneCompositionError> {
        plan.validate()?;
        let mut islands = 0usize;
        for layer in &plan.layers {
            if !self.domains.contains(&layer.domain) {
                return Err(SceneCompositionError::UnsupportedDomain {
                    layer: layer.id.clone(),
                    domain: format!("{:?}", layer.domain),
                });
            }
            islands += usize::from(layer.domain == RenderDomain::ThreeD);
            for effect in &layer.effects {
                if !self.effects.contains(effect) {
                    return Err(SceneCompositionError::UnsupportedEffect {
                        layer: layer.id.clone(),
                        effect: effect.clone(),
                    });
                }
            }
            if !self.aovs.contains(layer.required_aovs) {
                return Err(SceneCompositionError::MissingAov {
                    layer: layer.id.clone(),
                    aov: "coverage/depth/motion/normal/albedo".into(),
                });
            }
        }
        if islands > 1 && !self.multiple_islands {
            return Err(SceneCompositionError::UnsupportedDomain {
                layer: "multiple 3D islands".into(),
                domain: "three_d".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::compositor::{
        AlphaMode, ColorStage, CompositionLayer, CompositionSource, SourceColorSpace,
        premultiplied_over, to_linear_premultiplied,
    };

    fn layer(id: &str, order: i32) -> CompositionLayer {
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
            effects: Vec::new(),
            required_aovs: RequiredAovs::default(),
        }
    }

    #[test]
    fn plan_sort_is_stable_and_serializable() {
        let mut plan = SceneCompositionPlan::new([1920, 1080], [1920, 1080]);
        plan.layers = vec![
            layer("titles", 90),
            layer("world", 30),
            layer("weather", 75),
        ];
        plan.sort_layers();
        assert_eq!(
            plan.layers
                .iter()
                .map(|layer| layer.id.as_str())
                .collect::<Vec<_>>(),
            ["world", "weather", "titles"]
        );
        let json = serde_json::to_string(&plan).unwrap();
        assert_eq!(
            serde_json::from_str::<SceneCompositionPlan>(&json).unwrap(),
            plan
        );
    }

    #[test]
    fn color_conversion_and_over_keep_premultiplied_edges() {
        let foreground = to_linear_premultiplied(
            [1.0, 0.0, 0.0, 0.5],
            SourceColorSpace::Srgb,
            AlphaMode::Straight,
        );
        let result = premultiplied_over(foreground, [0.0, 0.0, 1.0, 1.0]);
        assert!((result[0] - 0.5).abs() < 0.0001);
        assert!((result[2] - 0.5).abs() < 0.0001);
        assert_eq!(result[3], 1.0);
    }

    #[test]
    fn capabilities_reject_missing_motion_aov() {
        let mut plan = SceneCompositionPlan::new([16, 16], [16, 16]);
        let mut rain = layer("lens-rain", 90);
        rain.required_aovs.motion = true;
        plan.layers.push(rain);
        let capabilities = CompositorCapabilities {
            domains: HashSet::from([RenderDomain::Screen]),
            effects: HashSet::new(),
            aovs: RequiredAovs::default(),
            multiple_islands: false,
        };
        assert!(matches!(
            capabilities.validate(&plan),
            Err(SceneCompositionError::MissingAov { .. })
        ));
    }
}
