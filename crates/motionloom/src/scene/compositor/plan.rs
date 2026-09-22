// =========================================
// =========================================
// crates/motionloom/src/scene/compositor/plan.rs

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{
    AlphaMode, ColorStage, CompositionPixelFormat, RenderDomain, SceneCompositionError,
    SourceColorSpace,
};

pub const SCENE_COMPOSITION_PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequiredAovs {
    pub coverage: bool,
    pub depth: bool,
    pub motion: bool,
    pub normal: bool,
    pub albedo: bool,
}

impl RequiredAovs {
    pub const fn contains(self, required: Self) -> bool {
        (!required.coverage || self.coverage)
            && (!required.depth || self.depth)
            && (!required.motion || self.motion)
            && (!required.normal || self.normal)
            && (!required.albedo || self.albedo)
    }

    pub fn union(&mut self, other: Self) {
        self.coverage |= other.coverage;
        self.depth |= other.depth;
        self.motion |= other.motion;
        self.normal |= other.normal;
        self.albedo |= other.albedo;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositionSource {
    TwoDRun,
    ThreeDIsland,
    WorldCard,
    Surface,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionLayer {
    pub id: String,
    pub order: i32,
    pub domain: RenderDomain,
    pub source: CompositionSource,
    /// Row-major 2D affine matrix `[m00,m01,m02,m10,m11,m12]`.
    pub transform: [f32; 6],
    pub opacity: f32,
    pub blend: String,
    pub color_stage: ColorStage,
    pub source_color_space: SourceColorSpace,
    pub source_alpha: AlphaMode,
    #[serde(default)]
    pub effects: Vec<String>,
    #[serde(default)]
    pub required_aovs: RequiredAovs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionColorPipeline {
    pub working_format: CompositionPixelFormat,
    pub tone_mapping: String,
    pub output_color_space: String,
}

impl Default for CompositionColorPipeline {
    fn default() -> Self {
        Self {
            working_format: CompositionPixelFormat::default(),
            tone_mapping: "linear".into(),
            output_color_space: "srgb".into(),
        }
    }
}

/// Stable renderer-independent intermediate representation for one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneCompositionPlan {
    pub version: u32,
    pub logical_size: [u32; 2],
    pub output_size: [u32; 2],
    pub layers: Vec<CompositionLayer>,
    #[serde(default)]
    pub effects: Vec<String>,
    #[serde(default)]
    pub required_aovs: RequiredAovs,
    pub color_pipeline: CompositionColorPipeline,
}

impl SceneCompositionPlan {
    pub fn new(logical_size: [u32; 2], output_size: [u32; 2]) -> Self {
        Self {
            version: SCENE_COMPOSITION_PLAN_VERSION,
            logical_size,
            output_size,
            layers: Vec::new(),
            effects: Vec::new(),
            required_aovs: RequiredAovs::default(),
            color_pipeline: CompositionColorPipeline::default(),
        }
    }

    /// Canonical ordering is stable for equal authored order values.
    pub fn sort_layers(&mut self) {
        self.layers.sort_by_key(|layer| layer.order);
    }

    pub fn derive_required_aovs(&mut self) {
        let mut required = RequiredAovs::default();
        for layer in &self.layers {
            required.union(layer.required_aovs);
        }
        self.required_aovs = required;
    }

    pub fn validate(&self) -> Result<(), SceneCompositionError> {
        if self.version != SCENE_COMPOSITION_PLAN_VERSION {
            return Err(SceneCompositionError::UnsupportedVersion(self.version));
        }
        if self.output_size.contains(&0) {
            return Err(SceneCompositionError::EmptyOutput);
        }
        let mut ids = HashSet::new();
        for layer in &self.layers {
            if !ids.insert(&layer.id) {
                return Err(SceneCompositionError::DuplicateLayer(layer.id.clone()));
            }
        }
        Ok(())
    }
}
