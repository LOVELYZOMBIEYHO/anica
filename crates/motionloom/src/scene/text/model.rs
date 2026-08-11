use serde::{Deserialize, Serialize};

use super::animator::TextAnimatorNode;
use super::layout::TextLayoutNode;

fn default_text_scene_zero() -> String {
    "0".to_string()
}

fn default_text_scene_one() -> String {
    "1".to_string()
}

fn default_text_render_scale() -> String {
    "auto".to_string()
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TextNode {
    pub id: Option<String>,
    pub value: String,
    pub x: String,
    pub y: String,
    #[serde(default = "default_text_scene_zero")]
    pub rotation: String,
    #[serde(default = "default_text_scene_one")]
    pub scale: String,
    #[serde(default = "default_text_scene_one")]
    pub scale_x: String,
    #[serde(default = "default_text_scene_one")]
    pub scale_y: String,
    #[serde(default = "default_text_scene_zero")]
    pub skew_x: String,
    #[serde(default = "default_text_scene_zero")]
    pub skew_y: String,
    #[serde(default = "default_text_scene_zero")]
    pub transform_origin_x: String,
    #[serde(default = "default_text_scene_zero")]
    pub transform_origin_y: String,
    pub width: Option<String>,
    pub max_width: Option<String>,
    pub align: Option<String>,
    pub tracking: Option<String>,
    pub font_size: String,
    #[serde(default = "default_text_render_scale")]
    pub render_scale: String,
    pub antialias: Option<String>,
    pub edge_smoothing: Option<String>,
    pub blur: Option<String>,
    pub soft_edge: Option<String>,
    pub line_height: Option<String>,
    pub color: String,
    pub opacity: String,
    #[serde(rename = "box")]
    pub box_style: Option<String>,
    pub box_color: Option<String>,
    pub box_padding: Option<String>,
    pub box_padding_x: Option<String>,
    pub box_padding_y: Option<String>,
    pub box_radius: Option<String>,
    pub stroke: Option<String>,
    pub stroke_width: Option<String>,
    pub stroke_join: Option<String>,
    pub stroke_position: Option<String>,
    pub visible_chars: Option<String>,
    pub max_lines: Option<String>,
    pub font: Option<String>,
    pub font_family: Option<String>,
    pub font_weight: Option<String>,
    pub font_path: Option<String>,
    #[serde(default)]
    pub layout: Option<TextLayoutNode>,
    #[serde(default)]
    pub animators: Vec<TextAnimatorNode>,
}

impl TextNode {
    pub fn layout_width_expr(&self) -> Option<&str> {
        self.width
            .as_deref()
            .or(self.max_width.as_deref())
            .filter(|value| !value.trim().is_empty())
    }

    pub fn max_lines_expr(&self) -> Option<&str> {
        self.layout
            .as_ref()
            .and_then(|layout| layout.max_lines.as_deref())
            .or(self.max_lines.as_deref())
            .filter(|value| !value.trim().is_empty())
    }
}
