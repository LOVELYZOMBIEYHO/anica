//! High-level Scene authoring domains.
//!
//! Owns author-friendly concepts that lower into core Resource, Timeline,
//! Composition, Spatial, and Drawable nodes, such as Character, Part, Skeleton2D,
//! and future scene-specific actors.

mod action;
mod skeleton;
pub(crate) use action::{action_key_mix, apply_action_graph_at_time};
pub(crate) use skeleton::prepare_skeleton;
pub use skeleton::{
    ProportionProfile, SkeletonDiagnostic, SkeletonDiagnosticSeverity, SkeletonOverlayPrimitive,
    SkeletonPosePreset, SkeletonValidationReport, auto_correct_skeleton, build_skeleton_overlay,
    builtin_proportion_profile, builtin_proportion_profiles, builtin_skeleton_pose_presets,
    validate_skeleton,
};
