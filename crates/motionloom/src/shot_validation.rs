//! Renderer-side, DSL-independent shot quality validation.
//!
//! The validator deliberately consumes observations instead of adding quality
//! policy to the MotionLoom language. Renderers, editors and CI hosts can all
//! feed the same engine, while checks without an observation are reported as
//! unavailable rather than silently passing.

use std::collections::{BTreeMap, BTreeSet};

use image::RgbaImage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ShotValidationCheck {
    Occlusion,
    Framing,
    Penetration,
    CameraCollision,
    Exposure,
    CameraPath,
    Composition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ShotValidationStatus {
    Passed,
    Warning,
    Failed,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ShotValidationSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ShotValidationOptions {
    pub frame_start: u32,
    pub frame_end: Option<u32>,
    pub sample_interval: u32,
    pub check_occlusion: bool,
    pub check_framing: bool,
    pub check_penetration: bool,
    pub check_camera_collision: bool,
    pub check_exposure: bool,
    pub check_camera_path: bool,
    pub check_composition: bool,
    pub safe_frame_margin: f32,
    pub minimum_joint_in_frame_ratio: f32,
    pub minimum_joint_visibility_ratio: f32,
    pub maximum_penetration: f32,
    pub minimum_camera_clearance: f32,
    pub dark_luminance_threshold: f32,
    pub bright_luminance_threshold: f32,
    pub maximum_dark_pixel_ratio: f32,
    pub maximum_bright_pixel_ratio: f32,
    /// Ratio where at least one display channel is saturated. This is useful
    /// diagnostic information for strongly coloured skies and emissives, but
    /// is not by itself evidence that neutral highlight detail is lost.
    pub maximum_clipped_pixel_ratio: f32,
    /// Ratio where every display channel is saturated, indicating genuinely
    /// white, detail-free highlight clipping.
    pub maximum_white_clipped_pixel_ratio: f32,
    pub minimum_average_luminance: f32,
    pub maximum_average_luminance: f32,
    pub minimum_composition_score: f32,
    pub minimum_luminance_histogram_bins: u32,
    pub maximum_dominant_luminance_ratio: f32,
}

impl Default for ShotValidationOptions {
    fn default() -> Self {
        Self::cinematic()
    }
}

impl ShotValidationOptions {
    pub fn cinematic() -> Self {
        Self {
            frame_start: 0,
            frame_end: None,
            sample_interval: 3,
            check_occlusion: true,
            check_framing: true,
            check_penetration: true,
            check_camera_collision: true,
            check_exposure: true,
            check_camera_path: true,
            check_composition: true,
            safe_frame_margin: 0.025,
            minimum_joint_in_frame_ratio: 0.9,
            minimum_joint_visibility_ratio: 0.75,
            maximum_penetration: 0.015,
            minimum_camera_clearance: 0.08,
            dark_luminance_threshold: 0.025,
            bright_luminance_threshold: 0.9,
            maximum_dark_pixel_ratio: 0.72,
            maximum_bright_pixel_ratio: 0.3,
            maximum_clipped_pixel_ratio: 0.25,
            maximum_white_clipped_pixel_ratio: 0.03,
            minimum_average_luminance: 0.035,
            maximum_average_luminance: 0.88,
            minimum_composition_score: 0.6,
            minimum_luminance_histogram_bins: 8,
            maximum_dominant_luminance_ratio: 0.98,
        }
    }

    pub fn realtime_preview() -> Self {
        Self {
            sample_interval: 10,
            check_penetration: false,
            check_camera_path: false,
            check_composition: false,
            ..Self::cinematic()
        }
    }

    pub fn ci_strict() -> Self {
        Self {
            sample_interval: 1,
            minimum_joint_in_frame_ratio: 0.97,
            minimum_joint_visibility_ratio: 0.9,
            maximum_penetration: 0.005,
            maximum_clipped_pixel_ratio: 0.15,
            maximum_white_clipped_pixel_ratio: 0.015,
            ..Self::cinematic()
        }
    }

    pub fn normalized(mut self) -> Self {
        self.sample_interval = self.sample_interval.max(1);
        self.safe_frame_margin = self.safe_frame_margin.clamp(0.0, 0.49);
        self.minimum_joint_in_frame_ratio = self.minimum_joint_in_frame_ratio.clamp(0.0, 1.0);
        self.minimum_joint_visibility_ratio = self.minimum_joint_visibility_ratio.clamp(0.0, 1.0);
        self.maximum_penetration = self.maximum_penetration.max(0.0);
        self.minimum_camera_clearance = self.minimum_camera_clearance.max(0.0);
        self.dark_luminance_threshold = self.dark_luminance_threshold.clamp(0.0, 1.0);
        self.bright_luminance_threshold = self.bright_luminance_threshold.clamp(0.0, 1.0);
        self.maximum_dark_pixel_ratio = self.maximum_dark_pixel_ratio.clamp(0.0, 1.0);
        self.maximum_bright_pixel_ratio = self.maximum_bright_pixel_ratio.clamp(0.0, 1.0);
        self.maximum_clipped_pixel_ratio = self.maximum_clipped_pixel_ratio.clamp(0.0, 1.0);
        self.maximum_white_clipped_pixel_ratio =
            self.maximum_white_clipped_pixel_ratio.clamp(0.0, 1.0);
        self.minimum_average_luminance = self.minimum_average_luminance.clamp(0.0, 1.0);
        self.maximum_average_luminance = self.maximum_average_luminance.clamp(0.0, 1.0);
        self.minimum_composition_score = self.minimum_composition_score.clamp(0.0, 1.0);
        self.minimum_luminance_histogram_bins = self.minimum_luminance_histogram_bins.max(1);
        self.maximum_dominant_luminance_ratio =
            self.maximum_dominant_luminance_ratio.clamp(0.0, 1.0);
        self
    }

    fn enabled(&self, check: ShotValidationCheck) -> bool {
        match check {
            ShotValidationCheck::Occlusion => self.check_occlusion,
            ShotValidationCheck::Framing => self.check_framing,
            ShotValidationCheck::Penetration => self.check_penetration,
            ShotValidationCheck::CameraCollision => self.check_camera_collision,
            ShotValidationCheck::Exposure => self.check_exposure,
            ShotValidationCheck::CameraPath => self.check_camera_path,
            ShotValidationCheck::Composition => self.check_composition,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectedJointObservation {
    pub actor: String,
    pub bone: String,
    /// Pixel coordinate in the rendered frame.
    pub x: f32,
    /// Pixel coordinate in the rendered frame.
    pub y: f32,
    /// Positive camera-space depth means the joint is in front of the camera.
    pub depth: f32,
    /// `None` means no ID/depth visibility sample was available.
    pub visible: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActorVisibilityObservation {
    pub actor: String,
    pub visible_samples: u32,
    pub total_samples: u32,
    #[serde(default)]
    pub occluding_object_ids: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PenetrationObservation {
    pub a: String,
    pub b: String,
    pub depth: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraClearanceObservation {
    pub obstacle: String,
    /// Signed clearance in scene units. Negative values mean overlap.
    pub clearance: f32,
    #[serde(default)]
    pub along_path: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompositionObservation {
    pub score: f32,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotValidationFrameObservation {
    pub frame: u32,
    pub width: u32,
    pub height: u32,
    /// Checks the producer actually evaluated for this frame. This allows an
    /// empty contact/visibility list to mean "observed and clear" instead of
    /// "data unavailable".
    #[serde(default)]
    pub observed_checks: Vec<ShotValidationCheck>,
    #[serde(default)]
    pub projected_joints: Vec<ProjectedJointObservation>,
    #[serde(default)]
    pub actor_visibility: Vec<ActorVisibilityObservation>,
    #[serde(default)]
    pub penetrations: Vec<PenetrationObservation>,
    #[serde(default)]
    pub camera_clearances: Vec<CameraClearanceObservation>,
    pub composition: Option<CompositionObservation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExposureObservation {
    pub frame: u32,
    pub average_luminance: f32,
    pub dark_pixel_ratio: f32,
    pub bright_pixel_ratio: f32,
    pub clipped_pixel_ratio: f32,
    #[serde(default)]
    pub white_clipped_pixel_ratio: f32,
    #[serde(default)]
    pub luminance_histogram_bins: u32,
    #[serde(default)]
    pub dominant_luminance_ratio: f32,
    #[serde(default)]
    pub luminance_standard_deviation: f32,
    pub luminance_histogram: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotValidationIssue {
    pub code: String,
    pub check: ShotValidationCheck,
    pub severity: ShotValidationSeverity,
    pub frame: Option<u32>,
    pub subject: Option<String>,
    pub message: String,
    pub measured: Option<f32>,
    pub threshold: Option<f32>,
    pub suggestion: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotValidationCheckReport {
    pub check: ShotValidationCheck,
    pub status: ShotValidationStatus,
    pub observed_frames: u32,
    pub issue_count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotValidationSummary {
    pub status: ShotValidationStatus,
    pub sampled_frames: u32,
    pub passed_checks: u32,
    pub warning_checks: u32,
    pub failed_checks: u32,
    pub unavailable_checks: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ShotValidationReport {
    pub version: u32,
    pub frame_start: u32,
    pub frame_end: u32,
    pub sample_interval: u32,
    pub summary: ShotValidationSummary,
    pub checks: Vec<ShotValidationCheckReport>,
    pub issues: Vec<ShotValidationIssue>,
    pub exposure: Vec<ExposureObservation>,
}

impl ShotValidationReport {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Clone, Debug)]
pub struct ShotValidator {
    options: ShotValidationOptions,
    frame_start: u32,
    frame_end: u32,
    sampled_frames: BTreeSet<u32>,
    observed: BTreeMap<ShotValidationCheck, BTreeSet<u32>>,
    issues: Vec<ShotValidationIssue>,
    exposure: Vec<ExposureObservation>,
}

impl ShotValidator {
    pub fn new(options: ShotValidationOptions, frame_start: u32, frame_end: u32) -> Self {
        Self {
            options: options.normalized(),
            frame_start,
            frame_end: frame_end.max(frame_start),
            sampled_frames: BTreeSet::new(),
            observed: BTreeMap::new(),
            issues: Vec::new(),
            exposure: Vec::new(),
        }
    }

    pub fn observe_rgba(&mut self, frame: u32, image: &RgbaImage) {
        self.sampled_frames.insert(frame);
        self.mark_observed(ShotValidationCheck::Exposure, frame);
        self.mark_observed(ShotValidationCheck::Composition, frame);
        let exposure = analyze_exposure(frame, image, &self.options);
        if exposure.average_luminance < self.options.minimum_average_luminance {
            self.push_issue(
                "SHOT_TOO_DARK",
                ShotValidationCheck::Exposure,
                ShotValidationSeverity::Warning,
                frame,
                None,
                format!("Average luminance is {:.3}.", exposure.average_luminance),
                exposure.average_luminance,
                self.options.minimum_average_luminance,
                "Increase key/fill light or exposure while preserving highlight headroom.",
            );
        } else if exposure.average_luminance > self.options.maximum_average_luminance {
            self.push_issue(
                "SHOT_TOO_BRIGHT",
                ShotValidationCheck::Exposure,
                ShotValidationSeverity::Warning,
                frame,
                None,
                format!("Average luminance is {:.3}.", exposure.average_luminance),
                exposure.average_luminance,
                self.options.maximum_average_luminance,
                "Reduce exposure or light intensity.",
            );
        }
        if exposure.dark_pixel_ratio > self.options.maximum_dark_pixel_ratio {
            self.push_issue(
                "EXCESSIVE_SHADOW_AREA",
                ShotValidationCheck::Exposure,
                ShotValidationSeverity::Warning,
                frame,
                None,
                format!(
                    "{:.1}% of pixels are below the dark threshold.",
                    exposure.dark_pixel_ratio * 100.0
                ),
                exposure.dark_pixel_ratio,
                self.options.maximum_dark_pixel_ratio,
                "Add fill light or reframe away from unlit background area.",
            );
        }
        if exposure.bright_pixel_ratio > self.options.maximum_bright_pixel_ratio {
            self.push_issue(
                "EXCESSIVE_HIGHLIGHT_AREA",
                ShotValidationCheck::Exposure,
                ShotValidationSeverity::Warning,
                frame,
                None,
                format!(
                    "{:.1}% of pixels are above the bright threshold.",
                    exposure.bright_pixel_ratio * 100.0
                ),
                exposure.bright_pixel_ratio,
                self.options.maximum_bright_pixel_ratio,
                "Reduce exposure or compress highlights.",
            );
        }
        if exposure.clipped_pixel_ratio > self.options.maximum_clipped_pixel_ratio {
            self.push_issue(
                "CHANNEL_CLIPPING",
                ShotValidationCheck::Exposure,
                ShotValidationSeverity::Info,
                frame,
                None,
                format!(
                    "{:.1}% of pixels have at least one saturated colour channel.",
                    exposure.clipped_pixel_ratio * 100.0
                ),
                exposure.clipped_pixel_ratio,
                self.options.maximum_clipped_pixel_ratio,
                "Inspect saturated colours; lower exposure only if visible colour detail is lost.",
            );
        }
        if exposure.white_clipped_pixel_ratio > self.options.maximum_white_clipped_pixel_ratio {
            self.push_issue(
                "WHITE_HIGHLIGHTS_CLIPPED",
                ShotValidationCheck::Exposure,
                ShotValidationSeverity::Warning,
                frame,
                None,
                format!(
                    "{:.1}% of pixels have all RGB channels clipped.",
                    exposure.white_clipped_pixel_ratio * 100.0
                ),
                exposure.white_clipped_pixel_ratio,
                self.options.maximum_white_clipped_pixel_ratio,
                "Lower exposure, direct-light intensity, or bloom while preserving coloured highlights.",
            );
        }
        if exposure.luminance_histogram_bins < self.options.minimum_luminance_histogram_bins
            || exposure.dominant_luminance_ratio > self.options.maximum_dominant_luminance_ratio
        {
            let measured = if exposure.dominant_luminance_ratio
                > self.options.maximum_dominant_luminance_ratio
            {
                exposure.dominant_luminance_ratio
            } else {
                exposure.luminance_histogram_bins as f32
            };
            let threshold = if exposure.dominant_luminance_ratio
                > self.options.maximum_dominant_luminance_ratio
            {
                self.options.maximum_dominant_luminance_ratio
            } else {
                self.options.minimum_luminance_histogram_bins as f32
            };
            self.push_issue(
                "LOW_VISUAL_VARIANCE",
                ShotValidationCheck::Composition,
                ShotValidationSeverity::Error,
                frame,
                None,
                format!(
                    "Rendered frame uses {} luminance bins and its dominant bin covers {:.1}% of pixels.",
                    exposure.luminance_histogram_bins,
                    exposure.dominant_luminance_ratio * 100.0
                ),
                measured,
                threshold,
                "Move the camera outside nearby geometry or restore visible subject/background separation.",
            );
        }
        self.exposure.push(exposure);
    }

    /// Record that a frame belongs to the sampling schedule even when a
    /// backend delegates every enabled check to later observations.
    pub fn mark_sampled_frame(&mut self, frame: u32) {
        self.sampled_frames.insert(frame);
    }

    pub fn observe_frame(&mut self, observation: ShotValidationFrameObservation) {
        let frame = observation.frame;
        self.sampled_frames.insert(frame);
        for check in &observation.observed_checks {
            self.mark_observed(*check, frame);
        }
        if !observation.projected_joints.is_empty() {
            self.mark_observed(ShotValidationCheck::Framing, frame);
            self.analyze_framing(&observation);
        }
        if !observation.actor_visibility.is_empty()
            || observation
                .projected_joints
                .iter()
                .any(|joint| joint.visible.is_some())
        {
            self.mark_observed(ShotValidationCheck::Occlusion, frame);
            self.analyze_visibility(&observation);
        }
        if !observation.penetrations.is_empty() {
            self.mark_observed(ShotValidationCheck::Penetration, frame);
            for contact in &observation.penetrations {
                if contact.depth > self.options.maximum_penetration {
                    self.push_issue(
                        "COLLIDER_PENETRATION",
                        ShotValidationCheck::Penetration,
                        ShotValidationSeverity::Error,
                        frame,
                        Some(format!("{} / {}", contact.a, contact.b)),
                        format!("Collider penetration is {:.4} scene units.", contact.depth),
                        contact.depth,
                        self.options.maximum_penetration,
                        "Adjust the contact anchor, collider profile, or transition root offset.",
                    );
                }
            }
        }
        for clearance in &observation.camera_clearances {
            let check = if clearance.along_path {
                ShotValidationCheck::CameraPath
            } else {
                ShotValidationCheck::CameraCollision
            };
            self.mark_observed(check, frame);
            if clearance.clearance < self.options.minimum_camera_clearance {
                self.push_issue(
                    if clearance.along_path {
                        "UNSAFE_CAMERA_PATH"
                    } else {
                        "CAMERA_INSIDE_COLLIDER"
                    },
                    check,
                    ShotValidationSeverity::Error,
                    frame,
                    Some(clearance.obstacle.clone()),
                    format!(
                        "Camera clearance is {:.4} scene units.",
                        clearance.clearance
                    ),
                    clearance.clearance,
                    self.options.minimum_camera_clearance,
                    "Offset the camera path along the collision normal and ease adjacent keys.",
                );
            }
        }
        if let Some(composition) = &observation.composition {
            self.mark_observed(ShotValidationCheck::Composition, frame);
            if composition.score < self.options.minimum_composition_score {
                self.push_issue(
                    "LOW_COMPOSITION_SCORE",
                    ShotValidationCheck::Composition,
                    ShotValidationSeverity::Warning,
                    frame,
                    None,
                    if composition.notes.is_empty() {
                        format!("Composition score is {:.2}.", composition.score)
                    } else {
                        composition.notes.join(" ")
                    },
                    composition.score,
                    self.options.minimum_composition_score,
                    "Review headroom, subject separation, foreground obstruction, and visual balance.",
                );
            }
        }
    }

    pub fn finish(mut self) -> ShotValidationReport {
        self.issues.sort_by(|a, b| {
            a.frame
                .cmp(&b.frame)
                .then_with(|| a.code.cmp(&b.code))
                .then_with(|| a.subject.cmp(&b.subject))
        });
        self.exposure.sort_by_key(|entry| entry.frame);
        let mut checks = Vec::new();
        for check in all_checks() {
            if !self.options.enabled(check) {
                continue;
            }
            let observed_frames = self.observed.get(&check).map_or(0, |frames| frames.len()) as u32;
            let relevant: Vec<_> = self
                .issues
                .iter()
                .filter(|issue| issue.check == check)
                .collect();
            let status = if observed_frames == 0 {
                ShotValidationStatus::Unavailable
            } else if relevant
                .iter()
                .any(|issue| issue.severity == ShotValidationSeverity::Error)
            {
                ShotValidationStatus::Failed
            } else if relevant
                .iter()
                .any(|issue| issue.severity == ShotValidationSeverity::Warning)
            {
                ShotValidationStatus::Warning
            } else {
                ShotValidationStatus::Passed
            };
            checks.push(ShotValidationCheckReport {
                check,
                status,
                observed_frames,
                issue_count: relevant.len() as u32,
            });
        }
        let passed_checks = count_status(&checks, ShotValidationStatus::Passed);
        let warning_checks = count_status(&checks, ShotValidationStatus::Warning);
        let failed_checks = count_status(&checks, ShotValidationStatus::Failed);
        let unavailable_checks = count_status(&checks, ShotValidationStatus::Unavailable);
        let status = if failed_checks > 0 {
            ShotValidationStatus::Failed
        } else if warning_checks > 0 {
            ShotValidationStatus::Warning
        } else if passed_checks == 0 && unavailable_checks > 0 {
            ShotValidationStatus::Unavailable
        } else if unavailable_checks > 0 {
            ShotValidationStatus::Warning
        } else {
            ShotValidationStatus::Passed
        };
        ShotValidationReport {
            version: 1,
            frame_start: self.frame_start,
            frame_end: self.frame_end,
            sample_interval: self.options.sample_interval,
            summary: ShotValidationSummary {
                status,
                sampled_frames: self.sampled_frames.len() as u32,
                passed_checks,
                warning_checks,
                failed_checks,
                unavailable_checks,
            },
            checks,
            issues: self.issues,
            exposure: self.exposure,
        }
    }

    fn analyze_framing(&mut self, observation: &ShotValidationFrameObservation) {
        let mut actors = BTreeMap::<&str, (u32, u32)>::new();
        let margin_x = observation.width as f32 * self.options.safe_frame_margin;
        let margin_y = observation.height as f32 * self.options.safe_frame_margin;
        for joint in &observation.projected_joints {
            let entry = actors.entry(&joint.actor).or_default();
            entry.1 += 1;
            if joint.depth > 0.0
                && joint.x >= margin_x
                && joint.x <= observation.width as f32 - margin_x
                && joint.y >= margin_y
                && joint.y <= observation.height as f32 - margin_y
            {
                entry.0 += 1;
            }
        }
        for (actor, (inside, total)) in actors {
            let ratio = inside as f32 / total.max(1) as f32;
            if ratio < self.options.minimum_joint_in_frame_ratio {
                self.push_issue(
                    "ACTOR_OUT_OF_FRAME",
                    ShotValidationCheck::Framing,
                    ShotValidationSeverity::Error,
                    observation.frame,
                    Some(actor.to_string()),
                    format!(
                        "Only {:.1}% of projected joints are inside the safe frame.",
                        ratio * 100.0
                    ),
                    ratio,
                    self.options.minimum_joint_in_frame_ratio,
                    "Widen the shot or move the camera target toward the actor bounds.",
                );
            }
        }
    }

    fn analyze_visibility(&mut self, observation: &ShotValidationFrameObservation) {
        let mut actors = BTreeMap::<String, (u32, u32)>::new();
        for joint in &observation.projected_joints {
            if let Some(visible) = joint.visible {
                let entry = actors.entry(joint.actor.clone()).or_default();
                entry.1 += 1;
                entry.0 += u32::from(visible);
            }
        }
        for visibility in &observation.actor_visibility {
            let entry = actors.entry(visibility.actor.clone()).or_default();
            entry.0 += visibility.visible_samples;
            entry.1 += visibility.total_samples;
        }
        for (actor, (visible, total)) in actors {
            if total == 0 {
                continue;
            }
            let ratio = visible as f32 / total as f32;
            if ratio < self.options.minimum_joint_visibility_ratio {
                self.push_issue(
                    "ACTOR_OCCLUDED",
                    ShotValidationCheck::Occlusion,
                    ShotValidationSeverity::Warning,
                    observation.frame,
                    Some(actor),
                    format!("Estimated actor visibility is {:.1}%.", ratio * 100.0),
                    ratio,
                    self.options.minimum_joint_visibility_ratio,
                    "Raise or orbit the camera, or clear foreground geometry from the sight line.",
                );
            }
        }
    }

    fn mark_observed(&mut self, check: ShotValidationCheck, frame: u32) {
        if self.options.enabled(check) {
            self.observed.entry(check).or_default().insert(frame);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_issue(
        &mut self,
        code: &str,
        check: ShotValidationCheck,
        severity: ShotValidationSeverity,
        frame: u32,
        subject: Option<String>,
        message: String,
        measured: f32,
        threshold: f32,
        suggestion: &str,
    ) {
        if self.options.enabled(check) {
            self.issues.push(ShotValidationIssue {
                code: code.to_string(),
                check,
                severity,
                frame: Some(frame),
                subject,
                message,
                measured: Some(measured),
                threshold: Some(threshold),
                suggestion: Some(suggestion.to_string()),
            });
        }
    }
}

pub fn analyze_shot_observations(
    options: ShotValidationOptions,
    observations: impl IntoIterator<Item = ShotValidationFrameObservation>,
) -> ShotValidationReport {
    let observations: Vec<_> = observations.into_iter().collect();
    let start = observations
        .iter()
        .map(|item| item.frame)
        .min()
        .unwrap_or(0);
    let end = observations
        .iter()
        .map(|item| item.frame)
        .max()
        .unwrap_or(start);
    let mut validator = ShotValidator::new(options, start, end);
    for observation in observations {
        validator.observe_frame(observation);
    }
    validator.finish()
}

/// Build a deterministic inclusive sampling schedule. The final frame is
/// always included so a transition or hold cannot escape validation merely
/// because it falls between regular samples.
pub fn shot_validation_sample_frames(start: u32, end: u32, interval: u32) -> Vec<u32> {
    let end = end.max(start);
    let interval = interval.max(1);
    let mut frames = Vec::new();
    let mut frame = start;
    loop {
        frames.push(frame);
        let Some(next) = frame.checked_add(interval) else {
            break;
        };
        if next > end {
            break;
        }
        frame = next;
    }
    if frames.last().copied() != Some(end) {
        frames.push(end);
    }
    frames
}

pub fn analyze_exposure(
    frame: u32,
    image: &RgbaImage,
    options: &ShotValidationOptions,
) -> ExposureObservation {
    const BINS: usize = 64;
    let mut histogram = vec![0_u32; BINS];
    let mut luminance_sum = 0.0_f64;
    let mut luminance_square_sum = 0.0_f64;
    let mut dark = 0_u64;
    let mut bright = 0_u64;
    let mut clipped = 0_u64;
    let mut white_clipped = 0_u64;
    let mut count = 0_u64;
    for pixel in image.pixels() {
        if pixel[3] == 0 {
            continue;
        }
        let r = srgb_to_linear(pixel[0] as f32 / 255.0);
        let g = srgb_to_linear(pixel[1] as f32 / 255.0);
        let b = srgb_to_linear(pixel[2] as f32 / 255.0);
        let luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
        histogram[((luminance * (BINS - 1) as f32).round() as usize).min(BINS - 1)] += 1;
        luminance_sum += luminance as f64;
        luminance_square_sum += (luminance * luminance) as f64;
        dark += u64::from(luminance <= options.dark_luminance_threshold);
        bright += u64::from(luminance >= options.bright_luminance_threshold);
        clipped += u64::from(pixel[0] >= 250 || pixel[1] >= 250 || pixel[2] >= 250);
        white_clipped += u64::from(pixel[0] >= 250 && pixel[1] >= 250 && pixel[2] >= 250);
        count += 1;
    }
    let denominator = count.max(1) as f32;
    let average_luminance = (luminance_sum / count.max(1) as f64) as f32;
    let variance = (luminance_square_sum / count.max(1) as f64
        - f64::from(average_luminance).powi(2))
    .max(0.0);
    let luminance_histogram_bins = histogram.iter().filter(|&&bin| bin > 0).count() as u32;
    let dominant_luminance_ratio =
        histogram.iter().copied().max().unwrap_or(0) as f32 / denominator;
    ExposureObservation {
        frame,
        average_luminance,
        dark_pixel_ratio: dark as f32 / denominator,
        bright_pixel_ratio: bright as f32 / denominator,
        clipped_pixel_ratio: clipped as f32 / denominator,
        white_clipped_pixel_ratio: white_clipped as f32 / denominator,
        luminance_histogram_bins,
        dominant_luminance_ratio,
        luminance_standard_deviation: variance.sqrt() as f32,
        luminance_histogram: histogram,
    }
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn all_checks() -> [ShotValidationCheck; 7] {
    [
        ShotValidationCheck::Occlusion,
        ShotValidationCheck::Framing,
        ShotValidationCheck::Penetration,
        ShotValidationCheck::CameraCollision,
        ShotValidationCheck::Exposure,
        ShotValidationCheck::CameraPath,
        ShotValidationCheck::Composition,
    ]
}

fn count_status(checks: &[ShotValidationCheckReport], status: ShotValidationStatus) -> u32 {
    checks.iter().filter(|check| check.status == status).count() as u32
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    #[test]
    fn black_frame_reports_dark_exposure() {
        let image = RgbaImage::from_pixel(16, 16, Rgba([0, 0, 0, 255]));
        let options = ShotValidationOptions::cinematic();
        let mut validator = ShotValidator::new(options, 4, 4);
        validator.observe_rgba(4, &image);
        let report = validator.finish();
        assert_eq!(report.summary.status, ShotValidationStatus::Failed);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "SHOT_TOO_DARK")
        );
        assert_eq!(report.exposure[0].luminance_histogram[0], 256);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "LOW_VISUAL_VARIANCE")
        );
    }

    #[test]
    fn coloured_channel_saturation_is_distinct_from_white_clipping() {
        let image = RgbaImage::from_pixel(16, 16, Rgba([0, 160, 255, 255]));
        let mut options = ShotValidationOptions::cinematic();
        options.maximum_clipped_pixel_ratio = 0.5;
        let exposure = analyze_exposure(2, &image, &options);
        assert_eq!(exposure.clipped_pixel_ratio, 1.0);
        assert_eq!(exposure.white_clipped_pixel_ratio, 0.0);
    }

    #[test]
    fn neutral_white_clipping_reports_warning() {
        let image = RgbaImage::from_pixel(16, 16, Rgba([255, 255, 255, 255]));
        let mut validator = ShotValidator::new(ShotValidationOptions::cinematic(), 3, 3);
        validator.observe_rgba(3, &image);
        let report = validator.finish();
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "WHITE_HIGHLIGHTS_CLIPPED")
        );
    }

    #[test]
    fn missing_observation_is_unavailable_not_passed() {
        let report = ShotValidator::new(ShotValidationOptions::cinematic(), 0, 10).finish();
        assert!(
            report
                .checks
                .iter()
                .all(|check| check.status == ShotValidationStatus::Unavailable)
        );
        assert_eq!(report.summary.status, ShotValidationStatus::Unavailable);
    }

    #[test]
    fn observations_cover_framing_visibility_collision_and_composition() {
        let observation = ShotValidationFrameObservation {
            frame: 8,
            width: 100,
            height: 100,
            projected_joints: vec![ProjectedJointObservation {
                actor: "hero".to_string(),
                bone: "head".to_string(),
                x: 120.0,
                y: 20.0,
                depth: 2.0,
                visible: Some(false),
            }],
            observed_checks: vec![
                ShotValidationCheck::Framing,
                ShotValidationCheck::Occlusion,
                ShotValidationCheck::Penetration,
                ShotValidationCheck::CameraPath,
                ShotValidationCheck::Composition,
            ],
            penetrations: vec![PenetrationObservation {
                a: "hero".to_string(),
                b: "bench".to_string(),
                depth: 0.1,
            }],
            camera_clearances: vec![CameraClearanceObservation {
                obstacle: "tree".to_string(),
                clearance: -0.2,
                along_path: true,
            }],
            composition: Some(CompositionObservation {
                score: 0.2,
                notes: vec!["Subject is hidden by foreground geometry.".to_string()],
            }),
            ..ShotValidationFrameObservation::default()
        };
        let report = analyze_shot_observations(ShotValidationOptions::cinematic(), [observation]);
        for code in [
            "ACTOR_OUT_OF_FRAME",
            "ACTOR_OCCLUDED",
            "COLLIDER_PENETRATION",
            "UNSAFE_CAMERA_PATH",
            "LOW_COMPOSITION_SCORE",
        ] {
            assert!(
                report.issues.iter().any(|issue| issue.code == code),
                "missing {code}"
            );
        }
    }

    #[test]
    fn options_json_round_trip_uses_camel_case() {
        let json = serde_json::to_string(&ShotValidationOptions::realtime_preview()).unwrap();
        assert!(json.contains("sampleInterval"));
        let decoded: ShotValidationOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.sample_interval, 10);
    }

    #[test]
    fn sampling_always_includes_last_frame() {
        assert_eq!(
            shot_validation_sample_frames(0, 10, 3),
            vec![0, 3, 6, 9, 10]
        );
        assert_eq!(shot_validation_sample_frames(5, 5, 0), vec![5]);
    }

    #[test]
    fn delegated_validation_retains_sample_count() {
        let mut validator = ShotValidator::new(ShotValidationOptions::cinematic(), 0, 6);
        for frame in shot_validation_sample_frames(0, 6, 3) {
            validator.mark_sampled_frame(frame);
        }
        assert_eq!(validator.finish().summary.sampled_frames, 3);
    }
}
