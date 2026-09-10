// =========================================
// =========================================
// crates/motionloom/src/audio/mod.rs

//! Platform-independent, random-access audio editing. Hosts supply decoded PCM.
use crate::dsl::{GraphAssetKind, GraphParseError, GraphScript};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{PreparedAudio, prepare_audio};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioClipNode {
    pub id: String,
    pub asset: String,
    pub from: f64,
    pub duration: f64,
    pub source_in: f64,
    pub source_out: Option<f64>,
    pub looped: bool,
    pub gain_db: f64,
    pub pan: f64,
    pub playback_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioKeyNode {
    pub seconds: f64,
    pub value: f64,
    pub ease: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioTargetNode {
    pub node: String,
    pub property: String,
    pub keys: Vec<AudioKeyNode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioAssetRef {
    pub id: String,
    pub src: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioTimelinePlan {
    pub clips: Vec<AudioClipNode>,
    pub targets: Vec<AudioTargetNode>,
    pub assets: Vec<AudioAssetRef>,
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("invalid audio: {0}")]
    Invalid(String),
    #[error("missing decoded audio asset: {0}")]
    MissingAsset(String),
    #[error("audio I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("FFmpeg audio decode failed for {asset}: {message}")]
    Decode { asset: String, message: String },
}

/// Validate references before rendering; old graphs compile to an empty plan.
pub fn compile_audio_plan(graph: &GraphScript) -> Result<AudioTimelinePlan, GraphParseError> {
    let fail = |message: String| GraphParseError { line: 1, message };
    let mut ids = HashSet::new();
    let mut assets = Vec::new();
    for clip in &graph.audio_clips {
        if !ids.insert(clip.id.as_str()) {
            return Err(fail(format!("Duplicate AudioClip id: {}", clip.id)));
        }
        let asset = graph
            .assets
            .iter()
            .find(|a| a.id == clip.asset && a.kind == GraphAssetKind::Audio)
            .ok_or_else(|| {
                fail(format!(
                    "AudioClip {} requires AudioAsset {}",
                    clip.id, clip.asset
                ))
            })?;
        let src = asset
            .external_src()
            .ok_or_else(|| fail("AudioAsset must have an external source".into()))?;
        if !assets.iter().any(|a: &AudioAssetRef| a.id == asset.id) {
            assets.push(AudioAssetRef {
                id: asset.id.clone(),
                src: src.into(),
            });
        }
    }
    let mut channels = HashSet::new();
    for target in &graph.audio_targets {
        if !ids.contains(target.node.as_str()) {
            return Err(fail(format!(
                "AudioTarget references unknown AudioClip {}",
                target.node
            )));
        }
        if !channels.insert((&target.node, &target.property)) {
            return Err(fail(format!(
                "Duplicate AudioTarget channel {}.{}",
                target.node, target.property
            )));
        }
    }
    Ok(AudioTimelinePlan {
        clips: graph.audio_clips.clone(),
        targets: graph.audio_targets.clone(),
        assets,
    })
}

/// Decode adapters normalize to stereo, interleaved PCM; no codec dependency lives here.
pub struct AudioMixer {
    plan: AudioTimelinePlan,
    sample_rate: u32,
    assets: HashMap<String, Vec<f32>>,
}

impl AudioMixer {
    pub fn new(plan: AudioTimelinePlan, sample_rate: u32) -> Result<Self, AudioError> {
        if !(8000..=192000).contains(&sample_rate) {
            return Err(AudioError::Invalid(
                "sample rate must be 8000..192000 Hz".into(),
            ));
        }
        // Public Rust/JSON callers receive the same safety checks as parsed DSL callers.
        for clip in &plan.clips {
            if ![
                clip.from,
                clip.duration,
                clip.source_in,
                clip.gain_db,
                clip.pan,
                clip.playback_rate,
            ]
            .iter()
            .all(|v| v.is_finite())
                || clip.from < 0.0
                || clip.duration <= 0.0
                || clip.source_in < 0.0
                || !(-120.0..=24.0).contains(&clip.gain_db)
                || !(-1.0..=1.0).contains(&clip.pan)
                || !(0.125..=8.0).contains(&clip.playback_rate)
                || clip
                    .source_out
                    .is_some_and(|end| !end.is_finite() || end <= clip.source_in)
            {
                return Err(AudioError::Invalid(format!("invalid clip {}", clip.id)));
            }
        }
        let mut channels = HashSet::new();
        for target in &plan.targets {
            let bounds = match target.property.as_str() {
                "gainDb" => -120.0..=24.0,
                "pan" => -1.0..=1.0,
                "playbackRate" => 0.125..=8.0,
                _ => {
                    return Err(AudioError::Invalid(format!(
                        "unknown audio property {}",
                        target.property
                    )));
                }
            };
            if !plan.clips.iter().any(|c| c.id == target.node)
                || !channels.insert((&target.node, &target.property))
                || target.keys.is_empty()
                || target.keys.windows(2).any(|p| p[0].seconds >= p[1].seconds)
                || target.keys.iter().any(|k| {
                    !k.seconds.is_finite()
                        || k.seconds < 0.0
                        || !bounds.contains(&k.value)
                        || !matches!(
                            k.ease.as_str(),
                            "linear" | "step" | "ease_in" | "ease_out" | "ease_in_out"
                        )
                })
            {
                return Err(AudioError::Invalid(format!(
                    "invalid audio channel {}.{}",
                    target.node, target.property
                )));
            }
        }
        Ok(Self {
            plan,
            sample_rate,
            assets: HashMap::new(),
        })
    }
    pub fn add_asset(&mut self, id: &str, stereo: Vec<f32>) -> Result<(), AudioError> {
        if stereo.is_empty() || stereo.len() % 2 != 0 || stereo.iter().any(|x| !x.is_finite()) {
            return Err(AudioError::Invalid(format!(
                "{id}: expected finite interleaved stereo PCM"
            )));
        }
        if !self.plan.assets.iter().any(|a| a.id == id) {
            return Err(AudioError::Invalid(format!("unknown audio asset {id}")));
        }
        let end = stereo.len() as f64 / (2.0 * self.sample_rate as f64);
        for clip in self.plan.clips.iter().filter(|c| c.asset == id) {
            if clip.source_in >= end
                || clip
                    .source_out
                    .is_some_and(|out| out > end + 1.0 / self.sample_rate as f64)
            {
                return Err(AudioError::Invalid(format!(
                    "{}: source trim exceeds decoded duration",
                    clip.id
                )));
            }
        }
        self.assets.insert(id.into(), stereo);
        Ok(())
    }

    /// Sample by absolute output sample index so seeking/chunk size never changes the result.
    pub fn render(&self, start_sample: u64, frames: usize) -> Result<Vec<f32>, AudioError> {
        if frames > self.sample_rate as usize * 10 {
            return Err(AudioError::Invalid(
                "render chunks must be at most ten seconds".into(),
            ));
        }
        let mut output = vec![0.0; frames * 2];
        let sr = self.sample_rate as f64;
        for clip in &self.plan.clips {
            let pcm = self
                .assets
                .get(&clip.asset)
                .ok_or_else(|| AudioError::MissingAsset(clip.asset.clone()))?;
            let gain = self.channel(&clip.id, "gainDb");
            let pan = self.channel(&clip.id, "pan");
            let rate = self.channel(&clip.id, "playbackRate");
            let source_end = clip.source_out.unwrap_or(pcm.len() as f64 / (2.0 * sr));
            let span = source_end - clip.source_in;
            for frame in 0..frames {
                let t = (start_sample as f64 + frame as f64) / sr;
                if t < clip.from || t >= clip.from + clip.duration {
                    continue;
                }
                let elapsed = integral(rate, clip.from, t, clip.playback_rate);
                if !clip.looped && elapsed >= span {
                    continue;
                }
                let position = (clip.source_in
                    + if clip.looped {
                        elapsed.rem_euclid(span)
                    } else {
                        elapsed
                    })
                    * sr;
                let i = position.floor() as usize;
                let mix = (position - i as f64) as f32;
                let next = if (i + 1) as f64 >= source_end * sr {
                    if clip.looped {
                        (clip.source_in * sr).floor() as usize
                    } else {
                        i
                    }
                } else {
                    i + 1
                };
                let volume = 10.0_f64.powf(value(gain, t, clip.gain_db) / 20.0) as f32;
                // Stereo balance preserves unity at center and silences the opposite channel at each extreme.
                let balance = value(pan, t, clip.pan) as f32;
                for ch in 0..2 {
                    let a = pcm.get(i * 2 + ch).copied().unwrap_or(0.0);
                    let b = pcm.get(next * 2 + ch).copied().unwrap_or(a);
                    let weight = if ch == 0 {
                        1.0 - balance.max(0.0)
                    } else {
                        1.0 + balance.min(0.0)
                    };
                    output[frame * 2 + ch] += (a + (b - a) * mix) * volume * weight;
                }
            }
        }
        // Both backends use this same explicit hard clipping policy for overloads.
        output.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        Ok(output)
    }
    fn channel(&self, node: &str, property: &str) -> &[AudioKeyNode] {
        self.plan
            .targets
            .iter()
            .find(|t| t.node == node && t.property == property)
            .map(|t| t.keys.as_slice())
            .unwrap_or(&[])
    }
}

fn ease(x: f64, kind: &str) -> f64 {
    match kind {
        "step" => 0.0,
        "ease_in" => x * x,
        "ease_out" => 2.0 * x - x * x,
        "ease_in_out" if x < 0.5 => 2.0 * x * x,
        "ease_in_out" => -1.0 + 4.0 * x - 2.0 * x * x,
        _ => x,
    }
}
fn ease_integral(x: f64, kind: &str) -> f64 {
    match kind {
        "step" => 0.0,
        "ease_in" => x.powi(3) / 3.0,
        "ease_out" => x * x - x.powi(3) / 3.0,
        "ease_in_out" if x < 0.5 => 2.0 * x.powi(3) / 3.0,
        "ease_in_out" => -x + 2.0 * x * x - 2.0 * x.powi(3) / 3.0 + 1.0 / 6.0,
        _ => x * x / 2.0,
    }
}
fn value(keys: &[AudioKeyNode], t: f64, default: f64) -> f64 {
    let Some(first) = keys.first() else {
        return default;
    };
    if t < first.seconds {
        return first.value;
    }
    for pair in keys.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if t < b.seconds {
            return a.value
                + (b.value - a.value) * ease((t - a.seconds) / (b.seconds - a.seconds), &b.ease);
        }
    }
    keys.last().unwrap().value
}
/// Integrating rate curves preserves source position during arbitrary seeks and variable speed.
fn integral(keys: &[AudioKeyNode], start: f64, end: f64, default: f64) -> f64 {
    if keys.is_empty() {
        return (end - start) * default;
    }
    let mut cursor = start;
    let mut sum = 0.0;
    if cursor < keys[0].seconds {
        let stop = end.min(keys[0].seconds);
        sum += (stop - cursor) * keys[0].value;
        cursor = stop;
    }
    for pair in keys.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let lo = cursor.max(a.seconds);
        let hi = end.min(b.seconds);
        if hi > lo {
            let d = b.seconds - a.seconds;
            sum += a.value * (hi - lo)
                + (b.value - a.value)
                    * d
                    * (ease_integral((hi - a.seconds) / d, &b.ease)
                        - ease_integral((lo - a.seconds) / d, &b.ease));
            cursor = hi;
        }
    }
    if end > cursor {
        sum += (end - cursor) * keys.last().unwrap().value;
    }
    sum
}
