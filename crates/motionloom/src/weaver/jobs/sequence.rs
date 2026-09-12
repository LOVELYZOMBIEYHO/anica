// =========================================
// =========================================
// crates/motionloom/src/weaver/jobs/sequence.rs

use crate::weaver::{CancellationToken, RenderJob, RenderReport, SequenceProgress, WeaverError};
use std::ops::RangeInclusive;

/// Sequential frame export shares the job contract and cancellation. Per-frame
/// checkpoints remain resumable; cross-frame acceleration caching is separate work.
pub async fn render_sequence<F: FnMut(SequenceProgress)>(
    job: &RenderJob,
    frames: RangeInclusive<u32>,
    cancel: &CancellationToken,
    mut progress: F,
) -> Result<Vec<RenderReport>, WeaverError> {
    if frames.is_empty() {
        return Err(WeaverError::Invalid("empty frame range".into()));
    }
    let total = frames
        .end()
        .checked_sub(*frames.start())
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| WeaverError::Invalid("frame range overflow".into()))?;
    let mut reports = Vec::new();
    for frame in frames {
        if cancel.is_cancelled() {
            break;
        }
        let mut frame_job = job.clone();
        frame_job.frame = frame;
        let completed = reports.len() as u32;
        let report = super::render(&frame_job, cancel, |render| {
            progress(SequenceProgress {
                frame,
                completed_frames: completed,
                total_frames: total,
                render,
            })
        })
        .await?;
        let cancelled = report.status == "cancelled";
        reports.push(report);
        if cancelled {
            break;
        }
    }
    Ok(reports)
}
