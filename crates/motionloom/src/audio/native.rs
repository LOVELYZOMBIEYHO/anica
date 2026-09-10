// =========================================
// =========================================
// crates/motionloom/src/audio/native.rs

use super::{AudioError, AudioMixer, AudioTimelinePlan};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_AUDIO: AtomicU64 = AtomicU64::new(0);

/// Owns scratch files until FFmpeg has finished reading the mixed soundtrack.
pub struct PreparedAudio {
    directory: PathBuf,
}
impl PreparedAudio {
    pub fn path(&self) -> PathBuf {
        self.directory.join("mix.wav")
    }
}
impl Drop for PreparedAudio {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

/// Reuse the host FFmpeg binary for decoding/resampling; mix in bounded output blocks.
pub fn prepare_audio(
    ffmpeg: &str,
    plan: AudioTimelinePlan,
    root: &Path,
    duration: f64,
) -> Result<PreparedAudio, AudioError> {
    if !duration.is_finite() || !(0.0..=3600.0).contains(&duration) {
        return Err(AudioError::Invalid(
            "native audio export duration must be 0..3600s".into(),
        ));
    }
    let directory = loop {
        let id = NEXT_AUDIO.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("motionloom-audio-{}-{id}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => break path,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    };
    let scratch = PreparedAudio { directory };
    let rate = 48_000u32;
    let mut mixer = AudioMixer::new(plan.clone(), rate)?;
    let mut decoded_bytes = 0u64;
    for asset in &plan.assets {
        let raw = scratch.directory.join("decode.f32");
        let source = if asset.src.contains("://") || Path::new(&asset.src).is_absolute() {
            PathBuf::from(&asset.src)
        } else {
            root.join(&asset.src)
        };
        let result = Command::new(ffmpeg)
            .args(["-y", "-v", "error", "-nostdin", "-i"])
            .arg(source)
            .args(["-vn", "-ac", "2", "-ar", "48000", "-f", "f32le"])
            .arg(&raw)
            .output()?;
        if !result.status.success() {
            return Err(AudioError::Decode {
                asset: asset.id.clone(),
                message: String::from_utf8_lossy(&result.stderr).into_owned(),
            });
        }
        decoded_bytes += fs::metadata(&raw)?.len();
        if decoded_bytes > 256 * 1024 * 1024 {
            return Err(AudioError::Invalid(
                "decoded audio exceeds the 256 MiB session budget; trim source files first".into(),
            ));
        }
        let mut bytes = Vec::new();
        fs::File::open(&raw)?.read_to_end(&mut bytes)?;
        let samples = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        mixer.add_asset(&asset.id, samples)?;
        fs::remove_file(raw)?;
    }
    let frames = (duration * rate as f64).round() as u64;
    let data_len = u32::try_from(frames * 8)
        .map_err(|_| AudioError::Invalid("WAV output exceeds 4 GiB".into()))?;
    let mut out = std::io::BufWriter::new(fs::File::create(scratch.path())?);
    // IEEE float WAV keeps the shared mix unchanged until the final codec encodes it.
    out.write_all(b"RIFF")?;
    out.write_all(&(data_len + 36).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&3u16.to_le_bytes())?;
    out.write_all(&2u16.to_le_bytes())?;
    out.write_all(&rate.to_le_bytes())?;
    out.write_all(&(rate * 8).to_le_bytes())?;
    out.write_all(&8u16.to_le_bytes())?;
    out.write_all(&32u16.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data_len.to_le_bytes())?;
    let mut start = 0;
    while start < frames {
        let count = (frames - start).min(4096) as usize;
        for sample in mixer.render(start, count)? {
            out.write_all(&sample.to_le_bytes())?;
        }
        start += count as u64;
    }
    out.flush()?;
    Ok(scratch)
}
