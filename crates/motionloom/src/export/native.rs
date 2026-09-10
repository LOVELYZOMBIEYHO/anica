// =========================================
// =========================================
// crates/motionloom/src/export/native.rs

use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

use crate::export::{EncodeError, VideoEncoder};

/// Native FFmpeg subprocess video encoder.
pub struct FfmpegVideoEncoder {
    ffmpeg_bin: String,
    output_path: std::path::PathBuf,
    encoder_args: Vec<String>,
    audio_path: Option<std::path::PathBuf>,
    size_arg: Option<String>,
    fps_arg: Option<String>,
    child: Option<Child>,
    stdin: Option<std::process::ChildStdin>,
}

impl FfmpegVideoEncoder {
    pub fn new(ffmpeg_bin: &str, output_path: &Path) -> Self {
        Self {
            ffmpeg_bin: ffmpeg_bin.to_string(),
            output_path: output_path.to_path_buf(),
            encoder_args: default_encoder_args(),
            audio_path: None,
            size_arg: None,
            fps_arg: None,
            child: None,
            stdin: None,
        }
    }

    /// Replace the default H.264 encoder arguments with a custom set (e.g.
    /// ProRes or platform-specific H.264).
    pub fn with_encoder_args(mut self, args: Vec<String>) -> Self {
        self.encoder_args = args;
        self
    }

    /// Add a prepared soundtrack without changing the video-only encoder trait.
    pub fn with_audio_path(mut self, path: impl AsRef<Path>) -> Self {
        self.audio_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Stop the encoder immediately without finalizing the output file.
    pub fn abort(&mut self) {
        if let Some(stdin) = self.stdin.take() {
            drop(stdin);
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(target_os = "macos")]
fn default_encoder_args() -> Vec<String> {
    vec![
        "-c:v".to_string(),
        "h264_videotoolbox".to_string(),
        "-allow_sw".to_string(),
        "1".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "30M".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
    ]
}

#[cfg(target_os = "windows")]
fn default_encoder_args() -> Vec<String> {
    vec![
        "-c:v".to_string(),
        "h264_mf".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "30M".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
    ]
}

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
fn default_encoder_args() -> Vec<String> {
    vec![
        "-c:v".to_string(),
        "libopenh264".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-b:v".to_string(),
        "30M".to_string(),
        "-movflags".to_string(),
        "+faststart".to_string(),
    ]
}

impl VideoEncoder for FfmpegVideoEncoder {
    fn begin(&mut self, width: u32, height: u32, fps: f32) -> Result<(), EncodeError> {
        if let Some(parent) = self.output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| EncodeError::CreateOutputDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        self.size_arg = Some(format!("{}x{}", width.max(1), height.max(1)));
        self.fps_arg = Some(format!("{fps:.6}"));
        let output_arg = self.output_path.to_string_lossy().to_string();

        let mut command = Command::new(&self.ffmpeg_bin);
        command.args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            self.size_arg.as_ref().expect("size_arg set above"),
            "-r",
            self.fps_arg.as_ref().expect("fps_arg set above"),
            "-i",
            "pipe:0",
        ]);
        // Encode the optional soundtrack in the existing FFmpeg invocation.
        if let Some(audio) = &self.audio_path {
            command
                .arg("-i")
                .arg(audio)
                .args(["-map", "0:v:0", "-map", "1:a:0", "-c:a"]);
            let codec = match self.output_path.extension().and_then(|x| x.to_str()) {
                Some("webm") => "libopus",
                _ => "aac",
            };
            command.args([codec, "-b:a", "192k"]);
        } else {
            command.arg("-an");
        }
        let mut child = command
            .args(&self.encoder_args)
            .arg(output_arg.as_str())
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| EncodeError::StartEncoder(err.to_string()))?;

        let stdin = child.stdin.take().ok_or(EncodeError::MissingEncoderInput)?;
        self.child = Some(child);
        self.stdin = Some(stdin);
        Ok(())
    }

    fn push_frame(&mut self, _frame_index: u32, rgba: &[u8]) -> Result<(), EncodeError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or(EncodeError::MissingEncoderInput)?;
        if let Err(source) = stdin.write_all(rgba) {
            let stderr = if let Some(stdin) = self.stdin.take() {
                drop(stdin);
                if let Some(child) = self.child.take() {
                    child
                        .wait_with_output()
                        .ok()
                        .map(|output| String::from_utf8_lossy(&output.stderr).trim().to_string())
                        .filter(|message| !message.is_empty())
                } else {
                    None
                }
            } else {
                None
            };
            let message = if let Some(stderr) = stderr {
                format!("{source}; encoder stderr: {stderr}")
            } else {
                source.to_string()
            };
            return Err(EncodeError::WriteFrame(std::io::Error::new(
                source.kind(),
                message,
            )));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        if let Some(stdin) = self.stdin.take() {
            drop(stdin);
        }
        let child = self.child.take().ok_or(EncodeError::NotStarted)?;
        let output = child
            .wait_with_output()
            .map_err(|err| EncodeError::EncoderFailed(err.to_string()))?;
        if !output.status.success() {
            return Err(EncodeError::EncoderFailed(
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        Ok(())
    }
}

impl Drop for FfmpegVideoEncoder {
    fn drop(&mut self) {
        if self.stdin.is_some() {
            let _ = self.finish();
        }
    }
}
