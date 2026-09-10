# Audio editing (Native and WASM)

MotionLoom audio is additive. Existing scripts require **no migration**.
`AudioAsset` names a source; `AudioClip` places a trimmed region on the graph
clock; `AudioTarget` animates a numeric property of that clip. Keep these nodes
as direct children of `Graph`, before the final `Present`.

```xml
<Assets>
  <AudioAsset id="score" src="score.wav" />
</Assets>
<AudioClip id="intro" asset="score" from="0s" duration="10s" sourceIn="42s" />
<AudioClip id="return" asset="score" from="12s" duration="8s" sourceIn="82s" />
<AudioTarget node="intro" property="gainDb">
  <Key time="0s" value="-60" />
  <Key time="1s" value="-12" />
  <Key time="9.8s" value="-12" />
  <Key time="10s" value="-60" />
</AudioTarget>
```

Before: a visual-only graph exports without sound. After: add the nodes above
to that graph; its visuals and existing AnimationTarget keys retain their behavior.
The full runnable example is `examples/motionloom/scene/audio/audio_edit.motionloom`
in the Anica repository. Its original diagnostic WAV is generated mathematically,
contains no external media, and is intended for testing rather than soundtrack use.

## Timing and editing semantics

- `id`, `asset`, `from`, `duration` are required on AudioClip. Duration is positive.
- `from`, Key `time`, and Key `frame` are **global graph times**, not clip-local times.
- Clip intervals include their start and exclude their end.
- `sourceIn` defaults to 0. Optional `sourceOut` bounds the source region and must
  be greater than sourceIn. Source bounds are validated against decoded audio.
- `loop` defaults to false. If true, repeat the trimmed source region. If false,
  exhausting the source produces silence until the clip ends.
- `duration` is output timeline duration; changing playbackRate never moves a clip.
- Overlapping clips mix. To cut, use multiple clips. To crossfade, overlap clips
  and animate their gainDb independently.
- Clip defaults: `gainDb=0`, `pan=0`, `playbackRate=1`.
- AudioTarget properties: gainDb [-120,24], pan [-1,1], playbackRate [0.125,8].
  Pan is stereo balance: center preserves both channels, extremes mute one channel.
  Playback rate changes both speed and pitch; pitch-preserving stretching is not implemented.
- Key retains exactly `time` **or** `frame`, `value`, optional `ease` (linear by
  default). Audio supports linear, step, ease_in, ease_out, ease_in_out. Destination
  key easing controls the preceding segment. Step holds until the destination time.
- First/last key values extend outside the keyed interval while the clip is active.
  Targets override the clip's corresponding default. Duplicate channel targets,
  duplicate key times and missing clip/asset references are errors.
- Audio time uses f64 seconds and absolute integer output sample positions; time
  keys are not snapped to video frames. Source rate curves are integrated, so
  direct seeks give the same samples as sequential rendering.
- Mono sources use equal-power stereo upmix (-3 dB per channel) in both adapters.
  Multichannel downmix follows the host decoder; use stereo sources for matched channel balances.
- Output is 48 kHz stereo in the supplied adapters. The mixer sums tracks and
  hard-clips values outside [-1,1]; it does not automatically normalize loudness.
  Leave headroom when mixing. Fades do not silently change authored boundaries.

## Host integration

Use `motionloom::api::{compile_audio_plan, AudioMixer}`. Supply decoded interleaved
stereo f32 PCM at the mixer's sample rate via `add_asset`, then request blocks
using `render(start_sample, frames)`. Blocks are limited to ten seconds.

Native scene/root video export prepares a float WAV using the shared mixer and
passes it to the existing FFmpeg process alongside video. MP4/MOV use AAC; WebM
uses Opus. The root document API resolves relative audio paths against asset_root;
direct graph export uses configured scene asset roots. Scratch files are owned
by PreparedAudio and removed after export/error. PNG sequences remain image-only.
`prepare_audio` and `FfmpegVideoEncoder::with_audio_path` also support custom hosts.
The existing native preview application has not been given audio device playback;
this change's preview integration is the browser MotionLoom page.

WASM exposes `WasmAudioMixer`: constructor(script, sample_rate), plan_json(),
add_asset(id, Float32Array), render(start_sample, frames), free(). The browser
adapter uses Web Audio to decode/downmix/resample and schedule mixed blocks.
Its audio clock drives the visual timeline; pause/seek/recompile discard queued
sources. Browser autoplay rules may require pressing Play before sound starts.

Browser export uses the existing Mediabunny/WebCodecs integration. It feeds PCM
blocks to AudioBufferSource and muxes with video into MP4 (AAC) or WebM (Opus).
Codec support is checked before encoding; unsupported audio is an explicit error,
never a silent video export. Assets must be browser-readable URLs with applicable
CORS access. The browser adapter accepts a source document base URL.

No new codec/package dependencies are introduced. Both adapters currently retain
decoded source PCM, with a 256 MiB decoded session budget; mixed output is chunked.
Native WAV export is capped at one hour. Long-source streaming, native live audio
preview, pitch-preserving stretching and audio/beat detection are future work.

## Validation

```sh
cargo test -p motionloom --test audio
cargo test -p motionloom --test audio -- --ignored
```

The ignored test requires FFmpeg/ffprobe and checks actual audio/video muxing,
resampling, soundtrack length and leading silence. Pure tests cover old scripts,
trim, looping, automation, overlap, variable-speed seeking and diagnostics.
