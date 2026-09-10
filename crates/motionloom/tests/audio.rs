// =========================================
// =========================================
// crates/motionloom/tests/audio.rs

use motionloom::api::{AudioMixer, compile_audio_plan, parse_graph_script};
fn script(audio: &str) -> String {
    format!(
        r##"<Graph fps="30" duration="2s" size={{[32,32]}}>
<Assets><AudioAsset id="tone" src="tone.wav" /></Assets>
{audio}
<Scene id="main"><Timeline><Track id="track"><Sequence from="0s" duration="2s">
<Rect id="card" width="32" height="32" color="#ff0000" />
</Sequence></Track></Timeline></Scene>
<Present from="main" />
</Graph>"##
    )
    .replace("><", ">\n<")
}
fn mixer(audio: &str) -> AudioMixer {
    let graph = parse_graph_script(&script(audio)).unwrap();
    let mut mixer = AudioMixer::new(compile_audio_plan(&graph).unwrap(), 8000).unwrap();
    let pcm = (0..16000)
        .flat_map(|i| {
            let v = i as f32 / 16000.0;
            [v, v]
        })
        .collect();
    mixer.add_asset("tone", pcm).unwrap();
    mixer
}
#[test]
fn old_animation_keys_and_empty_soundtrack_remain_valid() {
    let graph = parse_graph_script(&script(
        r#"<AnimationTarget node="card" property="x">
<Key time="0s" value="0" /><Key frame="30" value="10" />
</AnimationTarget>"#,
    ))
    .unwrap();
    assert!(compile_audio_plan(&graph).unwrap().clips.is_empty());
    assert_eq!(graph.animation_targets.len(), 1);
    let mut json = serde_json::to_value(&graph).unwrap();
    json.as_object_mut().unwrap().remove("audioClips");
    json.as_object_mut().unwrap().remove("audioTargets");
    let old: motionloom::GraphScript = serde_json::from_value(json).unwrap();
    assert!(old.audio_clips.is_empty());
}
#[test]
fn trim_and_timeline_gap_are_sample_accurate() {
    let mix =
        mixer(r#"<AudioClip id="a" asset="tone" from="0.5s" duration="0.5s" sourceIn="1s" />"#);
    assert_eq!(mix.render(3999, 1).unwrap(), vec![0.0, 0.0]);
    assert_eq!(mix.render(4000, 1).unwrap(), vec![0.5, 0.5]);
    assert_eq!(mix.render(8000, 1).unwrap(), vec![0.0, 0.0]);
}
#[test]
fn trimmed_loop_and_balance_work_without_restarting_source() {
    let mix = mixer(
        r#"<AudioClip id="a" asset="tone" from="0s" duration="2s" sourceIn="0.5s" sourceOut="1s" loop="true" pan="1" />"#,
    );
    assert_eq!(mix.render(0, 1).unwrap(), vec![0.0, 0.25]);
    assert_eq!(mix.render(4000, 1).unwrap(), vec![0.0, 0.25]);
}
#[test]
fn variable_rate_seek_matches_sequential_chunks_and_known_integral() {
    let mix = mixer(
        r#"<AudioClip id="a" asset="tone" from="0s" duration="2s" />
<AudioTarget node="a" property="playbackRate"><Key time="0s" value="1" /><Key time="1s" value="2" /></AudioTarget>"#,
    );
    // Integral of rate 1->2 over 0.5 seconds = 0.625 seconds, source ramp = 0.3125.
    assert!((mix.render(4000, 1).unwrap()[0] - 0.3125).abs() < 1e-6);
    let whole = mix.render(0, 8000).unwrap();
    let chunks: Vec<f32> = (0..8)
        .flat_map(|i| mix.render(i * 1000, 1000).unwrap())
        .collect();
    assert_eq!(whole, chunks);
    assert_eq!(mix.render(7123, 200).unwrap(), whole[14246..14646]);
}
#[test]
fn gain_automation_and_overlapping_clips_mix() {
    let mix = mixer(
        r#"<AudioClip id="a" asset="tone" from="0s" duration="2s" />
<AudioClip id="b" asset="tone" from="0s" duration="2s" />
<AudioTarget node="a" property="gainDb"><Key time="0s" value="-20" /><Key time="1s" value="0" /></AudioTarget>"#,
    );
    assert!(
        (mix.render(4000, 1).unwrap()[0] - 0.25 * (1.0 + 10f32.powf(-10.0 / 20.0))).abs() < 1e-6
    );
    assert_eq!(mix.render(12000, 1).unwrap(), vec![1.0, 1.0]);
}
#[test]
fn malformed_audio_is_rejected_before_playback() {
    let clip = r#"<AudioClip id="a" asset="tone" from="0s" duration="1s" />"#;
    for invalid in [
        clip.replace("tone", "missing"),
        clip.replace("0s", "-1s"),
        clip.replace("1s", "0s"),
        clip.replace("duration=\"1s\"", "duration=\"NaN\""),
        format!("{clip}\n{clip}"),
        format!(
            "{clip}<AudioTarget node=\"a\" property=\"gainDb\"><Key time=\"0s\" value=\"0\"/><Key frame=\"0\" value=\"1\"/></AudioTarget>"
        ),
        format!(
            "{clip}<AudioTarget node=\"missing\" property=\"gainDb\"><Key time=\"0s\" value=\"0\"/></AudioTarget>"
        ),
        format!(
            "{clip}<AudioTarget node=\"a\" property=\"playbackRate\"><Key time=\"0s\" value=\"0\"/></AudioTarget>"
        ),
    ] {
        assert!(
            parse_graph_script(&script(&invalid)).is_err(),
            "accepted {invalid}"
        );
    }
}
#[test]
fn audio_does_not_create_unknown_tag_or_attribute_diagnostics() {
    let source = script(
        r#"<AudioClip id="a" asset="tone" from="0s" duration="1s" />
<AudioTarget node="a" property="pan"><Key time="0s" value="0" /></AudioTarget>"#,
    );
    let report = motionloom::api::analyze_motionloom_script(&source);
    let json = serde_json::to_string(&report).unwrap();
    assert!(!json.contains("UNKNOWN_TAG"), "{json}");
    assert!(!json.contains("UNKNOWN_ATTRIBUTE"), "{json}");
}

// Exercise actual FFmpeg decoding, resampling, muxing and soundtrack duration without a GPU.
#[test]
#[cfg(not(target_arch = "wasm32"))]
#[ignore = "requires ffmpeg and ffprobe on PATH"]
fn ffmpeg_export_contains_the_mixed_audio_track() {
    use motionloom::{FfmpegVideoEncoder, VideoEncoder};
    use std::process::Command;
    let root = std::env::temp_dir().join(format!("motionloom-av-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let tone = root.join("tone.wav");
    assert!(
        Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2:sample_rate=22050"
            ])
            .arg(&tone)
            .status()
            .unwrap()
            .success()
    );
    let graph = parse_graph_script(&script(
        r#"<AudioClip id="a" asset="tone" from="0.5s" duration="1s" sourceIn="0.2s" />"#,
    ))
    .unwrap();
    let audio =
        motionloom::api::prepare_audio("ffmpeg", compile_audio_plan(&graph).unwrap(), &root, 2.0)
            .unwrap();
    let out = root.join("av.mp4");
    let mut encoder = FfmpegVideoEncoder::new("ffmpeg", &out)
        .with_encoder_args(vec![
            "-c:v".into(),
            "mpeg4".into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
        ])
        .with_audio_path(audio.path());
    encoder.begin(32, 32, 30.0).unwrap();
    for i in 0..60 {
        encoder.push_frame(i, &vec![255; 32 * 32 * 4]).unwrap();
    }
    encoder.finish().unwrap();
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(&out)
        .output()
        .unwrap();
    let info: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    let streams = info["streams"].as_array().unwrap();
    assert_eq!(streams.len(), 2);
    let track = streams.iter().find(|s| s["codec_type"] == "audio").unwrap();
    assert_eq!(track["sample_rate"], "48000");
    assert!((track["duration"].as_str().unwrap().parse::<f64>().unwrap() - 2.0).abs() < 0.05);
    // Verify the rendered WAV has leading silence and audible samples inside the clip.
    let wav = std::fs::read(audio.path()).unwrap();
    assert!(wav[44..44 + 4000 * 8].iter().all(|b| *b == 0));
    assert!(wav[44 + 30000 * 8..44 + 31000 * 8].iter().any(|b| *b != 0));
    // Check the actual document API, including relative AudioAsset resolution and CPU scene export.
    let scene_out = root.join("scene.mov");
    pollster::block_on(
        motionloom::api::render_motionloom_document_to_video_with_progress(
            "ffmpeg",
            &script(r#"<AudioClip id="a" asset="tone" from="0.5s" duration="1s" />"#),
            &root,
            &scene_out,
            motionloom::api::SceneRenderProfile::Cpu,
            30,
            |_| {},
        ),
    )
    .unwrap();
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(&scene_out)
        .output()
        .unwrap();
    let info: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
    assert!(
        info["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["codec_type"] == "audio")
    );
    std::fs::remove_dir_all(root).unwrap();
}

// The checked-in authoring example must parse without any editor normalization.
#[test]
fn audio_example_is_valid() {
    let script = include_str!("../../../examples/motionloom/scene/audio/audio_edit.motionloom");
    let graph = parse_graph_script(script).unwrap();
    assert_eq!(compile_audio_plan(&graph).unwrap().clips.len(), 2);
}
