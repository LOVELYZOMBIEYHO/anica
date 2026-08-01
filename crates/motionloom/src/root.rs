use std::path::Path;
use std::sync::{Arc, atomic::AtomicBool};

use crate::dsl::{
    GraphScript, graph_root_start, parse_graph_script, validate_graph_present_placement,
};
use crate::error::GraphParseError;
use crate::error::MotionLoomError;
use crate::process::dsl::{is_process_graph_script, parse_process_graph_script};
use crate::scene::render::{
    SceneRenderProfile, SceneRenderProgress,
    render_scene_graph_to_png_sequence_with_progress_and_cancel,
    render_scene_graph_to_video_with_progress_and_cancel,
};
use crate::world::{WorldGraph, is_world_graph_script, parse_world_graph_script};
use crate::world::{
    WorldRenderProgress, render_world_graph_to_png_sequence_with_progress_and_cancel,
    render_world_graph_to_video_with_progress_and_cancel,
};

#[derive(Debug, Clone)]
pub enum MotionLoomDocument {
    Process(GraphScript),
    Scene(GraphScript),
    World(WorldGraph),
    Mixed(RootGraphShell),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RootGraphDomain {
    Process,
    Scene,
    World,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RootGraphShell {
    pub has_process: bool,
    pub has_scene: bool,
    pub has_world: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum MotionLoomRenderProgress {
    Scene(SceneRenderProgress),
    World(WorldRenderProgress),
}

impl MotionLoomRenderProgress {
    pub fn rendered_frames(self) -> u32 {
        match self {
            Self::Scene(progress) => progress.rendered_frames,
            Self::World(progress) => progress.rendered_frames,
        }
    }

    pub fn total_frames(self) -> u32 {
        match self {
            Self::Scene(progress) => progress.total_frames,
            Self::World(progress) => progress.total_frames,
        }
    }
}

impl RootGraphShell {
    pub fn domains(&self) -> Vec<RootGraphDomain> {
        let mut out = Vec::new();
        if self.has_process {
            out.push(RootGraphDomain::Process);
        }
        if self.has_scene {
            out.push(RootGraphDomain::Scene);
        }
        if self.has_world {
            out.push(RootGraphDomain::World);
        }
        out
    }

    pub fn is_mixed(&self) -> bool {
        self.domains().len() > 1
    }
}

pub fn inspect_root_graph(script: &str) -> Result<RootGraphShell, GraphParseError> {
    validate_graph_present_placement(script)?;
    let graph_start = graph_root_start(script)?;
    let graph_body = &script[graph_start..];
    if !graph_body.contains("</Graph>") {
        return Err(GraphParseError {
            line: 1,
            message: "Missing </Graph> close tag.".to_string(),
        });
    }
    if contains_open_tag(graph_body, "World") {
        return Err(GraphParseError {
            line: 1,
            message: "<World> has been removed from MotionLoom DSL. Put renderable 3D content inside <Scene> using <CompositeGroup space=\"3d\">; space=\"world\" remains available as a coordinate-space value.".to_string(),
        });
    }

    Ok(RootGraphShell {
        has_process: graph_body.contains("<Process"),
        has_scene: graph_body.contains("<Scene"),
        has_world: false,
    })
}

pub fn parse_motionloom_document(script: &str) -> Result<MotionLoomDocument, GraphParseError> {
    let shell = inspect_root_graph(script)?;
    if !shell.has_process && contains_legacy_root_process_nodes(script) {
        return Err(GraphParseError {
            line: 1,
            message:
                "Root-level process nodes are no longer supported. Wrap <Input>/<Tex>/<Pass>/<Output> nodes in <Process id=\"...\">...</Process>."
                    .to_string(),
        });
    }
    if shell.is_mixed() {
        return Ok(MotionLoomDocument::Mixed(shell));
    }
    if shell.has_world || is_world_graph_script(script) {
        return parse_world_graph_script(script).map(MotionLoomDocument::World);
    }
    if shell.has_process || is_process_graph_script(script) {
        return parse_process_graph_script(script).map(MotionLoomDocument::Process);
    }
    parse_graph_script(script).map(MotionLoomDocument::Scene)
}

pub async fn render_motionloom_document_to_video_with_progress<F>(
    ffmpeg_bin: &str,
    script: &str,
    asset_root: impl AsRef<Path>,
    output_path: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    progress_callback: F,
) -> Result<(), MotionLoomError>
where
    F: FnMut(MotionLoomRenderProgress),
{
    render_motionloom_document_to_video_with_progress_and_cancel(
        ffmpeg_bin,
        script,
        asset_root,
        output_path,
        profile,
        progress_every_frames,
        None,
        progress_callback,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn render_motionloom_document_to_video_with_progress_and_cancel<F>(
    ffmpeg_bin: &str,
    script: &str,
    asset_root: impl AsRef<Path>,
    output_path: &Path,
    profile: SceneRenderProfile,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    mut progress_callback: F,
) -> Result<(), MotionLoomError>
where
    F: FnMut(MotionLoomRenderProgress),
{
    let shell = inspect_root_graph(script)?;

    if shell.has_scene || (shell.has_world && shell.has_process) {
        let graph = parse_graph_script(script)?;
        return render_scene_graph_to_video_with_progress_and_cancel(
            ffmpeg_bin,
            &graph,
            output_path,
            profile,
            progress_every_frames,
            cancel,
            |progress| progress_callback(MotionLoomRenderProgress::Scene(progress)),
        )
        .await
        .map_err(MotionLoomError::from);
    }

    if shell.has_process {
        let graph = parse_process_graph_script(script)?;
        if process_graph_needs_external_input(&graph) {
            return Err(MotionLoomError::UnsupportedDocument {
                message:
                    "Process-only graphs with input:clip0 need a source clip. Use the timeline Layer FX export path, or wrap the effect around a <Scene> source for MotionLoom Page render."
                        .to_string(),
            });
        }
        return render_scene_graph_to_video_with_progress_and_cancel(
            ffmpeg_bin,
            &graph,
            output_path,
            profile,
            progress_every_frames,
            cancel,
            |progress| progress_callback(MotionLoomRenderProgress::Scene(progress)),
        )
        .await
        .map_err(MotionLoomError::from);
    }

    if shell.has_world || is_world_graph_script(script) {
        let graph = parse_world_graph_script(script)?;
        return render_world_graph_to_video_with_progress_and_cancel(
            ffmpeg_bin,
            &graph,
            asset_root,
            output_path,
            profile,
            progress_every_frames,
            cancel,
            |progress| progress_callback(MotionLoomRenderProgress::World(progress)),
        )
        .await
        .map_err(MotionLoomError::from);
    }

    let graph = parse_graph_script(script)?;
    render_scene_graph_to_video_with_progress_and_cancel(
        ffmpeg_bin,
        &graph,
        output_path,
        profile,
        progress_every_frames,
        cancel,
        |progress| progress_callback(MotionLoomRenderProgress::Scene(progress)),
    )
    .await
    .map_err(MotionLoomError::from)
}

pub async fn render_motionloom_document_to_png_sequence_with_progress<F>(
    script: &str,
    asset_root: impl AsRef<Path>,
    output_dir: &Path,
    progress_every_frames: u32,
    progress_callback: F,
) -> Result<(), MotionLoomError>
where
    F: FnMut(MotionLoomRenderProgress),
{
    render_motionloom_document_to_png_sequence_with_progress_and_cancel(
        script,
        asset_root,
        output_dir,
        progress_every_frames,
        None,
        progress_callback,
    )
    .await
}

pub async fn render_motionloom_document_to_png_sequence_with_progress_and_cancel<F>(
    script: &str,
    asset_root: impl AsRef<Path>,
    output_dir: &Path,
    progress_every_frames: u32,
    cancel: Option<Arc<AtomicBool>>,
    mut progress_callback: F,
) -> Result<(), MotionLoomError>
where
    F: FnMut(MotionLoomRenderProgress),
{
    let shell = inspect_root_graph(script)?;

    if shell.has_scene || (shell.has_world && shell.has_process) {
        let graph = parse_graph_script(script)?;
        return render_scene_graph_to_png_sequence_with_progress_and_cancel(
            &graph,
            output_dir,
            progress_every_frames,
            cancel,
            |progress| progress_callback(MotionLoomRenderProgress::Scene(progress)),
        )
        .await
        .map_err(MotionLoomError::from);
    }

    if shell.has_process {
        let graph = parse_process_graph_script(script)?;
        if process_graph_needs_external_input(&graph) {
            return Err(MotionLoomError::UnsupportedDocument {
                message:
                    "Process-only graphs with input:clip0 need a source clip. Use the timeline Layer FX export path, or wrap the effect around a <Scene> source for MotionLoom Page render."
                        .to_string(),
            });
        }
        return render_scene_graph_to_png_sequence_with_progress_and_cancel(
            &graph,
            output_dir,
            progress_every_frames,
            cancel,
            |progress| progress_callback(MotionLoomRenderProgress::Scene(progress)),
        )
        .await
        .map_err(MotionLoomError::from);
    }

    if shell.has_world || is_world_graph_script(script) {
        let graph = parse_world_graph_script(script)?;
        return render_world_graph_to_png_sequence_with_progress_and_cancel(
            &graph,
            asset_root,
            output_dir,
            progress_every_frames,
            cancel,
            |progress| progress_callback(MotionLoomRenderProgress::World(progress)),
        )
        .await
        .map_err(MotionLoomError::from);
    }

    let graph = parse_graph_script(script)?;
    render_scene_graph_to_png_sequence_with_progress_and_cancel(
        &graph,
        output_dir,
        progress_every_frames,
        cancel,
        |progress| progress_callback(MotionLoomRenderProgress::Scene(progress)),
    )
    .await
    .map_err(MotionLoomError::from)
}

fn process_graph_needs_external_input(graph: &GraphScript) -> bool {
    graph.inputs.iter().any(|input| {
        input
            .from
            .as_deref()
            .is_some_and(|from| from.trim().starts_with("input:"))
    })
}

fn contains_legacy_root_process_nodes(script: &str) -> bool {
    ["Input", "Clip", "Tex", "Buffer", "Pass", "Output"]
        .iter()
        .any(|tag| contains_open_tag(script, tag))
}

fn contains_open_tag(script: &str, tag: &str) -> bool {
    let needle = format!("<{tag}");
    let mut offset = 0usize;
    while let Some(relative_pos) = script[offset..].find(&needle) {
        let start = offset + relative_pos;
        let after_tag = start + needle.len();
        let Some(next) = script[after_tag..].chars().next() else {
            return true;
        };
        if next == '>' || next == '/' || next.is_ascii_whitespace() {
            return true;
        }
        offset = after_tag;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{MotionLoomDocument, parse_motionloom_document};

    #[test]
    fn root_dispatcher_parses_single_process_block_as_process() {
        let script = r#"
<Graph fps={30} duration="1s" size={[1920,1080]}>
  <Process id="final_grade">
    <Tex id="src" fmt="rgba16f" size={[1920,1080]} />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx" kind="compute" effect="gaussian_5tap_blur"
          in={["src"]} out={["out"]} params={{ sigma: "10" }} />
  </Process>
  <Present from="final_grade" />
</Graph>
"#;
        let doc = parse_motionloom_document(script).expect("document should parse");
        let MotionLoomDocument::Process(graph) = doc else {
            panic!("expected process document");
        };
        assert_eq!(graph.id.as_deref(), Some("final_grade"));
        assert_eq!(graph.passes.len(), 1);
        assert_eq!(graph.present.from, "out");
    }

    #[test]
    fn root_dispatcher_keeps_scene_process_graph_as_mixed() {
        let script = r##"
<Graph fps={30} duration="1s" size={[1920,1080]}>
  <Scene id="title_scene">
    <Timeline>
      <Track id="main" space="screen" z="0">
        <Sequence from="0s" duration="1s">
          <Layer>
            <Rect x="0" y="0" width="1920" height="1080" color="#000000" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Process id="final_grade">
    <Tex id="src" fmt="rgba16f" from="scene:title_scene" />
    <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
    <Pass id="fx" kind="compute" effect="gaussian_5tap_blur"
          in={["src"]} out={["out"]} params={{ sigma: "10" }} />
  </Process>
  <Present from="final_grade" />
</Graph>
"##;
        let doc = parse_motionloom_document(script).expect("document should inspect");
        let MotionLoomDocument::Mixed(shell) = doc else {
            panic!("expected mixed document");
        };
        assert!(shell.has_scene);
        assert!(shell.has_process);
    }

    #[test]
    fn root_dispatcher_rejects_legacy_root_process_nodes() {
        let script = r#"
<Graph fps={30} size={[1920,1080]}>
  <Tex id="src" fmt="rgba16f" from="input:clip0" />
  <Tex id="out" fmt="rgba16f" size={[1920,1080]} />
  <Pass id="fx" kind="compute" effect="gaussian_5tap_blur"
        in={["src"]} out={["out"]} params={{ sigma: "10" }} />
  <Present from="out" />
</Graph>
"#;
        let err = parse_motionloom_document(script).expect_err("legacy shorthand should fail");
        assert!(err.message.contains("Root-level process nodes"));
    }

    #[test]
    fn root_dispatcher_rejects_removed_world_tag_but_keeps_world_space_name() {
        let removed_world = r#"
<Graph fps={30} duration="1s" size={[1920,1080]}>
  <World id="legacy_world" />
  <Present from="legacy_world" />
</Graph>
"#;
        let err =
            parse_motionloom_document(removed_world).expect_err("World tag should be removed");
        assert!(err.message.contains("<World> has been removed"));

        let world_space = r##"
<Graph fps={30} duration="1s" size={[256,256]}>
  <Scene id="scene0">
    <Timeline>
      <Track id="main" space="world">
        <Sequence duration="1s">
          <Layer>
            <Rect x="0" y="0" width="256" height="256" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="scene0" />
</Graph>
"##;
        let doc = parse_motionloom_document(world_space)
            .expect("world remains a valid Scene coordinate-space name");
        assert!(matches!(doc, MotionLoomDocument::Scene(_)));
    }

    #[test]
    fn root_dispatcher_does_not_treat_text_as_legacy_tex() {
        let script = r##"
<Graph fps={30} duration="1s" size={[1080,768]}>
  <Background color="#ffffff" />
  <Scene id="text_scene">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="1s">
          <Layer>
            <Text x="48" y="72" value="#3F5877" size="38" color="#3F5877" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="text_scene" />
</Graph>
"##;
        let doc = parse_motionloom_document(script).expect("scene with <Text> should parse");
        let MotionLoomDocument::Scene(graph) = doc else {
            panic!("expected scene document");
        };
        assert!(graph.has_scene_nodes());
    }

    #[test]
    fn root_dispatcher_accepts_leading_xml_comment() {
        let script = r##"
<!-- Dataset note: this comment is metadata and not a scene node. -->
<Graph fps={30} duration="1s" size={[256,256]}>
  <Background color="#ffffff" />
  <Scene id="commented_scene">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="1s" out="hold">
          <Layer>
            <Rect x="0" y="0" width="256" height="256" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>
  <Present from="commented_scene" />
</Graph>
"##;
        let doc = parse_motionloom_document(script).expect("leading comment should parse");
        let MotionLoomDocument::Scene(graph) = doc else {
            panic!("expected scene document");
        };
        assert_eq!(graph.scenes[0].id, "commented_scene");
    }
}
