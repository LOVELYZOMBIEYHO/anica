// =========================================
// =========================================
// crates/motionloom/src/world/dsl.rs

use std::collections::HashSet;

use crate::dsl::{graph_root_start, validate_graph_present_placement};
pub use crate::error::GraphParseError;
use crate::world::model::{
    WorldAction, WorldActionBone, WorldActionPose, WorldActor, WorldApplyAction, WorldBackground,
    WorldBackgroundFit, WorldBoneAxis, WorldBoneAxisMap, WorldCamera, WorldCameraControl,
    WorldCameraProjection, WorldDirectionFrame, WorldDirectionalCharacter, WorldGraph,
    WorldMaterial, WorldMaterialStyle, WorldModelProfile, WorldNode, WorldPathStyle, WorldPlay,
    WorldPresent, WorldProfileRetarget, WorldRetarget, WorldRetargetMap, WorldSpritePlayback,
};

const DEFAULT_WORLD_DURATION_MS: u64 = 1000;

pub fn is_world_graph_script(script: &str) -> bool {
    graph_root_start(script).is_ok() && script.contains("<World")
}

pub fn parse_world_graph_script(script: &str) -> Result<WorldGraph, GraphParseError> {
    validate_graph_present_placement(script)?;
    let script = strip_comments(script);
    let graph_start = graph_root_start(&script)?;
    let script = &script[graph_start..];
    let line = 1;
    let graph_open_end = script.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Missing <Graph ...> node.".to_string(),
    })?;
    let graph_open = &script[..=graph_open_end];
    if !graph_open.trim_start().starts_with("<Graph") {
        return Err(GraphParseError {
            line,
            message: "World DSL must start with <Graph ...>.".to_string(),
        });
    }
    let graph_close_start = script.rfind("</Graph>").ok_or_else(|| GraphParseError {
        line,
        message: "Missing </Graph> close node.".to_string(),
    })?;
    let graph_inner = &script[graph_open_end + 1..graph_close_start];
    if attr_value(graph_open, "scope").is_some() {
        return Err(GraphParseError {
            line,
            message: "Graph scope has been removed. Use unified <Graph ...> syntax.".to_string(),
        });
    }

    let fps = attr_value(graph_open, "fps")
        .map(|raw| parse_fps(&raw, line, "fps"))
        .transpose()?
        .unwrap_or(60.0);
    let duration_raw = attr_value(graph_open, "duration");
    let duration_explicit = duration_raw.is_some();
    let duration_ms = duration_raw
        .as_deref()
        .map(|raw| parse_duration_ms(raw, line, "duration"))
        .transpose()?
        .unwrap_or(DEFAULT_WORLD_DURATION_MS);
    let size = parse_size(
        &required_attr_value(graph_open, "size", line)?,
        line,
        "size",
    )?;
    let render_size = attr_value(graph_open, "renderSize")
        .map(|raw| parse_size(&raw, line, "renderSize"))
        .transpose()?;

    let mut model_profiles = Vec::new();
    for block in collect_tag_blocks(graph_inner, "ModelProfile")? {
        model_profiles.push(parse_model_profile_block(&block)?);
    }
    let graph_body = strip_tag_blocks(graph_inner, "ModelProfile")?;
    let graph_inner = graph_body.as_str();

    let default_background = collect_self_closing_blocks(graph_inner, "Background")?
        .first()
        .map(|node| parse_background_node(node))
        .transpose()?;
    let default_camera = collect_self_closing_blocks(graph_inner, "Camera")?
        .first()
        .map(|node| parse_camera_node(node))
        .transpose()?;

    let mut worlds = Vec::new();
    for block in collect_tag_blocks(graph_inner, "World")? {
        worlds.push(parse_world_block(
            &block,
            default_background.clone(),
            default_camera.clone(),
        )?);
    }
    if worlds.is_empty() {
        return Err(GraphParseError {
            line,
            message: "World graph requires at least one <World> block.".to_string(),
        });
    }
    let mut retargets = Vec::new();
    for block in collect_tag_blocks(graph_inner, "Retarget")? {
        retargets.push(parse_retarget_block(&block)?);
    }

    let mut actions = Vec::new();
    for block in collect_tag_blocks(graph_inner, "Action")? {
        actions.push(parse_action_block(&block)?);
    }

    let mut apply_actions = Vec::new();
    for block in collect_self_closing_blocks(graph_inner, "ApplyAction")? {
        apply_actions.push(parse_apply_action_node(&block)?);
    }

    let present_from = collect_self_closing_blocks(graph_inner, "Present")?
        .first()
        .map(|present_block| required_attr_value(present_block, "from", line))
        .transpose()?
        .unwrap_or_else(|| worlds[0].id.clone());
    let present_from = if worlds.iter().any(|world| world.id == present_from) {
        present_from
    } else if worlds.len() == 1 {
        worlds[0].id.clone()
    } else {
        return Err(GraphParseError {
            line,
            message: format!("Present references missing world '{}'.", present_from),
        });
    };
    let present = WorldPresent { from: present_from };
    validate_world_graph_refs(
        &worlds,
        &model_profiles,
        &retargets,
        &actions,
        &apply_actions,
    )?;

    Ok(WorldGraph {
        id: attr_value(graph_open, "id"),
        version: attr_value(graph_open, "version"),
        fps,
        duration_ms,
        duration_explicit,
        size,
        render_size,
        model_profiles,
        worlds,
        retargets,
        actions,
        apply_actions,
        animation_assets: Vec::new(),
        constraints: Vec::new(),
        lighting: crate::world::WorldLighting::default(),
        present,
    })
}

fn parse_world_block(
    block: &str,
    default_background: Option<WorldBackground>,
    default_camera: Option<WorldCamera>,
) -> Result<WorldNode, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <World> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</World>".len())];
    let id = required_attr_value(open, "id", line)?;

    let background = collect_self_closing_blocks(inner, "Background")?
        .first()
        .map(|node| parse_background_node(node))
        .transpose()?
        .or(default_background);
    let camera = collect_self_closing_blocks(inner, "Camera")?
        .first()
        .map(|node| parse_camera_node(node))
        .transpose()?
        .or(default_camera)
        .unwrap_or_default();

    let mut actors = Vec::new();
    for actor in collect_self_closing_blocks(inner, "Actor")? {
        actors.push(parse_actor_node(&actor, "")?);
    }
    for actor in collect_tag_blocks(inner, "Actor")? {
        let actor_open_end = actor.find('>').ok_or_else(|| GraphParseError {
            line,
            message: "Malformed <Actor> block.".to_string(),
        })?;
        let actor_open = &actor[..=actor_open_end];
        let actor_inner = &actor[actor_open_end + 1..actor.len().saturating_sub("</Actor>".len())];
        actors.push(parse_actor_node(actor_open, actor_inner)?);
    }
    let directional_characters = parse_directional_characters(inner)?;

    Ok(WorldNode {
        id,
        background,
        camera,
        actors,
        directional_characters,
    })
}

fn parse_directional_characters(
    inner: &str,
) -> Result<Vec<WorldDirectionalCharacter>, GraphParseError> {
    let mut directional_characters = Vec::new();
    for block in collect_tag_blocks(inner, "DirectionalCharacter")? {
        directional_characters.push(parse_directional_character_block(&block)?);
    }
    Ok(directional_characters)
}

fn parse_directional_character_block(
    block: &str,
) -> Result<WorldDirectionalCharacter, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <DirectionalCharacter> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</DirectionalCharacter>".len())];
    let direction_map = collect_tag_blocks(inner, "DirectionMap")?
        .first()
        .cloned()
        .ok_or_else(|| GraphParseError {
            line,
            message: "DirectionalCharacter requires a <DirectionMap> block.".to_string(),
        })?;
    let direction_open_end = direction_map.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <DirectionMap> block.".to_string(),
    })?;
    let direction_inner = &direction_map
        [direction_open_end + 1..direction_map.len().saturating_sub("</DirectionMap>".len())];
    let mut directions = Vec::new();
    for direction in collect_self_closing_blocks(direction_inner, "Direction")? {
        directions.push(parse_direction_node(&direction)?);
    }
    if directions.is_empty() {
        return Err(GraphParseError {
            line,
            message: "DirectionMap requires at least one <Direction ... /> node.".to_string(),
        });
    }
    let sheet = attr_value(open, "sheet");
    if sheet.is_none()
        && directions
            .iter()
            .any(|direction| direction.image.as_deref().is_none_or(str::is_empty))
    {
        return Err(GraphParseError {
            line,
            message:
                "DirectionalCharacter without sheet requires every <Direction> to set image=\"...\"."
                    .to_string(),
        });
    }
    let play_sprite = collect_self_closing_blocks(inner, "PlaySprite")?
        .first()
        .map(|node| parse_play_sprite_node(node))
        .transpose()?;

    Ok(WorldDirectionalCharacter {
        id: required_attr_value(open, "id", line)?,
        sheet,
        path_style: parse_path_style_attr(open, line)?,
        x: attr_value(open, "x").unwrap_or_else(|| "0".to_string()),
        y: attr_value(open, "y").unwrap_or_else(|| "0".to_string()),
        scale: attr_value(open, "scale").unwrap_or_else(|| "1".to_string()),
        yaw: attr_value(open, "yaw").unwrap_or_else(|| "0".to_string()),
        opacity: attr_value(open, "opacity").unwrap_or_else(|| "1".to_string()),
        play_sprite,
        directions,
    })
}

fn parse_play_sprite_node(block: &str) -> Result<WorldSpritePlayback, GraphParseError> {
    let line = 1;
    let frame_size = attr_value(block, "frameSize")
        .map(|raw| parse_size(&raw, line, "frameSize"))
        .transpose()?;
    let frame_width = attr_value(block, "frameWidth")
        .map(|raw| parse_u32(&raw, line, "frameWidth"))
        .transpose()?
        .or_else(|| frame_size.map(|size| size.0))
        .ok_or_else(|| GraphParseError {
            line,
            message: "PlaySprite requires frameWidth/frameHeight or frameSize={[w,h]}.".to_string(),
        })?;
    let frame_height = attr_value(block, "frameHeight")
        .map(|raw| parse_u32(&raw, line, "frameHeight"))
        .transpose()?
        .or_else(|| frame_size.map(|size| size.1))
        .ok_or_else(|| GraphParseError {
            line,
            message: "PlaySprite requires frameWidth/frameHeight or frameSize={[w,h]}.".to_string(),
        })?;
    Ok(WorldSpritePlayback {
        fps: attr_value(block, "fps").unwrap_or_else(|| "12".to_string()),
        r#loop: attr_value(block, "loop")
            .map(|raw| parse_bool(&raw, line, "loop"))
            .transpose()?
            .unwrap_or(true),
        frames: attr_value(block, "frames")
            .map(|raw| parse_u32(&raw, line, "frames"))
            .transpose()?
            .unwrap_or(1)
            .max(1),
        columns: attr_value(block, "columns")
            .map(|raw| parse_u32(&raw, line, "columns"))
            .transpose()?
            .unwrap_or(1)
            .max(1),
        frame_width: frame_width.max(1),
        frame_height: frame_height.max(1),
        start: attr_value(block, "start")
            .map(|raw| parse_u32(&raw, line, "start"))
            .transpose()?
            .unwrap_or(0),
        margin_x: attr_value(block, "marginX")
            .map(|raw| parse_u32(&raw, line, "marginX"))
            .transpose()?
            .unwrap_or(0),
        margin_y: attr_value(block, "marginY")
            .map(|raw| parse_u32(&raw, line, "marginY"))
            .transpose()?
            .unwrap_or(0),
        spacing_x: attr_value(block, "spacingX")
            .map(|raw| parse_u32(&raw, line, "spacingX"))
            .transpose()?
            .unwrap_or(0),
        spacing_y: attr_value(block, "spacingY")
            .map(|raw| parse_u32(&raw, line, "spacingY"))
            .transpose()?
            .unwrap_or(0),
    })
}

fn parse_direction_node(block: &str) -> Result<WorldDirectionFrame, GraphParseError> {
    let line = 1;
    let image = attr_value(block, "image").or_else(|| attr_value(block, "src"));
    let rect = attr_value(block, "rect")
        .map(|raw| parse_rect_u32(&raw, line, "rect"))
        .transpose()?;
    let anchor = attr_value(block, "anchor")
        .map(|raw| parse_point_f32(&raw, line, "anchor"))
        .transpose()?;
    if rect.is_none() && image.is_none() {
        return Err(GraphParseError {
            line,
            message: "Direction requires rect=\"[...]\" for sheet mode, or image=\"...\" for split-image mode."
                .to_string(),
        });
    }
    Ok(WorldDirectionFrame {
        name: attr_value(block, "name"),
        angle: attr_value(block, "angle")
            .map(|raw| parse_f32(&raw, line, "angle"))
            .transpose()?,
        camera_pitch: attr_value(block, "cameraPitch")
            .map(|raw| parse_f32(&raw, line, "cameraPitch"))
            .transpose()?,
        image,
        rect,
        anchor,
    })
}

fn parse_background_node(block: &str) -> Result<WorldBackground, GraphParseError> {
    let line = 1;
    Ok(WorldBackground {
        id: attr_value(block, "id"),
        src: attr_value(block, "src"),
        fit: attr_value(block, "fit")
            .map(|raw| parse_background_fit(&raw, line, "fit"))
            .transpose()?
            .unwrap_or_default(),
        color: attr_value(block, "color").unwrap_or_else(|| "#000000".to_string()),
        opacity: attr_value(block, "opacity").unwrap_or_else(|| "1".to_string()),
    })
}

fn parse_camera_node(block: &str) -> Result<WorldCamera, GraphParseError> {
    let line = 1;
    let mut camera = WorldCamera::default();
    camera.id = attr_value(block, "id").or(camera.id);
    if attr_value(block, "mode").is_some() {
        return Err(GraphParseError {
            line,
            message:
                "<World> Camera is always Camera3D; use control=\"orbit\" or control=\"free\" instead of mode=\"...\"."
                    .to_string(),
        });
    }
    camera.control = attr_value(block, "control")
        .or_else(|| attr_value(block, "cameraControl"))
        .map(|raw| parse_camera_control(&raw, line, "control"))
        .transpose()?
        .unwrap_or(camera.control);
    camera.projection = attr_value(block, "projection")
        .map(|raw| parse_camera_projection(&raw, line, "projection"))
        .transpose()?
        .unwrap_or(camera.projection);
    camera.target = attr_value(block, "target");
    let x_attr = attr_value(block, "x");
    let y_attr = attr_value(block, "y");
    let z_attr = attr_value(block, "z");
    camera.x = x_attr.clone().unwrap_or(camera.x);
    camera.y = y_attr.clone().unwrap_or(camera.y);
    camera.z = z_attr.clone().unwrap_or(camera.z);
    camera.target_x = attr_value(block, "targetX")
        .or(x_attr)
        .unwrap_or(camera.target_x);
    camera.target_y = attr_value(block, "targetY")
        .or(y_attr)
        .unwrap_or(camera.target_y);
    camera.target_z = attr_value(block, "targetZ")
        .or(z_attr)
        .unwrap_or(camera.target_z);
    camera.yaw = attr_value(block, "yaw").unwrap_or(camera.yaw);
    camera.pitch = attr_value(block, "pitch").unwrap_or(camera.pitch);
    camera.roll = attr_value(block, "roll").unwrap_or(camera.roll);
    camera.up_x = attr_value(block, "upX").unwrap_or(camera.up_x);
    camera.up_y = attr_value(block, "upY").unwrap_or(camera.up_y);
    camera.up_z = attr_value(block, "upZ").unwrap_or(camera.up_z);
    camera.distance = attr_value(block, "distance").unwrap_or(camera.distance);
    camera.zoom = attr_value(block, "zoom").unwrap_or(camera.zoom);
    camera.fov = attr_value(block, "fov").unwrap_or(camera.fov);
    camera.orthographic_scale = attr_value(block, "orthographicScale");
    Ok(camera)
}

fn parse_actor_node(open: &str, inner: &str) -> Result<WorldActor, GraphParseError> {
    let line = 1;
    let material = collect_self_closing_blocks(inner, "Material")?
        .first()
        .map(|node| parse_material_node(node))
        .transpose()?;
    let mut parsed_plays = collect_self_closing_blocks(inner, "Play")?
        .iter()
        .map(|node| parse_play_node(node))
        .collect::<Result<Vec<_>, _>>()?;
    let play = if parsed_plays.is_empty() {
        None
    } else {
        Some(parsed_plays.remove(0))
    };
    Ok(WorldActor {
        id: required_attr_value(open, "id", line)?,
        model: required_attr_value(open, "model", line)?,
        primitive: None,
        terrain: None,
        vegetation: None,
        path_style: parse_path_style_attr(open, line)?,
        hide_meshes: parse_name_list_attr(open, "hideMeshes"),
        hide_materials: parse_name_list_attr(open, "hideMaterials"),
        camera_hidden_bones: Vec::new(),
        profile: attr_value(open, "profile"),
        rig: attr_value(open, "rig"),
        retarget: attr_value(open, "retarget"),
        x: attr_value(open, "x").unwrap_or_else(|| "0".to_string()),
        y: attr_value(open, "y").unwrap_or_else(|| "0".to_string()),
        z: attr_value(open, "z").unwrap_or_else(|| "0".to_string()),
        yaw: attr_value(open, "yaw").unwrap_or_else(|| "0".to_string()),
        pitch: attr_value(open, "pitch").unwrap_or_else(|| "0".to_string()),
        roll: attr_value(open, "roll").unwrap_or_else(|| "0".to_string()),
        rotation_quaternion: None,
        scale: attr_value(open, "scale").unwrap_or_else(|| "1".to_string()),
        scale_mode: attr_value(open, "scaleMode").unwrap_or_else(|| "none".to_string()),
        opacity: attr_value(open, "opacity").unwrap_or_else(|| "1".to_string()),
        material,
        play,
        plays: parsed_plays,
    })
}

fn parse_retarget_block(block: &str) -> Result<WorldRetarget, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <Retarget> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</Retarget>".len())];
    let mut maps = Vec::new();
    for map in collect_self_closing_blocks(inner, "Map")? {
        maps.push(WorldRetargetMap {
            from: required_attr_value(&map, "from", line)?,
            to: required_attr_value(&map, "to", line)?,
        });
    }
    if maps.is_empty() {
        return Err(GraphParseError {
            line,
            message: "Retarget requires at least one <Map from=\"...\" to=\"...\" />.".to_string(),
        });
    }
    Ok(WorldRetarget {
        id: required_attr_value(open, "id", line)?,
        actor: attr_value(open, "actor"),
        preset: attr_value(open, "preset").unwrap_or_else(|| "humanoid_v1".to_string()),
        maps,
    })
}

fn parse_model_profile_block(block: &str) -> Result<WorldModelProfile, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <ModelProfile> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</ModelProfile>".len())];
    let preset = attr_value(open, "preset").unwrap_or_else(|| "humanoid_v1".to_string());

    let retarget = collect_tag_blocks(inner, "Retarget")?
        .first()
        .map(|block| parse_profile_retarget_block(block, &preset))
        .transpose()?;
    let bone_axis_map = collect_tag_blocks(inner, "BoneAxisMap")?
        .first()
        .map(|block| parse_bone_axis_map_block(block))
        .transpose()?;
    if !collect_tag_blocks(inner, "RestPoseCorrection")?.is_empty() {
        return Err(GraphParseError {
            line,
            message: "RestPoseCorrection has been removed. Put rest pose offsets on <BoneAxisMap><Axis ... /> with restForward/restSide/restTwist/restBend/restTurn."
                .to_string(),
        });
    }

    Ok(WorldModelProfile {
        id: required_attr_value(open, "id", line)?,
        model: required_attr_value(open, "model", line)?,
        preset,
        retarget,
        bone_axis_map,
    })
}

fn parse_profile_retarget_block(
    block: &str,
    default_preset: &str,
) -> Result<WorldProfileRetarget, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <Retarget> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</Retarget>".len())];
    let mut maps = Vec::new();
    for map in collect_self_closing_blocks(inner, "Map")? {
        maps.push(WorldRetargetMap {
            from: required_attr_value(&map, "from", line)?,
            to: required_attr_value(&map, "to", line)?,
        });
    }
    if maps.is_empty() {
        return Err(GraphParseError {
            line,
            message: "ModelProfile Retarget requires at least one <Map from=\"...\" to=\"...\" />."
                .to_string(),
        });
    }
    Ok(WorldProfileRetarget {
        preset: attr_value(open, "preset").unwrap_or_else(|| default_preset.to_string()),
        maps,
    })
}

fn parse_bone_axis_map_block(block: &str) -> Result<WorldBoneAxisMap, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <BoneAxisMap> block.".to_string(),
    })?;
    let inner = &block[open_end + 1..block.len().saturating_sub("</BoneAxisMap>".len())];
    let mut axes = Vec::new();
    for axis in collect_self_closing_blocks(inner, "Axis")? {
        axes.push(parse_bone_axis_node(&axis)?);
    }
    if axes.is_empty() {
        return Err(GraphParseError {
            line,
            message: "BoneAxisMap requires at least one <Axis bone=\"...\" ... />.".to_string(),
        });
    }
    Ok(WorldBoneAxisMap { axes })
}

fn parse_bone_axis_node(block: &str) -> Result<WorldBoneAxis, GraphParseError> {
    let line = 1;
    Ok(WorldBoneAxis {
        bone: required_attr_value(block, "bone", line)?,
        forward: attr_value(block, "forward"),
        side: attr_value(block, "side"),
        twist: attr_value(block, "twist"),
        bend: attr_value(block, "bend"),
        turn: attr_value(block, "turn"),
        rest_forward: attr_value(block, "restForward"),
        rest_side: attr_value(block, "restSide"),
        rest_twist: attr_value(block, "restTwist"),
        rest_bend: attr_value(block, "restBend"),
        rest_turn: attr_value(block, "restTurn"),
    })
}

fn parse_action_block(block: &str) -> Result<WorldAction, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <Action> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</Action>".len())];
    let duration_ms = attr_value(open, "duration")
        .map(|raw| parse_duration_ms(&raw, line, "Action.duration"))
        .transpose()?
        .unwrap_or(DEFAULT_WORLD_DURATION_MS);

    let mut poses = Vec::new();
    for pose in collect_tag_blocks(inner, "Pose")? {
        poses.push(parse_action_pose_block(&pose)?);
    }
    if poses.is_empty() {
        return Err(GraphParseError {
            line,
            message: "Action requires at least one <Pose> block.".to_string(),
        });
    }
    // Runtime sampling relies on one canonical order and can then use binary lookup.
    poses.sort_by(|a, b| a.t.total_cmp(&b.t));
    Ok(WorldAction {
        id: required_attr_value(open, "id", line)?,
        skeleton: required_attr_value(open, "skeleton", line)?,
        intent: attr_value(open, "intent"),
        duration_ms,
        poses,
        iks: Vec::new(),
    })
}

fn parse_action_pose_block(block: &str) -> Result<WorldActionPose, GraphParseError> {
    let line = 1;
    let open_end = block.find('>').ok_or_else(|| GraphParseError {
        line,
        message: "Malformed <Pose> block.".to_string(),
    })?;
    let open = &block[..=open_end];
    let inner = &block[open_end + 1..block.len().saturating_sub("</Pose>".len())];
    let mut bones = Vec::new();
    for bone in collect_self_closing_blocks(inner, "Bone")? {
        bones.push(parse_action_bone_node(&bone)?);
    }
    if bones.is_empty() {
        return Err(GraphParseError {
            line,
            message: "Pose requires at least one <Bone> node.".to_string(),
        });
    }
    Ok(WorldActionPose {
        t: parse_pose_time(&required_attr_value(open, "t", line)?, line, "Pose.t")?,
        label: attr_value(open, "label"),
        bones,
    })
}

fn parse_action_bone_node(block: &str) -> Result<WorldActionBone, GraphParseError> {
    let line = 1;
    Ok(WorldActionBone {
        id: required_attr_value(block, "id", line)?,
        x: attr_value(block, "x"),
        y: attr_value(block, "y"),
        z: attr_value(block, "z"),
        rotation: attr_value(block, "rotation"),
        rotation_x: attr_value(block, "rotationX"),
        rotation_y: attr_value(block, "rotationY"),
        rotation_z: attr_value(block, "rotationZ"),
        forward: attr_value(block, "forward"),
        side: attr_value(block, "side"),
        twist: attr_value(block, "twist"),
        bend: attr_value(block, "bend"),
        turn: attr_value(block, "turn"),
        scale: attr_value(block, "scale"),
        opacity: attr_value(block, "opacity"),
        interpolation: attr_value(block, "interpolation"),
        in_tangent: attr_value(block, "inTangent").or_else(|| attr_value(block, "in_tangent")),
        out_tangent: attr_value(block, "outTangent").or_else(|| attr_value(block, "out_tangent")),
    })
}

fn parse_apply_action_node(block: &str) -> Result<WorldApplyAction, GraphParseError> {
    let line = 1;
    Ok(WorldApplyAction {
        target: required_attr_value(block, "target", line)?,
        action: required_attr_value(block, "action", line)?,
        at_ms: attr_value(block, "at")
            .map(|raw| parse_duration_ms(&raw, line, "ApplyAction.at"))
            .transpose()?
            .unwrap_or(0),
        r#loop: attr_value(block, "loop")
            .map(|raw| parse_bool(&raw, line, "ApplyAction.loop"))
            .transpose()?
            .unwrap_or(false),
        weight: attr_value(block, "weight").unwrap_or_else(|| "1".to_string()),
        speed: attr_value(block, "speed").unwrap_or_else(|| "1".to_string()),
        blend_in: attr_value(block, "blendIn").unwrap_or_else(|| "0".to_string()),
        blend_out: attr_value(block, "blendOut").unwrap_or_else(|| "0".to_string()),
        mode: attr_value(block, "mode").unwrap_or_else(|| "override".to_string()),
        mask: attr_value(block, "mask")
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        duration_ms: attr_value(block, "duration")
            .map(|raw| parse_duration_ms(&raw, line, "ApplyAction.duration"))
            .transpose()?,
        root_motion: attr_value(block, "rootMotion").or_else(|| attr_value(block, "root_motion")),
        destination: attr_value(block, "destination"),
        face: attr_value(block, "face"),
        sync_group: attr_value(block, "syncGroup").or_else(|| attr_value(block, "sync_group")),
        sync_marker: attr_value(block, "syncMarker").or_else(|| attr_value(block, "sync_marker")),
    })
}

fn parse_material_node(block: &str) -> Result<WorldMaterial, GraphParseError> {
    let line = 1;
    Ok(WorldMaterial {
        style: attr_value(block, "style")
            .map(|raw| parse_material_style(&raw, line, "style"))
            .transpose()?
            .unwrap_or_default(),
        outline: attr_value(block, "outline")
            .map(|raw| parse_bool(&raw, line, "outline"))
            .transpose()?
            .unwrap_or(true),
        outline_width: attr_value(block, "outlineWidth").unwrap_or_else(|| "2".to_string()),
        exposure: attr_value(block, "exposure").unwrap_or_else(|| "1".to_string()),
    })
}

fn parse_play_node(block: &str) -> Result<WorldPlay, GraphParseError> {
    let line = 1;
    Ok(WorldPlay {
        clip: attr_value(block, "clip"),
        r#loop: attr_value(block, "loop")
            .map(|raw| parse_bool(&raw, line, "loop"))
            .transpose()?
            .unwrap_or(true),
        speed: attr_value(block, "speed").unwrap_or_else(|| "1".to_string()),
        weight: attr_value(block, "weight").unwrap_or_else(|| "1".to_string()),
        blend_in: attr_value(block, "blendIn").unwrap_or_else(|| "0".to_string()),
        blend_out: attr_value(block, "blendOut").unwrap_or_else(|| "0".to_string()),
        mask: attr_value(block, "mask")
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default(),
    })
}

fn validate_world_graph_refs(
    worlds: &[WorldNode],
    model_profiles: &[WorldModelProfile],
    retargets: &[WorldRetarget],
    actions: &[WorldAction],
    apply_actions: &[WorldApplyAction],
) -> Result<(), GraphParseError> {
    let line = 1;
    let actor_ids = worlds
        .iter()
        .flat_map(|world| world.actors.iter().map(|actor| actor.id.as_str()))
        .collect::<HashSet<_>>();
    let retarget_ids = retargets
        .iter()
        .map(|retarget| retarget.id.as_str())
        .collect::<HashSet<_>>();
    let profile_ids = model_profiles
        .iter()
        .map(|profile| profile.id.as_str())
        .collect::<HashSet<_>>();
    let action_ids = actions
        .iter()
        .map(|action| action.id.as_str())
        .collect::<HashSet<_>>();

    for world in worlds {
        for actor in &world.actors {
            if let Some(retarget) = actor.retarget.as_deref()
                && !retarget_ids.contains(retarget)
            {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Actor '{}' references missing Retarget '{}'.",
                        actor.id, retarget
                    ),
                });
            }
            if let Some(profile) = actor.profile.as_deref()
                && !profile_ids.contains(profile)
            {
                return Err(GraphParseError {
                    line,
                    message: format!(
                        "Actor '{}' references missing ModelProfile '{}'.",
                        actor.id, profile
                    ),
                });
            }
        }
    }
    for retarget in retargets {
        if let Some(actor) = retarget.actor.as_deref()
            && !actor_ids.contains(actor)
        {
            return Err(GraphParseError {
                line,
                message: format!(
                    "Retarget '{}' references missing Actor '{}'.",
                    retarget.id, actor
                ),
            });
        }
    }
    for apply in apply_actions {
        if !actor_ids.contains(apply.target.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("ApplyAction references missing Actor '{}'.", apply.target),
            });
        }
        if !action_ids.contains(apply.action.as_str()) {
            return Err(GraphParseError {
                line,
                message: format!("ApplyAction references missing Action '{}'.", apply.action),
            });
        }
    }
    Ok(())
}

fn collect_self_closing_blocks(input: &str, tag: &str) -> Result<Vec<String>, GraphParseError> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel_start) = input[search_from..].find('<') {
        let start = search_from + rel_start;
        if !starts_open_tag(&input[start..], tag) {
            search_from = start + 1;
            continue;
        }
        let Some(rel_end) = input[start..].find('>') else {
            return Err(GraphParseError {
                line: 1,
                message: format!("Missing > for <{tag}>."),
            });
        };
        let end = start + rel_end + 1;
        let block = &input[start..end];
        if block.trim_end().ends_with("/>") {
            out.push(block.to_string());
        }
        search_from = end;
    }
    Ok(out)
}

fn collect_tag_blocks(input: &str, tag: &str) -> Result<Vec<String>, GraphParseError> {
    let mut out = Vec::new();
    let mut search_from = 0usize;
    while let Some(rel_start) = input[search_from..].find('<') {
        let start = search_from + rel_start;
        if !starts_open_tag(&input[start..], tag) {
            search_from = start + 1;
            continue;
        }
        let Some(rel_open_end) = input[start..].find('>') else {
            return Err(GraphParseError {
                line: 1,
                message: format!("Missing > for <{tag}>."),
            });
        };
        let open_end = start + rel_open_end + 1;
        if input[start..open_end].trim_end().ends_with("/>") {
            search_from = open_end;
            continue;
        }
        let end = find_matching_close_tag(input, tag, start)?;
        out.push(input[start..end].to_string());
        search_from = end;
    }
    Ok(out)
}

fn strip_tag_blocks(input: &str, tag: &str) -> Result<String, GraphParseError> {
    let mut out = String::with_capacity(input.len());
    let mut search_from = 0usize;
    while let Some(rel_start) = input[search_from..].find('<') {
        let start = search_from + rel_start;
        if !starts_open_tag(&input[start..], tag) {
            out.push_str(&input[search_from..start + 1]);
            search_from = start + 1;
            continue;
        }
        out.push_str(&input[search_from..start]);
        let Some(rel_open_end) = input[start..].find('>') else {
            return Err(GraphParseError {
                line: 1,
                message: format!("Missing > for <{tag}>."),
            });
        };
        let open_end = start + rel_open_end + 1;
        let end = if input[start..open_end].trim_end().ends_with("/>") {
            open_end
        } else {
            find_matching_close_tag(input, tag, start)?
        };
        search_from = end;
    }
    out.push_str(&input[search_from..]);
    Ok(out)
}

fn find_matching_close_tag(input: &str, tag: &str, start: usize) -> Result<usize, GraphParseError> {
    let close = format!("</{tag}>");
    let mut cursor = start;
    let mut depth = 0usize;
    loop {
        let Some(rel_next) = input[cursor..].find('<') else {
            return Err(GraphParseError {
                line: 1,
                message: format!("Missing closing tag {close}."),
            });
        };
        let ix = cursor + rel_next;
        if input[ix..].starts_with(&close) {
            depth = depth.saturating_sub(1);
            let end = ix + close.len();
            if depth == 0 {
                return Ok(end);
            }
            cursor = end;
            continue;
        }
        if starts_open_tag(&input[ix..], tag) {
            let Some(rel_open_end) = input[ix..].find('>') else {
                return Err(GraphParseError {
                    line: 1,
                    message: format!("Missing > for <{tag}>."),
                });
            };
            let open_end = ix + rel_open_end + 1;
            if !input[ix..open_end].trim_end().ends_with("/>") {
                depth += 1;
            }
            cursor = open_end;
            continue;
        }
        cursor = ix + 1;
    }
}

fn starts_open_tag(input: &str, tag: &str) -> bool {
    let Some(rest) = input.strip_prefix('<') else {
        return false;
    };
    let Some(after_tag) = rest.strip_prefix(tag) else {
        return false;
    };
    after_tag
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_whitespace() || ch == '>' || ch == '/')
}

fn strip_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 4..];
        if let Some(end) = after_start.find("-->") {
            rest = &after_start[end + 3..];
        } else {
            break;
        }
    }
    out.push_str(rest);
    out
}

fn parse_fps(raw: &str, line: usize, field: &str) -> Result<f32, GraphParseError> {
    let text = strip_wrappers(raw);
    let value = text.parse::<f32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field}: {text}"),
    })?;
    if value <= 0.0 || !value.is_finite() {
        return Err(GraphParseError {
            line,
            message: format!("{field} must be a positive finite number."),
        });
    }
    Ok(value)
}

fn parse_duration_ms(raw: &str, line: usize, field: &str) -> Result<u64, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    if let Some(ms) = text.strip_suffix("ms") {
        let value = ms.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field}: {text}"),
        })?;
        return Ok(value.max(0.0).round() as u64);
    }
    if let Some(sec) = text.strip_suffix('s') {
        let value = sec.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field}: {text}"),
        })?;
        return Ok((value.max(0.0) * 1000.0).round() as u64);
    }
    let value = text.parse::<f32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field}: {text}"),
    })?;
    Ok((value.max(0.0) * 1000.0).round() as u64)
}

fn parse_pose_time(raw: &str, line: usize, field: &str) -> Result<f32, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    if let Some(ms) = text.strip_suffix("ms") {
        let value = ms.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field}: {text}"),
        })?;
        return Ok((value.max(0.0) / 1000.0).max(0.0));
    }
    if let Some(sec) = text.strip_suffix('s') {
        let value = sec.trim().parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field}: {text}"),
        })?;
        return Ok(value.max(0.0));
    }
    let value = text.parse::<f32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field}: {text}"),
    })?;
    Ok(value.max(0.0))
}

fn parse_size(raw: &str, line: usize, field: &str) -> Result<(u32, u32), GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{field} must be an array [width,height]."),
        })?;
    let mut parts = inner.split(',').map(str::trim);
    let width = parts
        .next()
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{field} is missing width."),
        })?
        .parse::<u32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} width."),
        })?;
    let height = parts
        .next()
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{field} is missing height."),
        })?
        .parse::<u32>()
        .map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} height."),
        })?;
    Ok((width, height))
}

fn parse_rect_u32(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<(u32, u32, u32, u32), GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{field} must be an array [x,y,width,height]."),
        })?;
    let parts = inner.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 4 {
        return Err(GraphParseError {
            line,
            message: format!("{field} must have exactly 4 numbers."),
        });
    }
    let mut values = [0u32; 4];
    for (index, part) in parts.iter().enumerate() {
        values[index] = part.parse::<u32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} value '{}'.", part),
        })?;
    }
    if values[2] == 0 || values[3] == 0 {
        return Err(GraphParseError {
            line,
            message: format!("{field} width and height must be positive."),
        });
    }
    Ok((values[0], values[1], values[2], values[3]))
}

fn parse_point_f32(raw: &str, line: usize, field: &str) -> Result<(f32, f32), GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let inner = text
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| GraphParseError {
            line,
            message: format!("{field} must be an array [x,y]."),
        })?;
    let parts = inner.split(',').map(str::trim).collect::<Vec<_>>();
    if parts.len() != 2 {
        return Err(GraphParseError {
            line,
            message: format!("{field} must have exactly 2 numbers."),
        });
    }
    Ok((
        parts[0].parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} x value."),
        })?,
        parts[1].parse::<f32>().map_err(|_| GraphParseError {
            line,
            message: format!("Invalid {field} y value."),
        })?,
    ))
}

fn parse_f32(raw: &str, line: usize, field: &str) -> Result<f32, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    let value = text.parse::<f32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field}: {text}"),
    })?;
    if !value.is_finite() {
        return Err(GraphParseError {
            line,
            message: format!("{field} must be finite."),
        });
    }
    Ok(value)
}

fn parse_u32(raw: &str, line: usize, field: &str) -> Result<u32, GraphParseError> {
    let text = strip_wrappers(raw).trim();
    text.parse::<u32>().map_err(|_| GraphParseError {
        line,
        message: format!("Invalid {field}: {text}"),
    })
}

fn parse_path_style_attr(block: &str, line: usize) -> Result<WorldPathStyle, GraphParseError> {
    let Some(raw) = attr_value(block, "pathstyle").or_else(|| attr_value(block, "pathStyle"))
    else {
        return Ok(WorldPathStyle::Relative);
    };
    match normalize_ident(&raw).as_str() {
        "relative" => Ok(WorldPathStyle::Relative),
        "absolute" => Ok(WorldPathStyle::Absolute),
        other => Err(GraphParseError {
            line,
            message: format!(
                "Invalid pathstyle '{other}'. Expected pathstyle=\"relative\" or pathstyle=\"absolute\"."
            ),
        }),
    }
}

fn parse_bool(raw: &str, line: usize, field: &str) -> Result<bool, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {field} '{other}'. Expected true or false."),
        }),
    }
}

fn parse_background_fit(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<WorldBackgroundFit, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "cover" => Ok(WorldBackgroundFit::Cover),
        "contain" => Ok(WorldBackgroundFit::Contain),
        "stretch" => Ok(WorldBackgroundFit::Stretch),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {field} '{other}'. Expected cover, contain, or stretch."),
        }),
    }
}

fn parse_camera_control(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<WorldCameraControl, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "orbit" => Ok(WorldCameraControl::Orbit),
        "free" => Ok(WorldCameraControl::Free),
        other => Err(GraphParseError {
            line,
            message: format!(
                "Invalid {field} '{other}'. <World> Camera is always 3D; use control=\"orbit\" or control=\"free\"."
            ),
        }),
    }
}

fn parse_camera_projection(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<WorldCameraProjection, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "perspective" => Ok(WorldCameraProjection::Perspective),
        "orthographic" => Ok(WorldCameraProjection::Orthographic),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {field} '{other}'. Expected perspective or orthographic."),
        }),
    }
}

fn parse_material_style(
    raw: &str,
    line: usize,
    field: &str,
) -> Result<WorldMaterialStyle, GraphParseError> {
    match normalize_ident(raw).as_str() {
        "toon" => Ok(WorldMaterialStyle::Toon),
        "pbr" => Ok(WorldMaterialStyle::Pbr),
        "unlit" => Ok(WorldMaterialStyle::Unlit),
        other => Err(GraphParseError {
            line,
            message: format!("Invalid {field} '{other}'. Expected toon, pbr, or unlit."),
        }),
    }
}

fn normalize_ident(raw: &str) -> String {
    strip_wrappers(raw)
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
}

fn required_attr_value(block: &str, key: &str, line: usize) -> Result<String, GraphParseError> {
    attr_value(block, key).ok_or_else(|| GraphParseError {
        line,
        message: format!("Missing required attribute: {key}"),
    })
}

fn parse_name_list_attr(block: &str, key: &str) -> Vec<String> {
    attr_value(block, key)
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn attr_value(block: &str, key: &str) -> Option<String> {
    let start = find_attr_start(block, key)?;
    let mut rest = block[start..].trim_start();
    if !rest.starts_with('=') {
        return None;
    }
    rest = rest[1..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    if let Some(stripped) = rest.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        return Some(stripped[..end].to_string());
    }
    if let Some(stripped) = rest.strip_prefix('{') {
        let mut depth = 1usize;
        let mut in_double_quote = false;
        let mut escape = false;
        let mut out = String::new();
        for ch in stripped.chars() {
            if escape {
                out.push(ch);
                escape = false;
                continue;
            }
            if ch == '\\' {
                out.push(ch);
                escape = true;
                continue;
            }
            if ch == '"' {
                in_double_quote = !in_double_quote;
                out.push(ch);
                continue;
            }
            if !in_double_quote && ch == '{' {
                depth += 1;
                out.push(ch);
                continue;
            }
            if !in_double_quote && ch == '}' {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(out);
                }
                out.push(ch);
                continue;
            }
            out.push(ch);
        }
        return None;
    }
    let end = rest
        .find(|ch: char| ch.is_whitespace() || ch == '>' || ch == '/')
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn find_attr_start(block: &str, key: &str) -> Option<usize> {
    let bytes = block.as_bytes();
    let key_bytes = key.as_bytes();
    if key_bytes.is_empty() || bytes.len() < key_bytes.len() + 1 {
        return None;
    }
    let mut in_double_quote = false;
    let mut in_single_quote = false;
    let mut i = 0usize;
    while i + key_bytes.len() < bytes.len() {
        let b = bytes[i];
        if b == b'"' && !in_single_quote {
            in_double_quote = !in_double_quote;
            i += 1;
            continue;
        }
        if b == b'\'' && !in_double_quote {
            in_single_quote = !in_single_quote;
            i += 1;
            continue;
        }
        if in_double_quote || in_single_quote {
            i += 1;
            continue;
        }
        if &bytes[i..i + key_bytes.len()] == key_bytes {
            let prev_ok = i == 0 || bytes[i - 1].is_ascii_whitespace() || bytes[i - 1] == b'<';
            let mut j = i + key_bytes.len();
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if prev_ok && j < bytes.len() && bytes[j] == b'=' {
                return Some(i + key_bytes.len());
            }
        }
        i += 1;
    }
    None
}

fn strip_wrappers(raw: &str) -> &str {
    let mut text = raw.trim();
    loop {
        if text.starts_with('{') && text.ends_with('}') && text.len() >= 2 {
            text = text[1..text.len() - 1].trim();
            continue;
        }
        if text.starts_with('"') && text.ends_with('"') && text.len() >= 2 {
            text = text[1..text.len() - 1].trim();
            continue;
        }
        if text.starts_with('\'') && text.ends_with('\'') && text.len() >= 2 {
            text = text[1..text.len() - 1].trim();
            continue;
        }
        break;
    }
    text
}

#[cfg(test)]
mod tests {
    use crate::world::{WorldCameraControl, WorldCameraProjection, WorldPathStyle};

    use super::parse_world_graph_script;

    #[test]
    fn parses_basic_world_graph() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="3s" size={[1280,720]}>
  <World id="stage">
    <Background id="bg" src="../scene/environments/forest_path_static.png" fit="cover" color="#87c9ff" />
    <Camera id="cam" control="orbit" target="hero" yaw={curve("0:0:linear,3:360:linear")} pitch="0" distance="3.4" fov="35" />
    <Actor id="hero" model="characters/your_character.glb" x="0" y="0" z="0" yaw="0" scale="1">
      <Material style="toon" outline="true" outlineWidth="2" />
      <Play clip="Idle" loop="true" speed="1" />
    </Actor>
  </World>
  <Present from="stage" />
</Graph>"##,
        )
        .expect("world graph");
        assert_eq!(graph.worlds.len(), 1);
        assert_eq!(graph.present.from, "stage");
        assert_eq!(graph.worlds[0].actors[0].id, "hero");
        assert_eq!(graph.duration_ms, 3000);
    }

    #[test]
    fn parses_retarget_action_and_apply_action() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="2s" size={[640,360]}>
  <World id="stage">
    <Background color="#000000" />
    <Actor id="anime" model="characters/your_character.glb" rig="humanoid_v1" retarget="anime_humanoid_map" />
  </World>

  <Retarget id="anime_humanoid_map" actor="anime" preset="humanoid_v1">
    <Map from="Hips_183" to="hips" />
    <Map from="Right arm_68" to="upper_arm_r" />
    <Map from="Right elbow_67" to="forearm_r" />
  </Retarget>

  <Action id="wave_hand" skeleton="humanoid_v1" duration="2s">
    <Pose t="0">
      <Bone id="upper_arm_r" rotationZ="0" />
      <Bone id="forearm_r" rotationZ="0" />
    </Pose>
    <Pose t="0.5">
      <Bone id="upper_arm_r" rotationZ="-55" />
      <Bone id="forearm_r" rotationZ="-70" />
    </Pose>
  </Action>

  <ApplyAction target="anime" action="wave_hand" at="0s" duration="1.5s"
               loop="true" rootMotion="clip" destination="exit"
               face="forward" syncGroup="walk" syncMarker="left_foot" />
  <Present from="stage" />
</Graph>"##,
        )
        .expect("world graph");
        assert_eq!(graph.retargets.len(), 1);
        assert_eq!(graph.retargets[0].maps.len(), 3);
        assert_eq!(graph.actions.len(), 1);
        assert_eq!(graph.actions[0].poses.len(), 2);
        assert_eq!(graph.apply_actions.len(), 1);
        assert_eq!(graph.apply_actions[0].duration_ms, Some(1_500));
        assert_eq!(graph.apply_actions[0].root_motion.as_deref(), Some("clip"));
        assert_eq!(graph.apply_actions[0].destination.as_deref(), Some("exit"));
        assert_eq!(graph.apply_actions[0].face.as_deref(), Some("forward"));
        assert_eq!(graph.apply_actions[0].sync_group.as_deref(), Some("walk"));
        assert_eq!(
            graph.apply_actions[0].sync_marker.as_deref(),
            Some("left_foot")
        );
        assert_eq!(
            graph.worlds[0].actors[0].retarget.as_deref(),
            Some("anime_humanoid_map")
        );
    }

    #[test]
    fn parses_action_interpolation_and_tangents_without_changing_linear_default() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <World id="stage"><Actor id="hero" model="hero.glb" /></World>
  <Action id="curve" skeleton="humanoid_v1" duration="1s">
    <Pose t="0s"><Bone id="hips" y="0" interpolation="bezier" outTangent="1.5" /></Pose>
    <Pose t="1s"><Bone id="hips" y="1" inTangent="-0.5" /></Pose>
  </Action>
  <ApplyAction target="hero" action="curve" />
  <Present from="stage" />
</Graph>"##,
        )
        .expect("world action interpolation");
        let first = &graph.actions[0].poses[0].bones[0];
        let second = &graph.actions[0].poses[1].bones[0];
        assert_eq!(first.interpolation.as_deref(), Some("bezier"));
        assert_eq!(first.out_tangent.as_deref(), Some("1.5"));
        assert_eq!(second.in_tangent.as_deref(), Some("-0.5"));
        assert_eq!(second.interpolation, None);
    }

    #[test]
    fn parses_action_poses_into_runtime_time_order() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <World id="stage"><Actor id="hero" model="hero.glb" /></World>
  <Action id="ordered" skeleton="humanoid_v1" duration="1s">
    <Pose t="1s"><Bone id="hips" y="1" /></Pose>
    <Pose t="0s"><Bone id="hips" y="0" /></Pose>
    <Pose t="0.5s"><Bone id="hips" y="0.5" /></Pose>
  </Action>
  <ApplyAction target="hero" action="ordered" />
  <Present from="stage" />
</Graph>"##,
        )
        .expect("world action pose ordering");
        let times = graph.actions[0]
            .poses
            .iter()
            .map(|pose| pose.t)
            .collect::<Vec<_>>();
        assert_eq!(times, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn parses_no_scope_world_block_with_top_level_camera() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="4s" size={[1280,720]}>
  <Background color="#000000" />
  <Camera id="main_camera" target="anime" x={curve("0:-0.35:ease_in_out, 4:0:ease_in_out")} y="0" zoom={curve("0:1.0:linear, 4:1.0:ease_in_out")} fov="35" />
  <World id="character_world">
    <Actor id="anime" model="characters/your_character.glb" rig="humanoid_v1" retarget="map" />
    <Retarget id="map" actor="anime" preset="humanoid_v1">
      <Map from="Right arm_68" to="upper_arm_r" />
    </Retarget>
    <Action id="wave_hand" skeleton="humanoid_v1" duration="2s">
      <Pose t="0">
        <Bone id="upper_arm_r" rotationZ="8" />
      </Pose>
      <Pose t="1">
        <Bone id="upper_arm_r" rotationZ="-48" />
      </Pose>
    </Action>
    <ApplyAction target="anime" action="wave_hand" at="0s" loop="true" />
  </World>
  <Present from="final" />
</Graph>"##,
        )
        .expect("world graph");
        assert_eq!(graph.worlds.len(), 1);
        assert_eq!(graph.worlds[0].id, "character_world");
        assert_eq!(graph.present.from, "character_world");
        assert_eq!(
            graph.worlds[0].camera.zoom,
            "curve(\"0:1.0:linear, 4:1.0:ease_in_out\")"
        );
        assert_eq!(graph.retargets.len(), 1);
        assert_eq!(graph.actions.len(), 1);
        assert_eq!(graph.apply_actions.len(), 1);
    }

    #[test]
    fn parses_world_camera_control_attr() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <World id="world0">
    <Camera control="free" projection="orthographic" orthographicScale="2.4" />
  </World>
  <Present from="world0" />
</Graph>"##,
        )
        .expect("world camera control");
        assert_eq!(graph.worlds[0].camera.control, WorldCameraControl::Free);
        assert_eq!(
            graph.worlds[0].camera.projection,
            WorldCameraProjection::Orthographic
        );
        assert_eq!(
            graph.worlds[0].camera.orthographic_scale.as_deref(),
            Some("2.4")
        );
    }

    #[test]
    fn rejects_world_camera_mode_attr() {
        let err = parse_world_graph_script(
            r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <World id="world0">
    <Camera mode="orbit" />
  </World>
  <Present from="world0" />
</Graph>"##,
        )
        .expect_err("World Camera mode attr must be rejected");
        assert!(err.message.contains("Camera3D"), "unexpected error: {err}");
    }

    #[test]
    fn parses_world_directional_character() {
        let graph = parse_world_graph_script(
            r##"<Graph fps={30} duration="1s" size={[320,180]}>
  <World id="sprite_stage">
    <Background color="#000000" />
    <Camera yaw="45" pitch="0" zoom="1" />
    <Actor id="glb_actor"
           pathstyle="absolute"
           model="/tmp/example.glb"
           x="0"
           y="0"
           z="0" />
    <DirectionalCharacter id="hero" pathstyle="relative" sheet="sprites/hero/run.png" x="160" y="170" scale="1" yaw="90">
      <PlaySprite fps="12" loop="true" frameSize={[74,86]} columns="7" frames="28" />
      <DirectionMap mode="nearest">
        <Direction angle="0" image="sprites/hero/front_0.png" anchor={[32,90]} />
        <Direction angle="90" image="sprites/hero/right_90.png" anchor={[32,90]} />
        <Direction name="top" cameraPitch="90" image="sprites/hero/top.png" anchor={[32,48]} />
      </DirectionMap>
    </DirectionalCharacter>
  </World>
  <Present from="sprite_stage" />
</Graph>"##,
        )
        .expect("directional character graph");
        assert_eq!(graph.worlds.len(), 1);
        assert_eq!(graph.worlds[0].id, "sprite_stage");
        assert_eq!(graph.worlds[0].actors.len(), 1);
        assert_eq!(
            graph.worlds[0].actors[0].path_style,
            WorldPathStyle::Absolute
        );
        assert_eq!(graph.worlds[0].directional_characters.len(), 1);
        let character = &graph.worlds[0].directional_characters[0];
        assert_eq!(character.id, "hero");
        assert_eq!(character.sheet.as_deref(), Some("sprites/hero/run.png"));
        assert_eq!(character.path_style, WorldPathStyle::Relative);
        let play_sprite = character.play_sprite.as_ref().expect("play sprite");
        assert_eq!(play_sprite.fps, "12");
        assert!(play_sprite.r#loop);
        assert_eq!(play_sprite.frame_width, 74);
        assert_eq!(play_sprite.frame_height, 86);
        assert_eq!(play_sprite.columns, 7);
        assert_eq!(play_sprite.frames, 28);
        assert_eq!(character.directions.len(), 3);
        assert_eq!(character.directions[1].angle, Some(90.0));
        assert_eq!(
            character.directions[1].image.as_deref(),
            Some("sprites/hero/right_90.png")
        );
        assert_eq!(character.directions[2].name.as_deref(), Some("top"));
    }
}
