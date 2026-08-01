# MotionLoom Example Guide

This guide is for AI agents and contributors generating MotionLoom DSL. Prefer selecting and adapting an existing example before writing a new graph from scratch.

## Core Rules

- Use `<Graph>` for standalone generated scenes and vector illustrations.
- Write graph attributes in MotionLoom DSL form: `fps={30}`, `duration="3s"`, `size={[1920,1080]}`, not `fps="60"` or `size="[1920,1080]"`.
- Keep `<Present from="..." />` outside `<Scene>`. The `from` value should match the scene id.
- Use `<Present from="scene0" />` for scene output. Use `scene:scene0` only inside `<Tex from="scene:scene0" />`.
- Prefer `curve("...")` for world timing instead of long inline arithmetic.
- Do not write arithmetic around `curve(...)` unless the runtime explicitly supports that context. Put full start/end values inside the curve points.
- Every generated standalone scene must include at least one semantic `Group id="..."`.
- Do not generate anonymous `<Group>` nodes. If the scene has parts, split them into semantic groups such as `background_group`, `eye_main`, `sclera_group`, `iris_group`, `highlight_group`, `lash_group`, `character_left`, or `chart_bars`.
- Prefer modifying one localized group over replacing the whole graph when attaching Vector Lab output.
- For exported vector drawing, use brush/stroke attributes on fewer semantic paths when possible instead of many repeated one-off paths.
- If the full-frame background is static, use `<Background color="..."/>` only. Do not add a full-canvas `<Rect>` that duplicates the same background color.
- Only use a full-canvas `<Rect>` for a background when it needs timeline animation, blend mode, opacity animation, masking, clipping, or scene-local layering.

## Curve Patterns

Use built-in easing:

```code
x={curve("0:140:ease_in_out, 1:760:linear")}
```

Use custom cubic easing:

```code
x={curve("0:140:ease(0.82,0,0.58,1), 1:760:linear")}
```

Avoid this form unless the runtime supports nested curve arithmetic in that attribute:

```code
x={140 + curve("0:0:ease_in_out, 1:620:linear")}
```

## Example Categories

### Components And Business Graphics

Use these for charts, dashboard-style scenes, cards, labels, simple data motion, and presentation graphics.

- `scene/components/business_barchart_defs.motionloom`
  - Animated business bar chart.
  - Good reference for `Defs`, gradients, Rect bars, Text labels, and timed bar growth.
- `scene/components/card_ani.motionloom`
  - Card-style world reference.
- `scene/components/polyline_and_path.motionloom`
  - Basic line/path primitive reference.

### Curve Motion

Use these when the task is mainly about timing, easing, or explaining world curves.

- `scene/curve_motion/ease_in_out_bezier.motionloom`
  - Reference for comparing built-in `ease_in_out` with custom cubic `ease(...)`.

### Eyes

Use these for anime eye structure, iris rendering, highlight layout, and blink world.

- `scene/eyes/eyes1/eyes_level0.motionloom` through `scene/eyes/eyes1/eyes_level12.motionloom`
  - Progressive eye construction references.
  - Use later levels for more detailed iris and eyelash layout.
- `scene/eyes/eyes2/eyes_level1.motionloom`
  - Palette/reference based single-eye reconstruction.
- `scene/eyes/eyes2/eyes_level2.motionloom`
  - Paired high-detail eyes.
  - Good reference for iris groups, mirrored left/right eye structure, highlights, and eyelid masses.
- `scene/eyes/eyes2/eyes_level3.motionloom`
  - Blink world reference using open-eye and closed-eye layers.
- `scene/eyes/eyes3/eyes_level1.motionloom`
  - Additional eye style reference.

### Characters

Use these for character assembly, face/hair/vector trace experiments, stick figures, and simple scene world.

- `scene/characters/characters1/character_level1.motionloom` through `scene/characters/characters1/character_level5.motionloom`
  - Progressive character/vector construction references.
  - Use `character_level5.motionloom` when simplifying Vector Lab output into grouped, semantic DSL.
- `scene/characters/characters2_stickman_office_three_scene/character_level1.motionloom`
  - Static three-stickman office scene.
- `scene/characters/characters2_stickman_office_three_scene/character_level2.motionloom`
  - Animated office stickman scene.
- `scene/characters/characters2_stickman_office_three_scene/character_level3.motionloom`
  - Improved animated office stickman scene with corrected head shake and bent-leg dance motion.
- `scene/characters/characters3/stickman_jump_walk_handshake.motionloom`
  - Simple action sequence reference: jump, walk, handshake.

### Legacy World GLB Experiments

These files document the retired `<World>` syntax for migration and renderer
compatibility testing only. Do not copy them into new DSL. New GLB work belongs
inside a `<Scene>` with a `space="3d"` track, `CompositeGroup`, `Camera3D`, and
`Model`.

- `world/scenes/glb_camera_static_bg_level1.motionloom`
  - Static forest background plus one GLB actor and orbit camera.
- `world/scenes/glb_retarget_action_run_level2.motionloom`
  - Reference for the preferred GLB action DSL structure: `<Actor rig="humanoid_v1" retarget="...">`, `<Retarget>`, `<Action>`, and `<ApplyAction>`.
  - Current renderer parses this structure; full bone skinning is a separate implementation step.

Preferred retarget/action structure:

```code
<Actor id="anime"
       model="your_character.glb"
       rig="humanoid_v1"
       retarget="anime_humanoid_map" />

<Retarget id="anime_humanoid_map" actor="anime" preset="humanoid_v1">
  <Map from="Right arm_68" to="upper_arm_r" />
  <Map from="Right elbow_67" to="forearm_r" />
</Retarget>

<Action id="wave_hand" skeleton="humanoid_v1" duration="2s">
  <Pose t="0.5">
    <Bone id="upper_arm_r" rotationZ="-55" />
    <Bone id="forearm_r" rotationZ="-70" />
  </Pose>
</Action>

<ApplyAction target="anime" action="wave_hand" at="0s" loop="true" />
```

## Vector Lab Output Guidelines

When sending Vector Lab content to MotionLoom:

- Use a group id that describes the asset, for example `hair_front_lines`, `left_eye_trace`, or `office_character_right`.
- If no group id is provided, generate a new group id such as `vector_group_01`, `vector_group_02`, etc.
- If a provided group id already exists, replace only that group, not the whole graph.
- Prefer compact brush definitions instead of repeating the same stroke attributes on every path.
- Preserve brush attributes such as `strokeStyle`, `strokeRoughness`, `strokeCopies`, `strokeTexture`, `strokeBristles`, and `strokePressure`.
- Merge related strokes into semantic groups where possible.

Compact brush pattern:

```code
<Defs>
  <Brush id="hair_pencil"
         stroke="#111111"
         strokeWidth="1.6"
         strokeStyle="pencil"
         strokeRoughness="1.8"
         strokeCopies="6"
         strokeTexture="0.7"
         strokeBristles="5"
         strokePressure="auto"
         strokePressureMin="0.2"
         strokePressureCurve="1.7"
         opacity="0.4"
         lineCap="round"
         lineJoin="round"
         fill="none" />
</Defs>

<Group id="hair_front_lines" brush="hair_pencil" x="0" y="0" opacity="1">
  <Path id="hair_line_01" d="M 268 293.6 L 266.7 287.3 L 270.4 289.8" />
  <Path id="hair_line_02" d="M 263.1 292.6 L 268.1 292.3 L 269.4 288.3" />
</Group>
```

Use `Path brush="another_brush"` only when one path needs a different brush from its parent group.

## Common Mistakes To Avoid

Wrong: `<Present>` inside `<Scene>`.

```code
<Scene id="demo">
  <Background color="#000000" />
  <Present from="demo" />
</Scene>
```

Correct:

```code
<Scene id="demo">
  <Background color="#000000" />
</Scene>

<Present from="demo" />
```

Wrong: replacing the entire MotionLoom graph when the user asks to attach Vector Lab output.

Correct: attach or replace only the target group.

Wrong: using many anonymous paths when a semantic group would be clearer.

Correct: group related paths under meaningful ids such as `eye_left`, `iris_group_left`, `hair_bangs`, or `bar_group_q1`.

Wrong: anonymous groups and stringified graph arrays.

```code
<Graph fps="60" duration="3" size="[1920,1080]">
  <Scene id="scene0">
    <Group>
      <Path d="M 0 0 L 100 100" stroke="#ffffff" />
    </Group>
  </Scene>
  <Present from="scene:scene0" />
</Graph>
```

Correct:

```code
<Graph fps={30} duration="3s" size={[1920,1080]}>
  <Scene id="scene0">
    <Group id="main_shape" x="0" y="0" opacity="1">
      <Path id="main_line" d="M 0 0 L 100 100" stroke="#ffffff" fill="none" />
    </Group>
  </Scene>
  <Present from="scene0" />
</Graph>
```
