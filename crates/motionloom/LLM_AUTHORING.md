# MotionLoom LLM Authoring Guide

Use this guide when generating or editing MotionLoom DSL. Prefer valid,
predictable, editable, and renderable output over the shortest possible script.

## MotionLoom Agent Authoring Protocol

An agent must treat MotionLoom authoring as a five-stage feedback loop:

```text
1. Example retrieval
        ↓
2. Syntax discovery
        ↓
3. DSL authoring
        ↓
4. Static analysis and repair
        ↓
5. Render verification
        └─────────── revise DSL and repeat ───────────┘
```

Parsing successfully is not the completion condition. A completed authoring
run has passed static analysis and has render evidence that is consistent with
the original visual intent.

### Stage 1 — Example retrieval

**Input:** the user's visual intent, format, duration, assets, and constraints.

Search `example.json` records by title, domain, features, and `teaches` fields.
Choose the smallest relevant set of examples. Metadata identifies what an
example is useful for; it is not executable DSL and must not be copied into a
MotionLoom document.

**Output:** one or more relevant showcase IDs and an explicit reason for each
selection.

### Stage 2 — Syntax discovery

Read the selected showcase's generated `schema.json` before copying or adapting
its syntax. It describes the tags, attributes, representative values,
animation properties, and asset kinds demonstrated by that example.

The per-showcase schema is a learning aid, not the complete engine schema. An
attribute absent from one showcase is not necessarily unsupported globally.
Query the engine capability APIs when introducing syntax not demonstrated by
the selected examples.

**Output:** the relevant demonstrated syntax plus any engine capabilities that
must be queried separately.

### Stage 3 — DSL authoring

Read `main.motionloom` to understand the complete hierarchy, IDs, dependency
flow, timing, and composition pattern. Generate or edit a complete portable DSL
document based on the user's request. Preserve unrelated source formatting and
IDs during local edits whenever possible.

Do not treat a copied showcase as the final answer: adapt duration, dimensions,
content, assets, animation, and presentation to the requested result.

**Output:** one candidate `.motionloom` document.

### Stage 4 — Static analysis and repair

Call `motionloom_analyze_script_json(script)` after every generated revision,
or use `motionloom_analyze_script_for_target_json(script, target)` when the
destination renderer is known.

- `unrenderable`: repair parse failure before rendering.
- `needs-repair`: apply error diagnostics, then analyze again.
- `needs-review`: decide whether every warning is intentional.
- `clean`: static validation is complete; continue to render verification.

Use the diagnostic `code`, source location, authored/effective values, runtime
effect, and `suggestions`. High-confidence suggestions may be applied
automatically, but the complete document must be analyzed again afterwards.

**Output:** a `clean` report, or a reviewed report whose remaining warnings are
explicitly accepted by the host or user.

### Stage 5 — Render verification

Render representative evidence at the intended aspect ratio and output size.
At minimum inspect an opening, middle, and final frame; motion-dependent work
also requires a short preview or equivalent temporal samples.

Evaluate the evidence against the original intent, including composition,
readability, timing, asset visibility, deformation, camera framing, and effect
strength. A clean authoring report proves structural validity, not visual
quality.

If the evidence does not match the intent, revise the DSL and return to Stage
4. Never bypass static analysis after a visual correction.

**Output:** accepted render evidence associated with the analyzed DSL revision.

### Artifact responsibilities

| Artifact | Answers | Must not be treated as |
| --- | --- | --- |
| `example.json` | What does this example demonstrate? | Executable DSL |
| `schema.json` | Which syntax does this example demonstrate? | The complete engine schema |
| `main.motionloom` | How is the complete composition authored? | Proof that the requested visual result is correct |
| Authoring report | What is invalid, ignored, conflicting, or target-incompatible? | A visual-quality score |
| Render evidence | Does the rendered result match the visual intent? | A replacement for static validation |

For retrieval or post-training datasets, keep these artifacts linked by a
stable example ID and DSL revision. Do not pair render evidence or an authoring
report with a different source revision.

For editable timelines, query `animation_property_schema_json()` before
authoring `AnimationTarget`. The registry declares number, Vector3, color,
path, and discrete value types plus their interpolation rules. Existing inline
`curve(...)` syntax remains valid and is preferable for compact numeric motion;
use `AnimationTarget` when a host editor must read and write explicit keys.

If both forms address the same node/property, `AnimationTarget` is the editor
override. Do not emit duplicate channels. Use `inspect_animation_targets()` to
obtain structured diagnostics and repair suggestions.

## Analyze After Every Revision

Call `motionloom_analyze_script_json(script)` after an LLM creates or edits a
document. This is a post-parse authoring report, not the per-showcase learning
schema. It keeps existing permissive parsing compatible while exposing ignored
attributes, duplicate attributes, parse and Process compile failures, invalid
animation targets, renderer limitations, and target compatibility issues.

The repair loop is:

1. Generate or edit the MotionLoom DSL.
2. Call `motionloom_analyze_script_json(script)`, or
   `motionloom_analyze_script_for_target_json(script, target)` for a specific
   host such as `wasm-webgpu`.
3. Do not render when `renderable` is `false`.
4. Apply high-confidence entries in each diagnostic's `suggestions` array.
5. Re-run the analyzer until `status` is `clean` or the remaining
   `needs-review` warnings are intentional.

Each diagnostic contains a stable `code`, `phase`, source `line` and `column`,
the affected `tag`, `nodeId`, and `attribute` when known, the authored and
effective values when they differ, the runtime `effect`, and concrete repair
suggestions. Parse failures are returned as normal JSON reports with
`status: "unrenderable"`; the WASM API does not throw merely because authored
DSL is invalid.

```json
{
  "status": "needs-repair",
  "parseSucceeded": true,
  "compileSucceeded": true,
  "renderable": false,
  "summary": { "errors": 1, "warnings": 0, "ignoredAttributes": 0 },
  "diagnostics": [{
    "code": "DUPLICATE_ATTRIBUTE",
    "phase": "authoring",
    "line": 239,
    "tag": "Group",
    "attribute": "y",
    "effect": "Only one value can be authoritative; parser behavior is ambiguous.",
    "suggestions": [{
      "kind": "remove-duplicate",
      "message": "Keep one y attribute and remove the duplicate.",
      "confidence": 1.0
    }]
  }]
}
```

Every showcase may also include a generated `schema.json`. That smaller file
describes only the tags, attributes, representative values, animation
properties, and asset kinds demonstrated by that example, so an LLM can learn
from the example without loading the complete engine registry. Generate it
with `motionloom_showcase_schema_json(script)`; do not substitute it for the
authoring report.

## Choose the Authoring Root

- **Scene graph**: all renderable 2D, 2.5D, and true-3D content, including
  vector graphics, text, animation, characters, cameras, masks, rigs, models,
  and composition.
- **Process graph**: media input, textures, compute effects, and multi-pass image
  processing.
- Put 3D content in a `<Scene>` using a `space="3d"` track and
  `CompositeGroup`, `Camera3D`, and `Model`.
- Scene and Process resources may coexist when a composition needs
  post-processing; keep their texture dependencies explicit.

## Canonical Scene Structure

Use the complete hierarchy. Do not invent shorthand that places visual nodes
directly below `<Scene>`.

```xml
<Graph fps={30} duration="3s" size={[1920,1080]}>
  <Background color="#000000" />

  <Scene id="example_scene">
    <Timeline>
      <Track id="main" space="world" z="0">
        <Sequence from="0s" duration="3s" out="hold">
          <Layer>
            <Text id="title" value="HELLO" x="center" y="center"
                  fontSize="120" color="#ffffff" />
          </Layer>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Present from="example_scene" />
</Graph>
```

`Graph -> Scene -> Timeline -> Track -> Sequence -> Layer` is the canonical
authoring grammar. Keeping one structure makes scripts easier for parsers, UI
editors, humans, and other LLMs to modify safely.

For true 3D, keep the same Scene timeline and place the 3D island inside its
own track:

```xml
<Graph fps={30} duration="4s" size={[1280,720]}>
  <Assets>
    <ModelAsset id="product_model" src="assets/product.glb" />
  </Assets>

  <Scene id="product_scene">
    <Timeline>
      <Track id="product_3d" space="3d" compositeOrder="20">
        <Sequence from="0s" duration="4s" out="hold">
          <CompositeGroup id="product_island" space="3d" depth="true">
            <Camera3D position={[0,0,6]} target={[0,0,0]} fov="35" />
            <Model asset="product_model"
                   exposure="1.4"
                   rotationY={curve("0:-25:linear, 4:25:ease_in_out")} />
          </CompositeGroup>
        </Sequence>
      </Track>
    </Timeline>
  </Scene>

  <Present from="product_scene" />
</Graph>
```

`ModelAsset src` accepts a local path or an absolute `https://` URL. Prefer a
stable published URL for small universal examples that must render unchanged
in browser and Desktop hosts. Browser hosts preload URL bytes; native hosts
fetch and cache them. Prefer a self-contained `.glb`, and do not rewrite a
portable URL to a machine-specific absolute path.

### Semantic GLB environments

Use `<Environment>` when a static GLB is the stage rather than a character or
product. It uses the existing Scene 3D model renderer, but adds typed surfaces
and anchors that an agent can reference instead of repeatedly guessing raw
coordinates:

```xml
<CompositeGroup id="stage" space="3d" depth="true">
  <Environment id="roof" asset="roof_asset" static="true"
               collision="mesh" up="+Y" forward="+Z"
               unitScale="1" scaleMode="normalize_height" scale="8">
    <Surface id="roof_floor" kind="ground" space="asset"
             normal={[0,1,0]} centroid={[0,2.4,0]}
             boundsMin={[-4,2.35,-3]} boundsMax={[4,2.45,3]} />
    <Anchor id="takeoff" surface="roof_floor" uv={[0.2,0.5]}
            offset={[0,0,0]} />
    <Anchor id="contact" surface="roof_floor" uv={[0.5,0.5]}
            offset={[0,1.2,0]} />
    <Anchor id="landing" surface="roof_floor" uv={[0.8,0.5]}
            offset={[0,0,0]} />
    <Anchor id="wide_camera" node="ML_CameraWide" offset={[0,0,0]} />
  </Environment>

  <EnvironmentDebug surfaces="true" anchors="true"
                    actionPath="true" cameras="true" />

  <Model id="runner" asset="runner_asset" profile="runner_profile"
         collision="kinematic" position="@takeoff" />
  <Camera3D id="wide" position="@wide_camera"
            target="@runner" up={[0,1,0]}
            horizonLock="true" roll="0" fov="38" />
</CompositeGroup>

<ApplyAction target="runner" action="vault"
             takeoff="takeoff" contact="contact" landing="landing"
             destination="landing"
             rootMotion="match_target" face="landing"
             ground="roof_floor" groundOffset="-0.58"
             footLock="auto" />
```

`Model collision="kinematic"` enables the deterministic Scene character
controller. MotionLoom samples the current retargeted humanoid pose, auto-fits
capsules to torso and limb chains, resolves them against every
`Environment collision="mesh"`, then applies contact correction to hands and
feet through two-bone IK when the active action uses `footLock="auto"`. The
solve uses a fixed iteration count, so preview, export and random-access frame
rendering agree. Omit `collision` (or use `none`) to preserve the original
animation exactly. Kinematic collision prevents penetration; it does not add
ragdoll dynamics, forces, bounce, cloth, or rigid-body simulation.

Inspect an unfamiliar GLB before writing surfaces or cameras. Native tools can
call `inspect_glb_environment_path()` or the `inspect_glb_environment` example;
browser hosts call `motionloom_inspect_glb_environment_json(asset, bytes)`.
The report applies GLB node transforms, lists asset bounds, detects upward
walkable triangle regions, proposes coordinate metadata, semantic surfaces and
anchors, and includes confidence plus repair recommendations. Copy generated
`space="asset"` surface measurements together; do not mix them with Scene-world
coordinates.

Prefer GLB empty nodes named `ML_Ground`, `ML_Takeoff`, `ML_Landing`,
`ML_Contact`, and `ML_CameraWide`. If they are absent, use explicit reviewed
`height`/`offset` fallbacks and keep the inspector warning. Never treat the mesh
lower bound as a guaranteed walkable floor: it may be a basement, hull, wheel,
or decorative object.

Environment `scale` follows the 3D renderer's normalized model-height unit;
`unitScale` is an additional declared asset-unit multiplier. Asset-space
Surface bounds and centroids are normalized through the same model bounds,
rotation, scale, and GLB node transforms as the rendered geometry. Surface UV
coordinates are normalized `[0,1]` values across those bounds. Explicit Anchor
offsets are Scene-world units. Runtime grounding raycasts the transformed
walkable triangles and uses the declared Surface as a semantic filter and
fallback. `groundOffset` is the target model's root-to-sole offset; it must not
be baked into the environment surface height.

For imported actions, put reusable timing semantics in the Action rather than
guessing percentages in each scene:

```xml
<Action id="vault" source="vault_clip" clip="Jump Over" duration="3s">
  <Marker id="takeoff" time="0.62s" role="takeoff" />
  <Marker id="contact" time="1.42s" role="contact" />
  <Marker id="landing" time="2.34s" role="landing" />
</Action>
```

`ApplyAction.takeoff`, `contact`, and `landing` bind those semantic phases to
Scene anchors. The renderer follows the marker timing, preserves airborne
height between takeoff and landing, and resumes surface grounding after
landing. Existing Actions without markers and destination-only ApplyAction
remain valid.

`Track space="world"` is a valid 2D Scene coordinate mode.

Text layout always uses logical Scene coordinates. Glyph rasterization is
resolution-aware by default: omit `renderScale` or use `renderScale="auto"` so
the renderer follows the final output transform with 2x anti-alias sampling
headroom, without changing wrapping, alignment, or position. Use an explicit
value such as `renderScale="4x"` only as a quality/performance override; it is
normally unnecessary.

Use `Model exposure="1"` as the neutral value. Raise it for dark product GLBs
when their embedded textures need more visibility under MotionLoom lighting.

### Skinned GLB character animation

Inspect the GLB before authoring character motion. A model with a skin but no
embedded animation clips needs `ModelProfile` + `Action`; moving the whole
`Model` only produces rigid-body motion.

Use `inspect_glb_skeleton_path()` in native hosts or
`motionloom_inspect_glb_skeleton_json(assetLabel, bytes)` in WASM before
guessing joint axes. The versioned JSON report contains:

- raw-joint to `humanoid_v1` mapping proposals and alternatives;
- detected rest pose and per-arm rest calibration;
- semantic `forward`, `side`, `bend`, `twist`, and `turn` axis/sign proposals;
- confidence per mapping and axis, actionable diagnostics, and
  `manualReviewRequired`;
- a complete proposed `<ModelProfile>` DSL block.

Treat the generated profile as a proposal. For every low-confidence axis,
preview a small semantic `+20` action and reverse that one binding's sign if
the visible motion is opposite. Never infer that left and right raw axes share
the same sign.

Map raw GLB joint names to canonical humanoid names once, then author reusable
actions against the canonical names:

```xml
<ModelProfile id="girl_profile" kind="3d" model="girl_asset"
              preset="humanoid_v1">
  <Retarget preset="humanoid_v1">
    <Map from="Right arm_68" to="upper_arm_r" />
    <Map from="Right elbow_67" to="forearm_r" />
    <Map from="Right wrist_64" to="hand_r" />
  </Retarget>
  <BoneAxisMap>
    <Axis bone="upper_arm_r" forward="rotationZ:-1"
          side="rotationX:1" twist="rotationY:1" restSide="-55.72" />
    <Axis bone="forearm_r" bend="rotationZ:-1" twist="rotationY:1" />
  </BoneAxisMap>
</ModelProfile>

<Action id="wave" skeleton="humanoid_v1" duration="1s">
  <Pose t="0s">
    <Bone id="upper_arm_r" forward="0" />
  </Pose>
  <Pose t="1s">
    <Bone id="upper_arm_r" forward="70" />
    <Bone id="forearm_r" bend="45" />
  </Pose>
</Action>

<ApplyAction target="girl" action="wave" at="2s"
             blendIn="0.2s" blendOut="0.2s"
             weight="1" mask="right_arm" />
```

The model in the 3D island must opt into that profile:

```xml
<Model id="girl" asset="girl_asset" profile="girl_profile" />
```

For an editor-authored bone channel, address the model and a canonical bone:

```xml
<AnimationTarget node="girl" property="bones.forearm_r.rotationZ">
  <Key time="0s" value="0" />
  <Key time="0.6s" value="-70" ease="ease_in_out" />
</AnimationTarget>
```

Supported bone components are `x`, `y`, `z`, `rotationX`, `rotationY`,
`rotationZ`, `rotation`, and `scale`. Prefer canonical names from the profile;
raw GLB names make actions model-specific. `AnimationTarget` bone components
remain explicit raw channels; portable `Action` poses should prefer semantic
fields with a reviewed `BoneAxisMap`.

Use analytic two-bone IK for hands and feet that must reach a target:

```xml
<Action id="reach" skeleton="humanoid_v1" duration="1s">
  <Pose t="0s">
    <Bone id="upper_arm_r" forward="0" />
  </Pose>
  <IK root="upper_arm_r" mid="forearm_r" end="hand_r"
      targetX="0.7" targetY="1.1" targetZ="0"
      poleX="0.4" poleY="0.9" poleZ="0"
      plane="xy" weight="1" />
</Action>
```

`plane` accepts `xy`, `xz`, or `yz`. `mask` accepts canonical bone ids and
common groups such as `upper_body`, `lower_body`, `left_arm`, `right_arm`,
`left_leg`, and `right_leg`. Set `mode="additive"` for an overlay action;
the default `override` replaces earlier values on the same bones. Use
`blendIn`, `blendOut`, `weight`, `speed`, and `loop` to layer actions without
hard cuts.

When a GLB contains named animation clips, one or more `<Play>` children can
crossfade them in source order. Masks allow a walk cycle and an upper-body
gesture to share the same model:

```xml
<Model id="girl" asset="girl_asset" profile="girl_profile">
  <Play clip="Walk" loop="true" weight="1" mask="lower_body" />
  <Play clip="Wave" loop="true" weight="0.65"
        blendIn="0.2s" mask="upper_body" />
</Model>
```

For a GLB with embedded clips, `<Model><Play clip="Idle" loop="true"
speed="1" /></Model>` declares clip playback. Do not invent a clip name: if
inspection reports no clips, use Actions or IK instead.

### Reusable external humanoid motion and multi-model staging

Use `AnimationAsset` only as a low-level raw clip container. Wrap it in an
executable `Action`; `ApplyAction.action` always references the `Action` id and
never an `AnimationAsset` id. The target `Model` supplies geometry, materials,
and its mapped skeleton:

```xml
<Assets>
  <ModelAsset id="model_a" src="character-a.glb" />
  <AnimationAsset id="sneak_walk_source" src="motions/sneak-walk.glb" />
</Assets>

<Action id="sneak_walk"
        source="sneak_walk_source"
        sourceProfile="mixamo_humanoid"
        clip="Sneak Walk"
        skeleton="humanoid_v1" />

<Model id="character_a" asset="model_a"
       profile="motionloom_humanoid_v1" position="@start" />
<Anchor id="start" relativeTo="character_b"
        offset={[0,0,-4.2]} space="local" />
<Anchor id="contact" relativeTo="character_b"
        offset={[0,0,-0.72]} space="local" />

<ApplyAction target="character_a" action="sneak_walk"
             at="0s" duration="3.2s"
             rootMotion="match_target" destination="contact"
             face="character_b" />
```

`sourceProfile="mixamo_humanoid"` canonicalizes common Mixamo joint names on
the motion source. `motionloom_humanoid_v1` is the clean built-in target profile;
custom `ModelProfile` mappings remain available for non-standard GLBs.
`rootMotion="match_target"` follows the named Anchor over the authored action
duration and `face` turns the model toward another model. Existing
`rootMotion="none"`, `in_place`, and `clip` values remain valid.

Use a root-level `Constraint` for a temporary cross-model hand or foot contact:

```xml
<Constraint kind="position"
            source="character_a.hand_r"
            target="character_b.shoulder_r"
            from="3.58s" to="4.05s"
            solver="two_bone_ik" weight="1" />
```

The endpoints use `model-id.canonical-bone`. The first implementation solves
canonical hand and foot chains and evaluates both skinned poses in the same
frame. Keep the contact window short and use a weight below `1` when the source
clip already approaches the target naturally.

Multiple actions may share `syncGroup` and `syncMarker`. Members of one group
must use the same `at` time and marker, which makes a contact cut deterministic.
The marker is currently a semantic synchronization label; it does not discover
or retime proprietary marker metadata inside arbitrary source files.

To cut among existing `Camera3D` nodes, animate the containing Scene rather
than introducing a second camera-sequence grammar:

```xml
<AnimationTarget node="FightScene" property="activeCamera">
  <Key time="0s" value="camera_feet" />
  <Key time="3.2s" value="camera_medium" />
  <Key time="4.15s" value="camera_overhead" />
</AnimationTarget>
```

`activeCamera` is discrete and uses step interpolation. Camera IDs therefore
switch exactly at key times; position, target, and FOV animation continue to
use the normal `Camera3D` properties.

## Canonical Process Structure

```xml
<Graph fps={30} size={[800,450]} renderSize={[800,450]}>
  <Process id="brightness_process">
    <Input id="clip0" type="video" from="input:clip0" />
    <Tex id="src" fmt="rgba16f" from="clip0" />
    <Tex id="out" fmt="rgba16f" size={[800,450]} />
    <Pass id="fx_brightness" kind="compute" effect="brightness"
          in={["src"]} out={["out"]}
          params={{ brightness: "0.3" }} />
  </Process>

  <Present from="brightness_process" />
</Graph>
```

Use explicit textures for multi-pass processing:

- `rgba8`: lightweight/final color where HDR precision is unnecessary.
- `rgba16f`: HDR and intermediate color processing.
- `r16f`: one-channel masks, depth, or scalar data.

Do not silently change texture formats between connected passes.

## Background Rule

If the full-frame background is static, use `<Background color="..."/>` only.
Do not add a full-canvas `<Rect>` that duplicates the same background color.

Only use a full-canvas `<Rect>` when the background needs timeline animation,
blend mode, opacity animation, masking, clipping, or scene-local layering.

## IDs and References

- Give every animated, referenced, interactive, or UI-editable node a stable,
  unique `snake_case` ID.
- Use semantic IDs such as `right_forearm`, `send_button`, and `title_reveal`.
- Every `from`, `in`, `out`, `target`, `attachTo`, `rig`, `skeleton`, and mask
  reference must resolve.
- Never depend on generated names such as `Group#01`; they are unstable across
  edits and render backends.
- Group related artwork semantically so one transform controls the intended
  object rather than many unrelated paths.

## Animation Rules

- Use `curve(...)` for concise deterministic numeric animation.
- Curve points must contain numeric values only:
  `curve("0:0:linear, 1:100:ease_out")`.
- Keep procedural expressions such as `sin(...)` or `random(...)` outside curve
  keyframe values.
- Use `<AnimationTarget>` and `<Key>` when animation must be editable as explicit
  timeline keyframes by the UI.
- Do not drive the same node property with both `curve(...)` and
  `<AnimationTarget>`.
- Prefer `time="1.5s"` keys when timing should survive FPS changes; use
  `frame="45"` when exact frame identity is intentional.
- Do not animate string attributes such as `Text.value` or `Path.d` with numeric
  curves. Use supported path morphing, transforms, opacity, trim, or masks.
- For typing and reveal effects, prefer one complete text node revealed by a
  real mask. Avoid stacking many text snapshots with one-frame opacity swaps.

## Rigging and Deformation

- Use nested `Group` transforms for simple parent-child motion.
- Use `Skeleton`, `Bone`, and `Action` for reusable FK animation.
- Use IK for target-driven limb or joint-chain solving.
- Use `Puppet` with `Pin` and auto mesh for ordinary image deformation.
- Choose one Puppet Warp target mode:
  - Use `target="GROUP_ID"` for a semantic arm, eye, hair lock, or other
    isolated part.
  - Use `target="@layer" capture="before"` when the current Layer should act as
    one universal surface and no Group id should be required.
- Place an `@layer` Puppet directly inside the Layer after every node that must
  deform. Put guides, labels, and other undeformed overlays after it.
- For `target="@layer"`, either omit `capture` (it defaults to `before`) or
  write `capture="before"` explicitly. Never invent other `@...` selectors.
- Use `PuppetWarp solver="bones"` for an arm or leg that must preserve rigid
  upper/lower segment lengths. Author three pins with `role="anchor"`,
  `role="joint"`, and `role="control"`, then bind topology vertices with
  `bone="upper|forearm|hand|joint"`. Use `bone="fixed"` for the static side of
  a shoulder or hip seam.
- When the limb is not a uniform-width capsule, add exactly one closed
  `<LimbEnvelope d="M ... Z" alphaClip="true" handFrom="control_pin_id" />`.
  Keep Anchor, Joint, Control, and envelope coordinates in the same space.
  Prefer this over guessing a wide rectangular `MeshTopology`; use explicit
  topology only when individual vertex weights are required.
- When the three functional parts can be outlined separately, prefer three
  closed `<LimbRegion>` nodes with `role="anchor"`, `role="joint"`, and
  `role="control"`. The anchor region follows the upper bone, the joint region
  blends both bones, and the control region follows the lower bone/hand.
  Slight overlap at the seams prevents gaps. Do not combine `LimbRegion` with
  a legacy `LimbEnvelope` or generated `MeshTopology` in the same warp.
- Use `PuppetWarp solver="chain"` for one non-branching tail, hair lock, rope,
  or tentacle. Give every pin an id, keep the root fixed, and link every later
  pin with `parent="PREVIOUS_PIN_ID"`. Add a sibling `SpringChain` whose target
  is the PuppetWarp id when follow-through is required.
- Do not rewrite surface-pin rigs as chain rigs or three-point bone rigs. The
  `soft`, `bones`, and `chain` solvers are separate authored behaviors.
- For a hard-bending elbow or knee, duplicate the seam vertices for the two
  adjacent bones and bridge them with triangles. Set `sampleX/sampleY` on the
  bridge vertices to interior skin or clothing coordinates, preventing the
  joint completion patch from sampling a transparent or background edge.
- Set `preserveOutside="true"` only when a local limb mesh targets a larger
  character Group; leave it false for already isolated artwork.
- Add `MeshTopology` only when advanced users need manual vertices, triangles,
  edges, or regions. Do not require topology in normal examples.

## Effects and Resources

- Use only documented effect names and parameters.
- Keep pass dependencies explicit through texture IDs.
- Keep `<Present ... />` as the final direct child of `<Graph>`.
- Define reusable fonts, gradients, brushes, masks, and textures in their
  documented scopes instead of duplicating them across nodes.
- Prefer existing primitives and features over approximating them with many
  unrelated nodes.

## Parametric Components

Declare typed inputs inside a `Component`, then bind them from `Use.params`:

```xml
<Component id="data_bar">
  <Param name="height" type="number" default="120" />
  <Param name="paint" type="color" default="#22D3EE" />
  <Param name="enabled" type="boolean" default="true" />
  <Param name="align" type="enum" values={["left","center"]} default="left" />
  <Derived name="halfHeight" value={param("height") * 0.5} />
  <Rect x="0" y="0" width="80" height={param("height")}
        fill={param("paint")} opacity={param("enabled")} />
  <Slot name="label">
    <Text x="0" y={derived("halfHeight")} value="DEFAULT" />
  </Slot>
</Component>

<Use ref="data_bar" params={{ height: "240", paint: "#4ADE80" }}>
  <Fill slot="label">
    <Text x="0" y="120" value="CUSTOM" />
  </Fill>
</Use>
```

- Supported parameter types are `number`, `color`, `text`, `path`, `boolean`,
  and `enum`. Enum parameters require `values={[...]}`.
- Number, color, path, boolean, and enum bindings are validated when parsed.
- Boolean values lower to numeric `1` or `0`, making them suitable for
  opacity-like numeric attributes.
- Parameter binding replaces `param("name")` in component attributes.
- `Derived` values resolve in declaration order and can reference parameters or
  earlier derived values with `derived("name")`.
- `Slot` supplies default scene children. A block `Use` can replace them with
  one matching `Fill`.
- Every parameter needs a non-empty default or an explicit `Use.params` value.
- Parameterized uses currently require `blend="normal"`.
- Do not invent arbitrary component props; declare every accepted value with
  `Param`.

## Seeded Repeat Variation

Use deterministic scatter when repeated artwork should not form a linear grid:

```xml
<Repeat count="80" distribution="scatter"
        bounds={[100,120,900,500]} seed="61001">
  <Variants choose="weighted" seed="42">
    <Circle x="0" y="0" radius="6" color="#67E8F9" weight="5" />
    <Rect x="-5" y="-5" width="10" height="10" color="#67E8F9" weight="2" />
  </Variants>
  <Vary property="color" values={["#67E8F9","#FDE047","#F472B6"]} />
  <Vary property="scale" range={[0.5,1.5]} />
  <Vary property="rotation" range={[-20,20]} />
</Repeat>
```

- `Variants choose="weighted"` accepts one or more direct scene children with a
  positive literal `weight`.
- `Vary` accepts exactly one literal `values` list or numeric `range`.
- `x`, `y`, `rotation`, `scale`, and `opacity` vary instance transforms.
  Other properties update matching attributes inside the selected artwork;
  `color` updates both primitive `color` and path `fill` fields.
- Advanced variation works with `linear`, `grid`, and `scatter` distribution.
- Advanced Repeat count, seed, bounds, weights, and ranges are literal values.
- `seed` makes all generated instance transforms reproducible.
- `Variants.seed` controls choices and `Vary` independently from the Repeat
  seed used for scatter placement.

## Declarative Scene Layout

Use `Layout` when children share a regular row, column, or grid structure:

```xml
<Layout id="cards" mode="grid" x="80" y="120"
        width="840" height="520" columns="3"
        itemWidth="240" itemHeight="160"
        padding={[40,60]} rowGap="24" columnGap="32"
        align="center" justify="spaceBetween">
  <Group layoutSpan="2">...</Group>
  <Group>...</Group>
  <Group>...</Group>
</Layout>
```

- Supported modes are `row`, `column`, and `grid`.
- `itemWidth`, `itemHeight`, `gap`, `rowGap`, and `columnGap` define placement.
- `columns` applies to grid mode.
- `padding` accepts one value, vertical/horizontal values, or top/right/bottom/left.
- `align` accepts `start`, `center`, or `end`.
- `justify` accepts `start`, `center`, `end`, `spaceBetween`, `spaceAround`, or
  `spaceEvenly`. Explicit `width` or `height` provides the distributable space.
- `layoutSpan` reserves multiple cells along the main axis or within a grid.
- Layout accepts normal group transforms such as `x`, `y`, `rotation`,
  `scale`, and `opacity`.
- Wrap a Layout in an animated `Group` when its entrance or exit needs a curve.

## Stage 3 Construction Order

1. Classify the request as Scene, Process, or Scene plus Process. Use Scene for
   every visual composition, including true 3D.
2. Use the Stage 1 retrieval result; prefer a focused core example for grammar
   and a showcase for composition patterns.
3. Copy its structural skeleton, not its decorative content.
4. Add stable semantic IDs before animation or references.
5. Build the static composition first.
6. Add animation, masks, rigs, or effects one system at a time.
7. Verify references, durations, texture formats, and presentation output.
8. Continue to Stage 4 analysis; do not render an unreviewed revision.

## Final Checklist

- The graph uses a documented canonical hierarchy.
- All IDs are unique and all references resolve.
- No duplicate attributes exist on a node.
- Curves contain numeric keyframe values only.
- No property has two competing animation sources.
- Static backgrounds do not include duplicate full-frame rectangles.
- Texture formats and pass inputs/outputs match.
- The graph duration covers every sequence and animation.
- `<Present>` is the last direct child of `<Graph>`.
- The authoring report is `clean`, or every remaining warning is intentional.
- Opening, middle, and final render evidence match the requested visual intent.
- The report and render evidence correspond to the same DSL revision.

## Sources of Truth

When guidance differs, use this order:

1. Current parser, schema, and tests.
2. This guide for DSL and agent behavior.
3. `PUBLIC_API.md` and ACP documentation for host integration.
4. Current `motionloom-example/core` examples.
5. Showcase examples for composition ideas, not minimal grammar.
