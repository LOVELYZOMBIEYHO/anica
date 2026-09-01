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
Call `motionloom_dsl_schema_json()` when introducing syntax not demonstrated
by the selected examples. Browser agents use the WASM export with the same
name; native agents use the Rust API. It returns the complete registered tag
and attribute catalog, validation coverage, expression support, and
AnimationTarget capability without requiring a JSON-authored MotionLoom file.

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

Use `PrimitiveAsset` for reusable engine-generated geometry. It is an asset
resource and only enters the Scene through `Model`:

```xml
<Assets>
  <ImageAsset id="stone_color" src="stone.jpg" colorSpace="srgb" />
  <MaterialAsset id="stone" shading="pbr" baseColorTexture="stone_color"
                 metallic="0" roughness="0.84" mapping="triplanar"
                 textureScale={[0.3,0.3]} variationAmount={[0.2,0.15]} />
  <PrimitiveAsset id="ball" shape="sphere" radius="0.5"
                  segments="32" color="#50E3E6" />
  <PrimitiveAsset id="ground" shape="plane" size={[12,8]}
                  color="#131B2E" collision="solid" />
  <PrimitiveAsset id="stone_step" shape="box" size={[4,0.3,0.9]}
                  material="stone" bevelRadius="0.025" bevelSegments="3"
                  collision="solid" collider="box" />
  <PrimitiveAsset id="trigger" shape="sphere" radius="1"
                  collision="sensor" collider="box" colliderSize={[2,2,2]} />
  <CompoundAsset id="two_steps">
    <Instance id="lower" asset="ground" position={[0,0,0]} />
    <Instance id="upper" asset="ground" position={[0,0.3,-1]} scale="0.8" />
  </CompoundAsset>
</Assets>

<Model id="ball_model" asset="ball" position={[0,4,0]} />
<RigidBody id="ball_body" target="ball_model" dimension="3d"
           type="dynamic" shape="auto" mass="1" />
```

V1 shapes are `box`, `sphere`, `plane`, `cylinder`, `cone`, and `wedge`.
Dimensions must be positive, tessellation attributes accept 3 through 256,
and colors use `#RRGGBB` or `#RRGGBBAA`. Dynamic planes are invalid; use a
static plane or a thin box. The removed `motionloom:box` source shorthand is
not accepted.

Primitive collision defaults to `collision="none"`. Use `solid` for blocking
geometry or `sensor` for non-blocking contact metadata. When collision is enabled,
`collider` defaults to `auto` and follows the visual shape; an explicit `box`,
`sphere`, `plane`, `cylinder`, `cone`, `convex`, or `mesh` may intentionally
differ from it. Collider dimensions can be overridden with `colliderSize`,
`colliderRadius`, `colliderHeight`, `colliderScale`, `colliderOffset`,
`colliderRotation`, and `colliderMargin`. `CompoundAsset` V1 composes only
`PrimitiveAsset` instances, preserving each child's visual and collision data.
Use `MaterialAsset`, not Scene-local `Defs/Material` or `MaterialBinding`, for
physical primitive surfaces. Supported PBR slots are `baseColorTexture`,
`metallicRoughnessTexture`, `normalTexture`, `occlusionTexture`, and
`emissiveTexture`; every slot references an `ImageAsset`. `mapping` accepts
`uv`, `box`, or `triplanar`. Box `bevelRadius`/`bevelSegments` affect only the
render mesh. Optional `materialSeed` on a primitive, compound, or instance
changes UV sampling deterministically and never changes geometry or collision.
It also does not create another texture or mesh resource: instance seeds are
shader parameters, while geometry, decoded ImageAsset pixels, and GPU textures
remain shared. Prefer one ImageAsset/MaterialAsset reused by many instances;
duplicating image declarations with different source strings prevents source
identity sharing.

Use `VegetationAsset` for bounded procedural plants, then place it with normal
`Model` nodes. V1 kinds are `tree`, `shrub`, `grass`, `flower`, `fern`, and
`deadwood`:

```xml
<VegetationAsset id="oak" kind="tree" height="7"
  trunkMaterial="bark" foliageMaterial="leaf_atlas"
  density="24" branchLevels="3" seed="20"
  lod="auto" wind="true" collision="solid" />
<VegetationAsset id="ferns" kind="fern" height="0.8"
  material="fern_atlas" density="18" seed="77" lod="auto" wind="true" />
```

Trees and shrubs require `trunkMaterial` and `foliageMaterial`. Grass and fern
require `material`; flower requires `material` and may set `stemMaterial`;
deadwood requires `trunkMaterial`. Only tree and deadwood accept
`collision="solid"`. `density` controls geometry within one asset; do not use
it as a world scatter count. Reuse one asset with multiple Model nodes instead
of duplicating declarations. Keep generated leaf, grass, flower, and fern
atlases transparent and reference them through alpha-mask PBR MaterialAssets.
Existing asset kinds require no migration; VegetationAsset is opt-in.

For architectural glass, water, or transparent plastic, prefer physical
transmission over lowering base-colour alpha:

```xml
<MaterialAsset id="glass" shading="pbr" baseColor="#E8F7FA"
  roughness="0.08" specular="1" transmission="0.94" ior="1.52"
  thickness="0.012" attenuationColor="#B7DDE2" attenuationDistance="6"
  depthWrite="auto" doubleSided="true" />
```

Keep `depthWrite="auto"` unless a specialist effect deliberately owns depth.
The renderer draws opaque/mask geometry first, then sorts blend/transmissive
surfaces far-to-near without depth writes. `sortPriority` is an integer expert
override for unavoidable overlapping transparent meshes. Material transmission
and PrimitiveAsset collision remain independent.

### Cinematic 3D lighting and HDRI/IBL

Keep lighting inside the same `CompositeGroup space="3d"` as the camera and
models. `EnvironmentLight` accepts an `ImageAsset` containing Radiance `.hdr`,
OpenEXR `.exr`, or an LDR image. HDR/EXR inputs remain linear and are uploaded
as a floating-point mip chain; LDR inputs are converted from display gamma.

```xml
<Assets>
  <ImageAsset id="courtyard_hdri" src="assets/courtyard.hdr" />
</Assets>

<CompositeGroup id="lit_stage" space="3d" depth="true" format="rgba16f">
  <EnvironmentLight id="ibl" asset="courtyard_hdri"
                    mapping="equirectangular" rotationY="25"
                    intensity="1" visible="true"
                    backgroundIntensity="0.7" backgroundBlur="0.1"
                    diffuseIntensity="0.85" specularIntensity="1.25" />
  <DirectionalLight id="sun" direction={[-0.4,-1,-0.3]}
                    color="#FFE3BC" intensity="3.5"
                    castShadow="true" shadowStrength="0.9" />
  <PointLight id="lamp" position={[2,2,1]}
              color="#FF8A58" intensity="18" range="8" />
  <SpotLight id="rim" position={[-2,3,-1]} direction={[0.5,-0.7,0.5]}
             color="#78D4FF" intensity="24" range="10"
             innerCone="18" outerCone="34" />
  <RectAreaLight id="softbox" position={[0,4,3]}
                 direction={[0,-0.8,-0.6]} intensity="8"
                 width="3" height="2" />
  <AmbientOcclusion id="ao" intensity="0.65" radius="1.2" />
  <ContactShadow id="contact" intensity="0.8"
                 distance="0.35" softness="0.6" />
  <ColorManagement id="grade" toneMapping="aces" exposure="1"
                   whiteBalance="5600" contrast="1.06" />
  <AtmosphereFog id="mist" mode="height" color="#BFD9C8"
                 density="0.018" start="3" end="34"
                 baseHeight="0.35" heightFalloff="0.12"
                 scattering="0.1" affectSky="true" />
  <Camera3D position={[6,4,8]} target={[0,1,0]} fov="38"
            depthOfField="true" focusTarget="@hero_model"
            focalLength="50" fStop="2.8" maxBlur="8" />
  <Model id="hero_model" asset="hero" castShadow="true" receiveShadow="true" />
</CompositeGroup>
```

`backgroundIntensity` changes only the visible environment;
`diffuseIntensity` and `specularIntensity` control IBL on materials. Rough
surfaces automatically sample blurrier environment mip levels. The first
shadow-casting authored light supplies the primary filtered shadow map; do not
mark every light as a shadow caster merely because it is bright.

Use `toneMapping="aces"` for cinematic HDR output, `reinhard` for a softer
technical preview, and `none` only when the host owns the final HDR transform.
Lighting and grading values such as `rotationY`, `intensity`, `exposure`,
`whiteBalance`, and `contrast` are registered `AnimationTarget` channels.
Query the animation property schema before writing keys rather than inventing
properties.

`AtmosphereFog` is world-owned: use `linear`, `exp`, or `height` mode and keep
density low for bright outdoor scenes. Omit `boundsMin` and `boundsMax` for the
legacy global medium. Provide both to confine fog to a world-space box; use
`edgeFeather` to soften its boundary. A camera outside the box can still see
fog along rays that pass through it, so indoor-to-outdoor shots do not require
a screen mask or a timeline switch.

```xml
<AtmosphereFog mode="exp" color="#668FA8" density="0.045"
               boundsMin={[-20,-2,-30]} boundsMax={[20,20,-3.2]}
               edgeFeather="1.5" affectSky="true" />
```

This is an additive `0.1.x` extension: existing tags without bounds keep their
previous result. Before, local fog required an approximate compositing mask;
now the optional bounds provide depth-aware world-space confinement.

Depth of field is camera-owned and is a real depth-buffer post pass.
`focusTarget` accepts an Anchor or Model reference; otherwise the camera target
distance is used. Omit `depthOfField` to preserve the zero-cost sharp render
path.

### Deterministic humanoid traversal

For a simple finite flat stage, declare a procedural box Surface and opt a
kinematic humanoid into Scene gravity:

```xml
<CompositeGroup id="stage" space="3d" depth="true">
  <Physics gravity={[0,-9.81,0]} fixedStep="1/120s" iterations="4" />
  <Surface id="floor" kind="ground" collider="box"
           center={[0,-0.1,0]} size={[20,0.2,20]} color="#202838" />
  <Model id="hero" asset="hero_asset" profile="hero_profile"
         position={[0,5,0]} collision="kinematic"
         gravity="scene" ground="floor" />
</CompositeGroup>
```

The Surface is both visible GPU geometry and finite collision geometry. Gravity
is evaluated from absolute time with a fixed step, so frame 60 is identical
whether frames 0–59 were rendered first or not. A character outside the
Surface's X/Z bounds continues falling. Omit `<Physics>` and Model `gravity` to
retain existing authored placement exactly.

### Unified rigid bodies

Use only `<RigidBody>`. Both `dimension` and `type` are required; the removed
`RigidBody2D` spelling is invalid and must not be generated.

For 3D Models, `scaleMode="none"` is the default and preserves the glTF local
origin and authored units. Use this for rigid bodies so the visible mesh and
collider share one Transform3D. Only write `scaleMode="normalize_height"` for
characters or imported product models whose `scale` intentionally means a
target world height and whose origin should be bottom-centred.

```xml
<!-- A 2D body binds to a Group in a Layer. -->
<RigidBody id="card_body" target="card"
           dimension="2d" type="dynamic" shape="box"
           size={[180,96]} velocity={[80,0]} gravity={[0,180]} />

<!-- A 3D body binds to a Model in the same CompositeGroup. -->
<Physics gravity={[0,-9.81,0]} fixedStep="1/120s" iterations="4" />
<RigidBody id="crate_body" target="crate"
           dimension="3d" type="dynamic" shape="box"
           size={[1,1,1]} mass="2" friction="0.6"
           rollingFriction="0.08" restitution="0.2"
           restitutionThreshold="0.5" sleep="true"
           sleepLinearThreshold="0.015"
           sleepAngularThreshold="0.025" sleepTime="0.5" />
```

`static` bodies do not move, `dynamic` bodies integrate gravity and collide,
and `kinematic` bodies follow authored transforms while acting as colliders.
Use `size` for box/auto/convex proxy dimensions, `radius` for sphere/capsule,
and `height` for capsule/cylinder. 2D vectors contain two numbers; 3D vectors
contain three. A dynamic concave `shape="mesh"` is rejected; use
`shape="convexHull"` or a compound set of simple bodies.

`gravity` on a 2D body is local to that body. A 3D body must not declare
`gravity`; all 3D bodies use the containing Scene's `<Physics gravity>`.
This prevents LLM-authored objects in one Scene from silently using different
world forces.

For 3D bodies, `friction` resists sliding while `rollingFriction` removes
residual angular motion at supported contacts. `restitutionThreshold` disables
bounce below the specified impact speed. Sleeping requires both linear speed
below `sleepLinearThreshold` and angular speed below `sleepAngularThreshold`
for the full `sleepTime`; do not use `sleep="false"` for ordinary props that
should eventually rest.

Do not animate a dynamic body's Model transform and expect both systems to own
it: physics owns the dynamic transform. Animate a kinematic body when authored
motion must drive collision. Existing documents without `<RigidBody>` retain
the previous renderer fast path.

Use `<PhysicsDebug colliders="true" contacts="true" sweep="true"
corrections="true" />` while diagnosing a Scene. It is the physics-focused
alias of EnvironmentDebug and does not change simulation results.

Use `collision="kinematic"` on a humanoid `<Model>` when it must move across a
mesh environment. This is a deterministic character controller rather than a
rigid-body simulation: it sweeps one stable body collider, slides at contacts,
snaps to walkable floors, and applies pose-only IK to head, feet, and hands.

For a vault, roll, jump, or climb, put semantic markers on the executable
`Action` and align them to environment anchors through `ApplyAction`:

```xml
<Action id="vault" source="vault_clip" sourceProfile="fbx_humanoid"
        skeleton="humanoid_v1" duration="3s">
  <Marker id="takeoff" role="takeoff" time="0.6s" />
  <Marker id="contact" role="contact" time="1.4s" />
  <Marker id="landing" role="landing" time="2.3s" />
</Action>

<ApplyAction target="hero" action="vault" duration="3s"
             rootMotion="match_target"
             takeoff="takeoff_anchor" contact="contact_anchor"
             landing="landing_anchor" destination="landing_anchor"
             colliderProfile="auto" footLock="auto" />
```

`colliderProfile="auto"` selects standing, airborne, rolling, and crouched
shapes from marker time. Advanced scripts may explicitly choose `standing`,
`crouched`, `airborne`, `rolling`, or `prone`; optional `safeMargin`,
`floorSnap`, `maxSlides`, and `sweepStep` tune the controller without changing
the old defaults. Use `<EnvironmentDebug colliders="true" contacts="true"
sweep="true" corrections="true" />` only while diagnosing a scene because it
emits a structured per-frame kinematic report.

### Semantic GLB environments

Use `<Environment>` when a static GLB is the stage rather than a character or
product. It uses the existing Scene 3D model renderer, but adds typed surfaces
and anchors that an agent can reference instead of repeatedly guessing raw
coordinates:

```xml
<CompositeGroup id="stage" space="3d" depth="true">
  <Environment id="roof" asset="roof_asset" static="true"
               collision="surfaces" up="+Y" forward="+Z"
               unitScale="1" scaleMode="normalize_height" scale="8">
    <Surface id="roof_floor" kind="ground" space="asset"
             collision="true" collider="plane"
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
the selected humanoid collider profile, resolves it against the enabled
Environment collision geometry, then applies contact correction when the
active action uses `footLock="auto"`. The
solve uses a fixed iteration count, so preview, export and random-access frame
rendering agree. Omit `collision` (or use `none`) to preserve the original
animation exactly. Kinematic collision prevents penetration; it does not add
ragdoll dynamics, forces, bounce, cloth, or rigid-body simulation.

Use `Environment collision="surfaces"` for detailed or scanned render assets.
Only nested surfaces with `collision="true"` participate. `collider="plane"`
creates a finite ground plane from the authored bounds; `collider="box"`
creates a coarse closed obstacle. `heightfield` and `mesh` filter source
triangles to the selected Surface bounds, with a plane fallback when no source
triangle is available. This keeps visual tessellation separate from stable
character collision. Existing `collision="mesh"` retains its original full-GLB
behavior for collision-ready assets.

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

For a reusable kneel, plant, brace, or hand-contact interval, declare
normalized clip-phase contacts inside the executable `Action`, then bind the
semantic `ground` target to a concrete Scene `Surface`, or to the Model id of
a `TerrainAsset collision="solid"`, at application time:

```xml
<Action id="repair_kneel" source="character_clips" clip="Fixing_Kneeling">
  <Contact id="left_knee_contact" effector="knee_l" target="ground"
           from="18%" to="72%" mode="lock" weight="1" />
  <Contact id="right_foot_contact" effector="foot_r" target="ground"
           from="16%" to="76%" mode="lock" weight="0.9" />
</Action>

<ApplyAction target="technician" action="repair_kneel"
             ground="ship_deck" contactCorrection="auto" />
```

Contact percentages refer to the imported GLB clip's normalized time, not the
Graph or ApplyAction duration. `contactCorrection="auto"` is opt-in and
requires both a `ground` binding and at least one `<Contact />` on the
referenced Action. Use `collision="kinematic"` on a moving humanoid Model when
solid terrain is present. The deterministic solver corrects the actor root first,
then applies only small residual two-bone IK adjustments to distal hands and
feet. Knee and elbow contacts use root correction because they are middle
joints, not two-bone IK endpoints. Actions without contacts retain their
existing behavior.

For sitting, lying, leaning, or other prop-relative support, do not resize the
prop to hide mesh penetration. Declare a semantic `ContactSurface` and bind the
Action target slot:

```xml
<ContactSurface id="bench_seat" source="bench_seat_model" kind="seat"
                plane="top" forward={[0,0,1]} bounds={[2.8,0.72]}
                margin="0.02" />
<Action id="sit" skeleton="humanoid_v1" duration="2s">
  <Pose t="0s">...</Pose>
  <Contact id="pelvis_seat" effector="pelvis" target="seat"
           from="62%" to="100%" mode="surface" weight="1" />
</Action>
<ApplyAction target="character" action="sit" contactCorrection="auto"
             contactTargets={{ seat: "bench_seat" }} />
```

Use `source` as a Scene Model id, not an asset id. Prefer `plane="top"` for a
PrimitiveAsset seat. Use explicit `position`, `normal`, and `forward` when the
support is imported geometry. Keep `ground` alongside `contactTargets` when
feet must also follow terrain. A persistent seated idle should author the seat
Contact from `0` to `1`; the solver then keeps support active at direct seeks
and Action boundaries.

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

For large Action Editor output, prefer a standalone `ActionLibrary` document
instead of pasting thousands of Pose keys into the scene:

```xml
<ActionLibrary id="performance" src="actions/performance.motionloom"
               actions={["formal_bow","kneel_down","stand_up"]} />
<ApplyAction target="character_a" action="performance.formal_bow" />
```

Use the declaration id as a namespace. Do not invent `ActionAsset`, do not put
`ActionLibrary` inside `<Assets>`, and do not reference an unlisted library
Action. The library file contains an `ActionLibrary` root with authored
`Action` children; v1 intentionally excludes AnimationAsset-backed Actions.

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
        sourceProfile="fbx_humanoid"
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

`sourceProfile="fbx_humanoid"` canonicalizes common namespaced humanoid joint names on
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

For a first-person shot, keep the complete actor and attach camera Anchors to
its canonical `head` bone. `hiddenBones` is camera-local, so a later camera
automatically sees the complete character again:

```xml
<Anchor id="eyes" relativeTo="hero" node="head"
        offset={[0,0.03,0.1]} space="local" />
<Anchor id="look" relativeTo="hero" node="head"
        offset={[0,-0.25,2.0]} space="local" />
<Camera3D id="first_person" position="@eyes" target="@look"
          hiddenBones={["hero:head"]} fov="60" />
<Camera3D id="third_person" position={[4,3,6]} target="@eyes" fov="38" />
```

Each selector is `model_id:canonical_bone`. The selected bone and its skinned
descendants are omitted only from that camera's view passes. The actor pose,
collision, animation, other cameras, and shadow-casting geometry are unchanged.
Prefer this over deleting the GLB or globally hiding a mesh when the timeline
will return to third-person or close-up coverage.

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

For deterministic world-space precipitation or atmosphere, reuse `Repeat`
inside a 3D CompositeGroup:

```xml
<Repeat id="snow" mode="volume" count="96" seed="42"
        boundsMin={[-8,0,-8]} boundsMax={[8,10,8]}
        velocity={[0,-2,0.2]} lifetime="4s"
        phase="random" respawn="random" scaleRange={[0.7,1.3]}>
  <Model asset="snowflake" castShadow="false" receiveShadow="false" />
</Repeat>
```

- `mode="volume"` only works in `CompositeGroup space="3d"`.
- It accepts exactly one self-closing `Model` template.
- `count` and `seed` are literal integers; bounds, velocity, lifetime, phase,
  respawn, and scale range control a seed-stable 3D lifecycle.
- Use separate outdoor bounds to exclude sheltered areas. Do not restore a
  dense screen-space overlay as a substitute for world depth.
- Existing 2D Repeat distribution and variation syntax is unchanged.

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
