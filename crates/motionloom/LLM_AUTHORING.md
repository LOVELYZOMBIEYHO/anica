# MotionLoom LLM Authoring Guide

Use this guide when generating or editing MotionLoom DSL. Prefer valid,
predictable, editable, and renderable output over the shortest possible script.

## Choose One Graph Family

- **Scene graph**: vector graphics, text, animation, characters, cameras, masks,
  rigs, and composition.
- **Process graph**: media input, textures, compute effects, and multi-pass image
  processing.
- **World graph**: 3D/world content where the documented world components are
  required.
- Do not mix graph families unless the composition genuinely requires it and a
  documented example demonstrates the connection.

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

## Reliable Generation Workflow

1. Classify the request as Scene, Process, or World.
2. Find the nearest working example in `motionloom-example/core`.
3. Copy its structural skeleton, not its decorative content.
4. Add stable semantic IDs before animation or references.
5. Build the static composition first.
6. Add animation, masks, rigs, or effects one system at a time.
7. Verify all references, durations, texture formats, and presentation output.
8. Render a representative frame and test the GPU path when relevant.

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
- The script has been parsed or rendered with a current MotionLoom tool.

## Sources of Truth

When guidance differs, use this order:

1. Current parser, schema, and tests.
2. This guide and `README.md`.
3. `PUBLIC_API.md` and ACP documentation.
4. Current `motionloom-example/core` examples.
5. Showcase examples for composition ideas, not minimal grammar.
