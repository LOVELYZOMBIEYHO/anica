# Changelog

## Unreleased

- BREAKING: rename the universal explicit mesh DSL to `MeshAsset`, `Vertex`,
  and `Face`. The previous subdivision-specific tag names are rejected rather
  than treated as aliases. `MeshAsset` defaults to `subdivision="0"`, supports
  levels 0–2, and accepts `subdivisionScheme="catmullClark"`. The underlying
  polygon cage, UV interpolation, renderer, and GLB export logic are unchanged.

- BREAKING: replace flat Eye iris/lid styling fields with an optional nested
  Iris and repeatable Eyeliner components. Eye Texture now maps to the generated
  convex sclera instead of a flat whole-eye card; old fields are rejected.
  Iris supports Eye-local position, circle/ellipse/square geometry, and a geometry
  scale separate from its Texture UV transform.
  Eye position Z now moves only the eyeball and Iris; Eyeliner stays on the lid,
  while socketDepth exclusively controls the surrounding orbital skin. This
  prevents rectangular skin extrusion.

- Add parametric FaceLayout Eyebrow ribbons with optional nested Texture and
  head-material fallback. Replace S86's explicit eyebrow cards with components.

- BREAKING: remove flat FaceLayout attributes. Use explicit Eye, Nose, Mouth,
  and Ear children with unique IDs and component-local Texture bindings.
  Migrate S85/S86 sources; see FACE_COMPONENTS.md for coordinate conventions
  and current front-surface generator limits.

- Replace the eye-specific explicit cage DSL with generic polygon mesh nodes.
- Add `HeadAsset topology="facialCage|explicit"`, a versioned Rust facial-cage
  generator, reusable profiled-surface internals, cage inspection APIs, and the
  corresponding WASM export.
- Remove `EyeAsset`, `EyeVertex`, and `EyeFace` without a compatibility parser.

## Unreleased

- Added HairDefaults, HairGuide numeric overrides/root normal, and HairMirror.
  Fixed accumulated roll, curved-card normals, uneven curve sampling and tip
  closure. Existing scripts parse, but corrected hair geometry and mesh/UV hashes
  change. Rust HairGuideNode gains an optional normal field. See HAIR_CARDS.md.

- Added experimental camera-independent geometry snapshots, UV diagnostic
  images/reports and static GLB byte export. Scene model lowering is shared with
  rendering. Source DSL and its cameras remain intact; see GEOMETRY_TOOLING.md.

- Improved native/WebGPU immediate 3D materials without new DSL settings:
  semantic material mipmaps with 8x anisotropic filtering, independent material
  AO affecting indirect light, and RGBA16Float intermediate rendering through
  transparency and depth of field before the final display curve. Graph output
  size and RGBA8 Scene composition remain unchanged. See
  [immediate preview notes](IMMEDIATE_PREVIEW.md) for scope and measurements.

- Breaking DSL and Rust API change: removed the `RenderQuality` resource,
  `Scene.renderQuality`, and all resolved quality fields. Old DSL and serialized
  graph JSON are rejected. Immediate native and WASM rendering now use the Graph
  render size, a fixed 1536 shadow map, existing analytic ambient occlusion, and
  no quality-selected anti-aliasing. Use Graph `renderSize` to request larger
  output dimensions.

- Extended the existing `Repeat` tag with deterministic
  `mode="volume"` inside 3D CompositeGroups. A regular Model template can now
  populate bounded world space with seeded phase, velocity, lifetime, respawn,
  and scale variation for rain, snow, dust, embers, or debris. Existing 2D
  linear/grid/scatter Repeat scripts retain their previous defaults and output.

- Added universal transmissive PBR materials through existing `MaterialAsset`
  attributes: `transmission`, `ior`, optical `thickness`, `attenuationColor`,
  `attenuationDistance`, `depthWrite`, and `sortPriority`. The GPU renderer now
  submits opaque/mask geometry before far-to-near transparent and transmissive
  queues; automatic transparent depth writes are disabled so glass cannot hide
  later Character GLB draws. Existing materials retain their previous defaults.
  Migration is additive: existing
  `<MaterialAsset baseColor="#B7DDE255" alphaMode="blend" />` remains valid,
  while physical glass should use
  `<MaterialAsset baseColor="#E8F7FA" transmission="0.94" ior="1.52" thickness="0.012" depthWrite="auto" />`.

- Added camera-local humanoid visibility through
  `Camera3D.hiddenBones={["model_id:canonical_bone"]}`. Hidden bones include
  their skinned descendants in beauty and CPU view passes while the complete
  actor continues to cast shadows. Camera position and target Anchors can now
  follow final animated humanoid joints after collision/contact correction.
  Discrete `activeCamera` cuts with three or more keys now take effect on the
  exact intermediate key frame instead of one frame later.

- Split retained PrimitiveAsset identity into geometry, material, decoded
  ImageAsset, GPU texture, and per-instance UV variation layers. Compound
  children with different `materialSeed` values now share mesh buffers and
  immutable texture pixels while preserving deterministic visual variation.
  Texture cache revisions track file metadata or resolver byte content for
  targeted hot reload. Added incremental preview preload sessions, cold-resource
  profiling counters, and shared renderer fallback textures. SHOWCASE 76 cold
  first-frame preparation dropped from roughly 34.25 seconds to 1.12 seconds
  in the reference debug build, with one Stone texture decode instead of
  repeated per-step decoding.

- Added first-class `MaterialAsset shading="pbr"` resources for typed
  primitives, with reusable image-backed base-color, metallic/roughness,
  normal, occlusion and emissive inputs; scalar PBR controls; UV/box/triplanar
  projection; repeat wrapping; and deterministic CompoundAsset texture
  variation. Added visual-only rounded box bevels that preserve the original
  bounds and simple collider.

- Added asset-owned universal PrimitiveAsset collision with disabled, solid,
  and sensor modes; auto or explicitly mismatched collider shapes; adjustable
  collider dimensions and transforms; collision filtering and material data;
  and reusable CompoundAsset composition. Solid primitive instances now feed
  the shared character collision world, including stair step-up and opt-in
  standing foot contact correction.

- Breaking DSL migration: added first-class typed `PrimitiveAsset` resources
  for box, sphere, plane, cylinder, cone, and wedge geometry. Primitive Models
  share the GLB PBR, shadow, lighting, physics, bounds, and retained GPU cache
  paths. Removed the `motionloom:box` source shorthand with a migration error,
  and moved implicit `Surface` geometry onto the same typed asset path.

- Added the public Scene 3D lighting stack: HDR/EXR equirectangular
  `EnvironmentLight`, roughness-aware diffuse/specular IBL, directional,
  point, spot and rectangular area lights, a filtered primary shadow map,
  ambient/contact shadow controls, and ACES/Reinhard color management.
- Registered lighting and grading properties for `AnimationTarget`, strict
  authoring analysis, showcase schema generation, native rendering and WASM
  WebGPU rendering. Scenes without authored lighting keep the previous studio
  fallback.

- Breaking DSL migration: replaced `<RigidBody2D>` with one explicit
  `<RigidBody dimension="2d|3d" type="static|dynamic|kinematic">` contract.
  The old tag is rejected with a migration diagnostic and has no compatibility
  alias. Added deterministic 2D/3D collision, static and kinematic colliders,
  damping, friction, restitution, continuous-collision substeps, random-access
  frame sampling, and retained 3D timeline baking for static initial poses.
- Breaking 3D transform correction: `Model.scaleMode` now defaults to `none`,
  preserving the authored glTF origin and units. `normalize_height` is now an
  explicit content-import mode rather than a renderer side effect.
- 3D rigid bodies now share the renderer quaternion, resolve `shape="auto"`
  from effective model bounds, use multi-point contact manifolds and
  swept-AABB CCD, and expose `<PhysicsDebug>` diagnostics.
- Added `motionloom_dsl_schema_json()` for a complete machine-readable
  tag/attribute catalog including required rigid-body attributes.

- Added universal Layer Puppet Warp with `target="@layer" capture="before"`.
  It captures all earlier visual siblings into one deformation surface while
  leaving later siblings as normal overlays.
- Migration: existing `target="GROUP_ID"` Puppet Warp behavior is unchanged.
  Universal Layer capture is additive and opt-in.
- Added opt-in `PuppetWarp solver="bones"` with role-based two-bone IK,
  fixed-length reach clamping, rigid vertex regions, joint volume preservation,
  and local `preserveOutside` replacement for full-character targets.
- Migration: existing Puppet Warp scripts continue to use the `soft` solver.
- Added `<LimbEnvelope d="... Z" alphaClip="true" handFrom="pin_id" />` for
  exact Path-shaped Bone IK areas. It lowers to local topology, preserves
  pixels outside the envelope, and keeps the hand/foot end region rigid.
- Added role-specific `<LimbRegion role="anchor|joint|control" d="... Z" />`
  areas so upper limb, bend seam, and lower limb/hand can be outlined and
  bound independently. The legacy single `LimbEnvelope` remains supported.
- Migration: existing scalar Limb Width and explicit `MeshTopology` bone rigs
  are unchanged; explicit topology remains authoritative.
- Added opt-in `PuppetWarp solver="chain"` with explicit parent-linked pins,
  fixed segment lengths, serial rigid deformation, and deterministic
  `SpringChain` follow-through for tails, hair, ropes, and tentacles.
- Migration: `soft` surface pins and `bones` Two-Bone IK remain unchanged;
  chain behavior is additive and selected only with `solver="chain"`.
- Added typed Component parameters, ordered Derived bindings, and Slot/Fill content.
- Added deterministic weighted Repeat Variants and per-property Vary controls.
- Added Layout padding, independent gaps, alignment, justification, and layoutSpan.
- Migration: existing Component, Repeat, and Layout scripts keep their previous
  defaults and require no changes; the new child tags and attributes are opt-in.
- Added `FaceLayout.noseHeight` as an absolute face-space Y coordinate, allowing
  eye and nose placement to be edited independently. Existing scripts that omit
  it retain their previous generated nose position; new scripts should author it
  explicitly when independent facial editing is required.

## 0.1.0

Initial public MotionLoom crate release.

- Parses MotionLoom graph DSL for scene, process, and mixed scene/process graphs.
- Renders scene/composition frames through CPU and wgpu-backed paths.
- Exports single frames and PNG sequences without FFmpeg.
- Exports video through a caller-supplied FFmpeg binary.
- Provides process/effect runtime evaluation and a process catalog for host UI integration.
- Provides preview APIs for MotionLoom-owned wgpu textures, caller-owned wgpu targets, and platform preview surfaces.
- Exposes `motionloom::api` as the recommended stable integration surface.

- Added an experimental read-only MeshAsset edit snapshot with runtime camera
  projection, exposed to WASM for source-preserving vertex editing in the landing
  page's Model editing (MeshAsset only) panel. HeadAsset remains parametric.
