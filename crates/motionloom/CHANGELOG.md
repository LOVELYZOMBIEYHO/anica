# Changelog

## Unreleased

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

## 0.1.0

Initial public MotionLoom crate release.

- Parses MotionLoom graph DSL for scene, process, and mixed scene/process graphs.
- Renders scene/composition frames through CPU and wgpu-backed paths.
- Exports single frames and PNG sequences without FFmpeg.
- Exports video through a caller-supplied FFmpeg binary.
- Provides process/effect runtime evaluation and a process catalog for host UI integration.
- Provides preview APIs for MotionLoom-owned wgpu textures, caller-owned wgpu targets, and platform preview surfaces.
- Exposes `motionloom::api` as the recommended stable integration surface.
