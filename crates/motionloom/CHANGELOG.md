# Changelog

## Unreleased

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
