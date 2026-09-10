# Facial and subdivision cages

FaceLayout now uses explicit components. See
[Face components and breaking migration](FACE_COMPONENTS.md).

MotionLoom has one control-cage intermediate representation and three authoring
levels. This keeps detailed modeling possible without making every head tens of
thousands of DSL lines.

## Compact semantic head

```xml
<HeadAsset id="hero_head" material="skin" archetype="humanoid"
           topology="facialCage" symmetry="x">
  <HeadShape size={[1.50,1.833,1.67]} />
  <FacialCage generatorVersion="1" segments="192" profileSegments="96"
              samplesPerSection="6" subdivision="1" orbitalRings="10"
              mouthRings="8" preserveProfile="true" uvMode="frontBack" />
  <HeadProfile>
    <HeadSection id="chin" at="-0.833" width="0.018"
                 frontDepth="0.462" backDepth="0.450" />
    <HeadSection id="forehead" at="0.200" width="1.398"
                 frontDepth="0.726" backDepth="-0.861" />
    <HeadSection id="crown_transition" at="0.420" width="1.500"
                 frontDepth="0.710" backDepth="-0.840" />
  </HeadProfile>
  <HeadDome start="0.200" top="1.0" centerDepth="-0.07"
            frontRadius="0.80" backRadius="0.80" samples="80" />
  <FaceLayout>
  <Eye id="eye_a" position={[-0.316,-0.113,0]} width="0.380" opening="0.210" tilt="0" socketWidth="0.550" socketHeight="0.510" socketDepth="0.139" />
  <Eye id="eye_b" position={[0.316,-0.113,0]} width="0.380" opening="0.210" tilt="0" socketWidth="0.550" socketHeight="0.510" socketDepth="0.139" />
  <Nose id="nose" position={[0,-0.363,0]} length="0.180" width="0.100" projection="0.060" />
  <Mouth id="mouth" position={[0,-0.561,0]} width="0.086" opening="0.014" upperLip="0.025" lowerLip="0.03" muzzleLength="0" muzzleWidth="0.3" />
</FaceLayout>
</HeadAsset>
```

The Rust generator samples the ordered profile with Catmull–Rom interpolation,
adds the dome, cuts front-surface patches, bridges orbital and mouth support
rings, closes the cage, compacts unused vertices, and generates UVs. The
`generatorVersion` freezes this algorithm. Native and WASM compile the same Rust
code.

`uvMode="fallbackXY"` reproduces the historical position-derived UV values.
`frontBack` assigns separate front/back regions, but it is only a deterministic
starting layout. Run the UV checker before painting.

## Explicit semantic head

```xml
<HeadAsset id="sculpt" material="skin" archetype="humanoid" topology="explicit">
  <HeadShape size={[1,1,1]} />
  <HeadCage subdivision="1">
    <Vertex position={[-1,0,0]} uv={[0,0]} pinned="true" />
    <Vertex position={[1,0,0]} uv={[1,0]} pinned="true" />
    <Vertex position={[1,1,0]} uv={[1,1]} />
    <Vertex position={[-1,1,0]} uv={[0,1]} />
    <Face indices={[0,1,2,3]} />
  </HeadCage>
</HeadAsset>
```

This level can contain a complete 25,000-vertex custom model. It preserves head
identity for comparison/export while keeping every unusual point available to
an LLM or editor. The fitting API can compare explicit heads but refuses
parameter fitting because individual vertices have no safe semantic edit rule.

## Generic surface

Use `MeshAsset` with the same `Vertex` and `Face`
children for clothing, limbs, props, creature parts, and other custom geometry.
Its subdivision defaults to 0; levels 1–2 use
`subdivisionScheme="catmullClark"` (the only supported scheme).
All three forms compile to `ControlCageNode` and share validation,
Catmull–Clark subdivision, UV interpolation, native/WASM rendering, inspection,
and GLB export.

The generic profiled-surface module separates cross-section interpolation and
front-patch selection from facial semantics. A future torso or limb generator
can reuse those Rust algorithms and the generic cage IR. MotionLoom does not add
a speculative `BodyAsset` contract until body-specific controls and deformation
rules have been validated.

## Breaking mesh syntax migration

The old `SubdivisionSurfaceAsset`, `ControlVertex`, and `ControlFace` tags now
fail parsing. Rename the container and children; geometry values do not change:

```xml
<!-- Before: rejected -->
<SubdivisionSurfaceAsset id="part" material="skin" subdivision="1">
  <ControlVertex position={[0,0,0]} uv={[0,0]} />
  <ControlFace indices={[0,1,2]} />
</SubdivisionSurfaceAsset>

<!-- After -->
<MeshAsset id="part" material="skin" subdivision="1" subdivisionScheme="catmullClark">
  <Vertex position={[0,0,0]} uv={[0,0]} />
  <Face indices={[0,1,2]} />
</MeshAsset>
```

## Inspection

Rust hosts call `api::generated_control_cage` for the complete cage or
`api::inspect_control_cage` for counts, pinning, edge incidence, subdivision,
and UV provenance. Browser hosts call
`motionloom_inspect_control_cage_json(script, assetId)`. Inspection does not
render, mutate the DSL, or read external files.

The removed `EyeAsset`, `EyeVertex`, and `EyeFace` tags are unsupported. Use the
generic names even when a cage happens to represent an eye.
