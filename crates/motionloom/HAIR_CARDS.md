# Guide-authored hair cards

HairAsset compiles editable guides into closed, curved hair cards. The DSL is
the source of truth. Native and WASM use the same CPU mesh generator; this
extension adds no package, renderer quality setting, or runtime simulation.

```xml
<HairAsset id="bob_bangs" material="hair_brown" space="asset_local">
  <HairGroom>
    <HairGroup id="bangs" role="silhouette">
      <HairDefaults width="0.18" camber="0.12" roll="0" />
      <HairGuide id="bang_left" normal={[0,0,1]}>
        <HairPoint position={[-0.12,0.92,0.35]} width="0.12" />
        <HairPoint position={[-0.25,0.65,0.62]} />
        <HairPoint position={[-0.34,0.28,0.73]} width="0.14" />
        <HairPoint position={[-0.38,-0.12,0.67]} width="0.025" />
      </HairGuide>
      <HairMirror id="bang_right" source="bang_left" axis="x" />
    </HairGroup>
  </HairGroom>
  <HairRepresentations>
    <HairCards id="cards" lengthSegments="24" widthSegments="5"
               thickness="0.012" crossSection="arched" tipShape="point" />
  </HairRepresentations>
</HairAsset>
```

Declare `hair_brown` as an ordinary MaterialAsset and instantiate `bob_bangs`
with Model. A complete rotating example is available in the sibling
motionloom-example repository: `showcase/s-000086/hair-cards-review.motionloom`.
It is a pair of diagnostic bangs, not a generated complete hairstyle.

## Defaults and overrides

HairDefaults is optional, occurs at most once per HairGroup, and must precede
all guides/mirrors. The precedence is point > guide > group HairDefaults >
system defaults. Values do not inherit from a preceding point or another group.
HairGuide accepts the same numeric settings as HairDefaults, plus id and normal.

| Setting | Default | Meaning |
| --- | --- | --- |
| width | 0.1 | Full card width in asset units; positive |
| camber | 0 | Arch height as a fraction of width; 0–1 |
| roll | 0 | Absolute degrees about the transported root frame |
| radius | 0.003 | Stored for future representations; unused by static cards |
| stiffness | 1 | Stored for future dynamics; unused by static cards |

Positions remain required on HairPoint. Set camber explicitly when using an
arched or v_shape section: camber=0 makes either section flat. Group role is
metadata, not an automatic hairstyle preset. HairAsset seed affects the existing
material seed path, not procedural guide distribution.

## Root direction and mirroring

HairGuide normal is an optional outward root direction in the asset's declared
coordinate space. It is projected perpendicular to the first guide segment.
Zero vectors or directions parallel to that segment produce a parse error.
When omitted, a deterministic axis-based frame is used. Points run root to tip.

HairMirror accepts id, source, and axis (x/y/z). Reflection is across the plane
through zero in asset coordinates. Source must name an earlier guide or mirror
in the same HairGroup. Guide ids, including generated ones, must be unique across
the HairAsset. Forward references, cycles, missing sources and duplicate ids fail.
Mirrors copy resolved point settings, reflect positions and the root normal, and
negate roll. They expand into ordinary guides; they add no runtime node/resource.
The symmetric card surface preserves outward normals and triangle winding.
Each resulting card still uses the full 0–1 UV domain; mirrors do not allocate an
atlas or guarantee a desired orientation for asymmetric painted motifs.

## Geometry behavior

- Centripetal Catmull-Rom centerlines reduce overshoot for uneven control spacing.
- A fixed integration table, independent of lengthSegments, transports the unrolled
  frame. Absolute roll is applied afterward. Arc-length resampling distributes
  mesh rows and longitudinal UVs evenly (approximately, using the fixed table).
- Arched surface normals follow the generated geometry. Side/root walls use
  separate normals so their shading does not flatten the curved front surface.
- Point tips taper width and thickness to zero over the final 22% of length.
  Round tips use a rounded elliptical taper over the final 15%. Blunt tips retain
  their authored terminal width/thickness. Degenerate terminal triangles are omitted.
- Card UV U runs across the width and V runs root to tip. No automatic scalp
  binding, collision avoidance, atlas selection or generated child clumps is added.
- One HairCards representation is currently supported. defaultRepresentation and
  HairLOD are optional references, not automatic distance-based LOD switching.

## Migration and validation

Old explicit HairPoint widths remain parseable. For example:

```xml
<!-- Before -->
<HairGuide id="lock">
  <HairPoint position={[0,1,0]} width="0.18" camber="0.12" />
  <HairPoint position={[0,0,0.2]} width="0.18" camber="0.12" />
</HairGuide>
<!-- After, within the same HairGroup -->
<HairDefaults width="0.18" camber="0.12" />
<HairGuide id="lock" normal={[0,0,1]}>
  <HairPoint position={[0,1,0]} />
  <HairPoint position={[0,0,0.2]} />
</HairGuide>
```

This owner-approved correction changes generated geometry for existing guides:
accumulated roll is removed, spacing changes, normals follow curvature, and tips
close differently. Review old hair looks and regenerate geometry/UV signatures
and exports rather than treating previous mesh hashes as unchanged. Serialized
guides without normal continue to deserialize with the fallback frame. Rust
HairGuideNode literals must include `normal: None` or an explicit direction.

Run `cargo test -p motionloom --lib hair` and
`cargo test -p motionloom --test hair_cards`. GPU smoke rendering of the example
checks its visible result; a WASM build does not certify browser pixel parity.

Verification for this revision: 570 library tests passed (569 in the sandbox,
the local HTTP-server test passed separately with socket access), plus 5 hair
integration tests. WASM library compilation passed with an existing unused
DevicePoller-method warning. Native GPU frames 0 and 90 of the example were
rendered and visually inspected after fixing camera framing. The PNGs live
beside the example as hair-cards-review-front.png and hair-cards-review-quarter.png.
Browser rendering and real-time performance were not measured in this revision.
