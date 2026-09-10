# Geometry inspection and GLB export

Experimental authoring APIs sample the DSL into camera-independent static
geometry. No DSL tags or parser defaults change. Reports, images and GLBs are
derived outputs; the original script remains authoritative.

```rust,ignore
use motionloom::experimental::*;
let snapshot = extract_scene_geometry(&graph, &SceneGeometryOptions {
    scene_id: "S86AnimeHead".into(), frame: 0, include_hidden: false,
}).await?;
let diagnostics = check_scene_uvs(&snapshot, &UvCheckOptions::default())?;
let glb_bytes = export_scene_glb(&snapshot)?;
```

Extraction reuses Scene-to-World lowering, mesh generators, material resolution
and sampled skin matrices. It does not initialize a GPU. Camera nodes are
removed only from a working composite copy; camera visibility masks, lights,
backgrounds, RenderStyle and screen effects are omitted. Model transforms and
the selected pose are baked into world-space vertices. Each material draw
receives a named mesh node: this is a flattened static asset, not an animation
or original-node-hierarchy export.

`extract_scene_geometry_with_resolver` accepts preloaded assets for WASM hosts.
Core APIs return bytes/images rather than writing files. Native examples use
the existing `set_scene_asset_roots` convention. Browser UI integration is not
supplied by these command-line examples.

## UV checks

Reports include nonfinite/out-of-range UV vertices, zero-area UV triangles,
winding, UV-continuous islands, coverage/overlap estimates, padding conflicts
and average texel density. Images contain wireframe, island IDs, red overlaps,
amber padding conflicts and a checker pattern. UV (0,0) is the image top-left.

Meshes normally retain separate texture domains. Set `atlas_meshes` to indices
intended to share an atlas to append a combined report for cross-mesh overlap.
Mirroring and repeated instances can intentionally overlap. Coverage samples
pixel centers at the chosen resolution and clips to 0..1; subpixel overlap may
be missed and out-of-tile wrapping is not measured. Padding uses Manhattan
distance raster dilation, not exact geometric distances.

Vertex already substitutes XY projection when UVs are omitted. A cage whose
UVs all equal XY is flagged `fallback_xy`; this is an inference because the
AST does not retain whether those values were explicit. GLB missing/partial
UVs are separately flagged. Export does not unwrap, pack or silently repair
UVs; diagnostics do not prevent exporting valid geometry for inspection.

## Revisions and scope

`topology_signature` hashes ordered vertex counts and triangle indices.
`uv_signature` separately hashes UV values and provenance. Moving a model
does not change its topology signature. Pin the engine version when reproducing
assets because mesh-generation rules can evolve. The S86 regression pins both
signatures and triangle counts, compares another camera-orbit frame and checks
GLB roundtrip coordinates.

Supported containers include Timeline, Track, Sequence, Chain, Group and inline
Layer. Model/Character, CompoundAsset, MeshAsset, HairAsset and model/bone sampling
reuse renderer lowering. 2D Group presentation transforms are omitted because
they act on the rendered island. Scene-referencing Layers, 2D Repeat/Use/Precompose
and terrain/vegetation return typed unsupported errors. Camera-dependent terrain
LOD and vegetation deformation need explicit export support.

GLB embeds PNG textures and carries positions, normals, tangent handedness,
UVs, linear vertex colors, metallic-roughness, normal, AO, emission and alpha.
Supported specular, emissive-strength, transmission, IOR, volume and unlit
properties map to Khronos extensions. MotionLoom shader styling, depth-write,
sort behavior and screen-space transmission appearance are not portable
material guarantees. Skeletal animation export is not included.

The imported glTF COLOR_0 path now accounts for linear vertex colors before
the renderer's legacy factor decode; this prevents a second gamma conversion
from darkening a roundtrip. A native GPU regression compares original and
exported material brightness.

## Native commands

From the `anica` checkout; arguments after output are optional scene id, frame
and (for UV checks) diagnostic resolution:

```sh
cargo run -p motionloom --example check_scene_uvs -- \
  ../motionloom-example/showcase/s-000086/main.motionloom \
  /tmp/s86-uv-check S86AnimeHead 0 512
cargo run -p motionloom --example export_scene_glb -- \
  ../motionloom-example/showcase/s-000086/main.motionloom /tmp/s86.glb S86AnimeHead 0
cargo test -p motionloom --test geometry_export -- --include-ignored --test-threads=1
```

S86 frame 0 contains 18 material meshes, 1,030,981 render vertices and 345,702
triangles. The high vertex count retains the renderer's split normal/tangent
vertices. Its main face uses historical XY fallback UVs and therefore still
needs a dedicated atlas pass. The
GLB is usable for geometry inspection; these diagrams are not a ready-to-paint
atlas. A dedicated seam/unwrap authoring step remains necessary. The current
export passes Khronos glTF Validator with zero errors and zero warnings.

Validation: 568 library tests, 9 non-ignored RenderStyle tests and all 7 geometry
tests (including the explicit S86 and GPU cases) passed. WASM compilation passed
with the existing unused DevicePoller warning. Browser execution was not tested.
The current S86 geometry signatures are recorded in its
`cage-measurements.json`; changing the compact generator version requires an
intentional baseline update.
