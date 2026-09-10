# Immediate preview: material foundation

The first-stage improvements are renderer internals shared by native and WASM
WebGPU. Existing MotionLoom scripts need no additional tags or quality settings.

- Material textures receive retained mip chains and 8x anisotropic filtering.
  Color filtering uses linear light with alpha-weighted mip generation; normal
  mip vectors are normalized; metallic, roughness and AO remain linear data.
  Texture caching includes the semantic role, including when a GLB reuses one
  image for multiple material slots.
- Material AO is sampled separately and attenuates indirect diffuse and specular
  light. It no longer stains base color or direct lighting. Primitive materials,
  GLB materials and blended terrain carry independent AO; missing AO is neutral.
- The 3D intermediate target and transmission snapshot use RGBA16Float. Tone
  mapping and output gamma run once after transparency and camera depth of field.
  Exposure, white balance and contrast remain before blending. The public Scene
  compositor still receives RGBA8; Scene Process bloom is not HDR bloom.

These changes improve handling of authored surface detail and bright highlights;
they do not generate rust, dirt, realistic skin, geometry or new illumination.
There is no temporal accumulation or mandatory offline rendering step. Mips are
built on texture-cache misses. Typical square texture mip storage adds about one
third; HDR intermediate targets use twice the bytes per pixel of RGBA8, and the
final resolve adds a fullscreen pass even when depth of field is disabled.
Immediate frame rate still depends on device, render size and scene complexity.

## Validation and S81 observation

S81 frame 360 (15 seconds at 24 fps), 1280x720, was captured before and after on
native Metal. Its predominantly flat toon materials have no decoded detail
textures, so the visual change is small. This is a regression comparison, not
evidence that the renderer can automatically reproduce a textured horror image.

Twelve warmed debug submissions averaged 47.2 ms before and 47.8 ms after in the
initial comparison; a final-build capture averaged 49.9 ms. These are CPU
submission timings, not GPU frame timings or
playback FPS; they do not establish browser or release performance. First-run
shader compilation also incurs startup cost and is excluded from this average.

GPU regression fixtures isolate material AO from direct light and verify that
high emission retains energy through defocus, including partial alpha coverage.
CPU mip tests cover linear-light filtering, transparent colors, normal vectors
and odd image edges. WASM compilation checks the shared code; browser runtime
performance and pixel equivalence require separate testing on target devices.

Validation on 2026-09-07: 568 library tests passed (the local HTTP server test
needed a rerun outside the sandbox), all 14 RenderStyle integration tests passed,
and all 3 native GPU material/optics tests passed. The WASM check passed with the
existing unused `DevicePoller` method warning.

Reproduce the native GPU fixtures with:

```sh
cargo test -p motionloom --test immediate_materials -- --ignored --test-threads=1
```
