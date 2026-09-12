# S90 leather shoe Weaver benchmark

S90 intentionally keeps the scene small: the supplied shoe GLB with its embedded
4096x4096 base-color and metallic/roughness textures, one neutral floor,
one physical camera and a three-source photographic RectAreaLight rig. MotionLoom
world units are metres: the pair is about 41 cm wide, rather than the earlier
four-metre interpretation. Preview AO/contact shadows remain preview-only aids;
Weaver obtains its contact and soft-shadow detail from sampled light transport.

From the `anica` directory:

```sh
WEAVER_TEST_WIDTH=480 WEAVER_TEST_HEIGHT=272 WEAVER_TEST_SAMPLES=128 \
WEAVER_DENOISER_LIBRARY=/Applications/Blender.app/Contents/Resources/lib/libOpenImageDenoise.dylib \
cargo test -p motionloom --features weaver --lib \
  weaver::tests::s90::s90_shoe_render --offline -- --ignored --nocapture
```

Omit `WEAVER_DENOISER_LIBRARY` to inspect only raw sampling. Output is written to
workspace `.render-output/weaver-s90/<job-hash>/`; the report prints the exact
directory. This fixture always renders frame zero.

The asset exposed a previous float-address limitation when two 4096x4096 images
placed the second texture above 16,777,216 packed pixels. Texture offsets now use
raw u32 bits in the float-backed material record, preserving the full address
without resizing either image.

The former directional/point baseline was job `e4e2fcaf3de59f8b`, 480x272,
128 fixed samples and 118,920 triangles. It is retained only as a historical
comparison; current output hashes include the physical scale and sampled-area
lighting changes.
