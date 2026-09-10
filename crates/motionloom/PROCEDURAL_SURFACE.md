# Procedural Surface (GPU)

`procedural_surface` is an additive Process effect. It generates a 2D image with virtual relief normals, not a mesh, fluid simulation or Camera3D material. Native scene rendering and WASM use the same WGSL and parameter evaluator.

Use `effect="procedural_surface"` with optional `kernel="procedural_surface.wgsl"`. Existing Pass requirements and defaults are unchanged. No migration is needed for old scripts. Before: an ordinary Process with existing effects. After: optionally insert this new pass; no other tags need changing.

## Parameters

All numeric parameters accept the existing time expressions and AnimationTarget mechanism. Numeric results must be finite; values outside the ranges below are clamped. Colors are static CSS-style colors understood by MotionLoom; alpha components are ignored because input alpha is preserved.

| Parameter | Default | Range / units |
| --- | --- | --- |
| amount | 1 | 0–1; zero is an exact input bypass |
| flowStrength | 0 | 0–1; opt-in directional ribbons and analytic transport; zero preserves original relief |
| seed | 83 | 0–65535 |
| scale | 3.8 | 0.1–30 |
| evolution | 0 | -10000–10000; authored phase, not accumulated delta time |
| warpStrength | 1.4 | 0–4 |
| detailStrength | 0.65 | 0–2 |
| baseColor | #171C32 | substrate color |
| veinColor | #D9AE50 | metal color |
| veinWidth | 0.035 | 0.001–0.2 |
| poolAmount | 0.22 | 0–1 |
| relief | 0.6 | 0–2 |
| roughness | 0.28 | 0.04–1 |
| lightAzimuth | 125 | degrees, -3600–3600 |
| lightElevation | 40 | degrees, 1–89 |
| viewX, viewY | 0 | surface-space coordinates, -10000–10000 |
| viewZoom | 1 | 0.1–20 |
| viewRotation | 0 | degrees, -3600–3600 |

## Composition and limitations

Attach to a Scene using `<Effects><Effect process="..." /></Effects>` (write each tag on its own line), or use a Process pass directly. Start with an opaque rectangle for a full-frame plate. The effect preserves input coverage and replaces its RGB; it does not guess which pixels are titles. Render titles in a transparent Scene and composite with `over` after the plate effect. The existing Group effect path is not used by this showcase.

The GPU output follows the current RGBA8 compositor contract, even if a DSL Tex declares rgba16f. This is not an HDR pipeline upgrade. Lighting is internal to the effect, not Scene Light3D lighting. Blur-based focus approximation is not depth-based DOF. CPU-only rendering returns an explicit unsupported error. Standalone WASM Process masks are rejected for this new effect; use input alpha instead. Scene-level existing mask composition remains external to the pass.

The native pipeline is lazy and cached; output textures are reused when their references are released. Uniforms use the existing post-process allocator. The standalone browser Process helper retains its existing per-call renderer lifecycle; this change does not claim to cache that entire host between calls.

Example: `examples/motionloom/scene/motion_graphics/procedural_surface.motionloom`.
