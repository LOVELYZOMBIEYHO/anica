# Atmosphere medium

`AtmosphereFog` lowers to renderer-independent `AtmosphereMediumPlan`.
MotionLoom's immediate WebGPU renderer injects single-scattered light into a
camera-aligned 3D grid, integrates
radiance and extinction along each view column, then composites the result
before temporal resolve, depth of field, and display tone mapping.

```xml
<AtmosphereFog id="sea" density="0.04" anisotropy="0.72"
               scatteringColor={[0.02,0.08,0.12]}
               boundsMin={[-20,-10,-35]} boundsMax={[20,12,4]}>
  <VolumetricScattering id="shafts" lightRef="sun" shaftStrength="1.4"
                        maxDistance="30" shadowed="true"
                        quality="high" maxBounces="4" />
  <WaterCaustics id="caustics" intensity="0.45" scale="0.09" speed="0.28"
                 attenuation="0.6" color="#BFE9FF"
                 volumeTerm="true" surfaceTerm="true" />
</AtmosphereFog>
```

`lightRef` must resolve to a shadow-casting `DirectionalLight` or `SpotLight`
in the same 3D `CompositeGroup` when `shadowed="true"`. One volume light is
supported per 3D group. Without children, the live preview uses the lower-cost
analytical evaluation and allocates no froxel textures.

The host preview profile controls the grid:

| Profile | XY tile | Z slices |
| --- | ---: | ---: |
| Portable | 16 px | 32 |
| Balanced | 12 px | 48 |
| Cinematic | 8 px | 64 |
| Ultra | 6 px | 96 |

The regression target at 1200×800 is at most 2.5 ms for Portable, 4 ms for
Balanced, and 6 ms for Cinematic on the reference desktop adapter. Hosts can
read `Scene3DFrameProfile.froxel_grid` and `froxel_bytes`; GPU pass timing
continues to come from the parent compositor's timestamp queries.

The implementation uses two `rgba16float` 3D textures. Injection stores local
in-scatter and density. Integration stores accumulated radiance and integrated
density; composition combines that density with the authored RGB extinction,
so water absorption remains wavelength dependent. Separate textures keep the
passes valid on WebGPU, which does not allow portable read/write storage
texture aliasing.

Current physical limits are intentional: single scattering, one shadow map,
Henyey-Greenstein phase approximation, and procedural caustics. The CPU
renderer parses and validates the feature but does not simulate it. Browser
rendering requires 3D storage textures; unsupported adapters report the device
validation failure instead of silently rendering a different effect.

Animated ids support `density`, `anisotropy`, `baseHeight`, `heightFalloff`,
`edgeFeather`, `shaftStrength`, `maxDistance`, `maxBounces`, `intensity`,
`scale`, `speed`, and `attenuation`.

Set `debugView` on `VolumetricScattering` to `density`, `shadow`, `inscatter`,
`opticalDepth`, `transmittance`, or `caustics` while diagnosing a scene. Keep
the default `none` for final output.
