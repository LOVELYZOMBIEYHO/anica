// =========================================
// =========================================
// crates/motionloom/src/world/primitive/face_textures.rs

use std::sync::Arc;

use super::*;
use crate::dsl::{FaceTextureNode, PrimitiveGeometry};

/// Retain component-local UVs while regenerating the attached surface after edits.
pub(super) fn append(
    mesh: &mut GlbMeshData,
    asset: &PrimitiveAssetNode,
    images: &HashMap<String, GlbTextureData>,
) {
    let PrimitiveGeometry::HeadSurface {
        topology,
        facial_cage,
        head_profile,
        head_dome,
        face_layout: Some(layout),
        ..
    } = &asset.geometry
    else {
        return;
    };
    let rows = facial_cage
        .as_ref()
        .filter(|_| topology == "facialcage")
        .map(|settings| {
            super::facial_cage_mesh::sampled_rows(settings, head_profile, head_dome.as_ref())
        });
    let triangles: Vec<_> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            [
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            ]
        })
        .collect();
    let surface = |x: f32, y: f32| match &rows {
        Some(rows) => {
            super::facial_cage_mesh::face_surface(rows, layout, x as f64, y as f64) as f32
        }
        None => sample(&triangles, x, y, 2, true),
    };
    for eye in &layout.eyes {
        append_eye(mesh, eye, images, &surface);
    }
    for brow in &layout.eyebrows {
        surface_patch(
            mesh,
            &brow.id,
            brow.position,
            [brow.width, brow.thickness],
            brow.tilt,
            brow.texture.as_ref(),
            images.get(&brow.id),
            &surface,
            Some(brow.arch),
        );
    }
    for nose in &layout.noses {
        if let Some(texture) = &nose.texture {
            patch(
                mesh,
                &nose.id,
                nose.position,
                [nose.width * 2.0, nose.length * 2.0],
                0.0,
                texture,
                images.get(&nose.id),
                &surface,
            );
        }
    }
    for mouth in &layout.mouths {
        if let Some(texture) = &mouth.texture {
            patch(
                mesh,
                &mouth.id,
                mouth.position,
                [
                    mouth.width,
                    mouth.opening + mouth.upper_lip + mouth.lower_lip,
                ],
                0.0,
                texture,
                images.get(&mouth.id),
                &surface,
            );
        }
    }
    for ear in &layout.ears {
        if let Some(texture) = &ear.texture {
            let first = mesh.positions.len();
            let first_triangle = mesh.triangles.len();
            let first_index = mesh.indices.len();
            let side = ear.position[0] >= 0.0;
            patch(
                mesh,
                &ear.id,
                [ear.position[2], ear.position[1], 0.0],
                [ear.width, ear.height],
                0.0,
                texture,
                images.get(&ear.id),
                &|z, y| sample(&triangles, z, y, 0, side),
            );
            // Side patches use Z/Y coordinates; restore head-local XYZ afterwards.
            for i in first..mesh.positions.len() {
                mesh.positions[i].swap(0, 2);
                if let Some(n) = &mut mesh.normals[i] {
                    n.swap(0, 2);
                }
            }
            for t in &mut mesh.triangles[first_triangle..] {
                t.indices.swap(1, 2);
            }
            for t in mesh.indices[first_index..].chunks_exact_mut(3) {
                t.swap(1, 2);
            }
        }
    }
}

/// Generate a convex sclera, optional iris/pupil, and any number of lid-bound liners.
fn append_eye(
    mesh: &mut GlbMeshData,
    eye: &crate::dsl::EyeNode,
    images: &HashMap<String, GlbTextureData>,
    surface: &impl Fn(f32, f32) -> f32,
) {
    let material = component_material(mesh, &eye.id, eye.texture.as_ref(), images.get(&eye.id));
    let part = begin_part(mesh, &eye.id);
    const SEGMENTS: u32 = 48;
    const RINGS: u32 = 12;
    let start = mesh.positions.len() as u32;
    let center_z = eye_ball_z(eye, surface, 0.0, 0.0);
    push_vertex(
        mesh,
        [eye.position[0], eye.position[1], center_z],
        [0.5, 0.5],
        [0.0, 0.0, 1.0],
    );
    for ring in 1..=RINGS {
        let r = ring as f32 / RINGS as f32;
        for segment in 0..SEGMENTS {
            let angle = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
            let local = eye_local(eye, r, angle);
            let [x, y] = rotate_local(eye.position, local, eye.tilt);
            let z = eye_ball_z(eye, surface, local[0], local[1]);
            let uv = transformed_uv(
                [0.5 + local[0] / eye.width, 0.5 - local[1] / eye.opening],
                eye.texture.as_ref(),
            );
            let normal = surface_normal(surface, x, y);
            push_vertex(mesh, [x, y, z], uv, normal);
        }
    }
    for segment in 0..SEGMENTS {
        let next = (segment + 1) % SEGMENTS;
        push_triangle(
            mesh,
            [start, start + 1 + segment, start + 1 + next],
            material,
            part,
        );
    }
    for ring in 1..RINGS {
        let a = start + 1 + (ring - 1) * SEGMENTS;
        let b = a + SEGMENTS;
        for segment in 0..SEGMENTS {
            let next = (segment + 1) % SEGMENTS;
            push_triangle(mesh, [a + segment, b + segment, a + next], material, part);
            push_triangle(mesh, [a + next, b + segment, b + next], material, part);
        }
    }
    if let Some(iris) = &eye.iris {
        append_iris(mesh, eye, iris, images, surface);
    }
    for liner in &eye.eyeliners {
        append_eyeliner(mesh, eye, liner, images, surface);
    }
}

fn eye_local(eye: &crate::dsl::EyeNode, radius: f32, angle: f32) -> [f32; 2] {
    let sine = angle.sin();
    [
        eye.width * 0.5 * radius * angle.cos(),
        eye.opening * 0.5 * radius * sine * sine.abs().powf(0.20),
    ]
}

fn rotate_local(center: [f32; 3], local: [f32; 2], tilt: f32) -> [f32; 2] {
    let (sin, cos) = tilt.to_radians().sin_cos();
    [
        center[0] + local[0] * cos - local[1] * sin,
        center[1] + local[0] * sin + local[1] * cos,
    ]
}

fn eye_ball_z(
    eye: &crate::dsl::EyeNode,
    surface: &impl Fn(f32, f32) -> f32,
    local_x: f32,
    local_y: f32,
) -> f32 {
    let [x, y] = rotate_local(eye.position, [local_x, local_y], eye.tilt);
    let nx = local_x / (eye.width * 0.5);
    let ny = local_y / (eye.opening * 0.5);
    let dome = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
    surface(x, y) + eye.position[2] + 0.004 + dome * eye.opening.min(eye.width) * 0.28
}

fn append_iris(
    mesh: &mut GlbMeshData,
    eye: &crate::dsl::EyeNode,
    iris: &crate::dsl::IrisNode,
    images: &HashMap<String, GlbTextureData>,
    surface: &impl Fn(f32, f32) -> f32,
) {
    let material = component_material(mesh, &iris.id, iris.texture.as_ref(), images.get(&iris.id));
    let part = begin_part(mesh, &iris.id);
    let start = mesh.positions.len() as u32;
    let center_local = [iris.position[0], iris.position[1]];
    let [center_x, center_y] = rotate_local(eye.position, center_local, eye.tilt);
    let center_z =
        eye_ball_z(eye, surface, center_local[0], center_local[1]) + iris.position[2] + 0.003;
    push_vertex(
        mesh,
        [center_x, center_y, center_z],
        transformed_uv([0.5, 0.5], iris.texture.as_ref()),
        [0.0, 0.0, 1.0],
    );
    const SEGMENTS: u32 = 48;
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
        let boundary = iris_boundary(iris, angle);
        let local = [
            iris.position[0] + boundary[0],
            iris.position[1] + boundary[1],
        ];
        let [x, y] = rotate_local(eye.position, local, eye.tilt);
        let z = eye_ball_z(eye, surface, local[0], local[1]) + iris.position[2] + 0.003;
        push_vertex(
            mesh,
            [x, y, z],
            transformed_uv(
                [
                    0.5 + 0.5 * boundary[0] / (iris.radius * iris.scale[0]),
                    0.5 - 0.5 * boundary[1] / (iris.radius * iris.scale[1]),
                ],
                iris.texture.as_ref(),
            ),
            surface_normal(surface, x, y),
        );
    }
    for segment in 0..SEGMENTS {
        push_triangle(
            mesh,
            [
                start,
                start + 1 + segment,
                start + 1 + (segment + 1) % SEGMENTS,
            ],
            material,
            part,
        );
    }
    if iris.pupil_radius > 0.0 {
        append_pupil(mesh, eye, iris, surface);
    }
}

fn iris_boundary(iris: &crate::dsl::IrisNode, angle: f32) -> [f32; 2] {
    let (sin, cos) = angle.sin_cos();
    let [mut x, mut y] = [cos, sin];
    if iris.shape == "square" {
        let edge = x.abs().max(y.abs()).max(1e-6);
        x /= edge;
        y /= edge;
    }
    [
        iris.radius * iris.scale[0] * x,
        iris.radius * iris.scale[1] * y,
    ]
}

/// Pupil radius remains independently visible even when the iris has no image.
fn append_pupil(
    mesh: &mut GlbMeshData,
    eye: &crate::dsl::EyeNode,
    iris: &crate::dsl::IrisNode,
    surface: &impl Fn(f32, f32) -> f32,
) {
    let material = solid_component_material(mesh, &format!("{}_pupil", iris.id), [6, 5, 12, 255]);
    let part = begin_part(mesh, &format!("{}_pupil", iris.id));
    let start = mesh.positions.len() as u32;
    let center_local = [iris.position[0], iris.position[1]];
    let [center_x, center_y] = rotate_local(eye.position, center_local, eye.tilt);
    let center_z =
        eye_ball_z(eye, surface, center_local[0], center_local[1]) + iris.position[2] + 0.006;
    push_vertex(
        mesh,
        [center_x, center_y, center_z],
        [0.5, 0.5],
        [0.0, 0.0, 1.0],
    );
    const SEGMENTS: u32 = 32;
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * segment as f32 / SEGMENTS as f32;
        let local = [
            iris.position[0] + iris.pupil_radius * iris.scale[0] * angle.cos(),
            iris.position[1] + iris.pupil_radius * iris.scale[1] * angle.sin(),
        ];
        let [x, y] = rotate_local(eye.position, local, eye.tilt);
        push_vertex(
            mesh,
            [
                x,
                y,
                eye_ball_z(eye, surface, local[0], local[1]) + iris.position[2] + 0.006,
            ],
            [0.5 + 0.5 * angle.cos(), 0.5 - 0.5 * angle.sin()],
            surface_normal(surface, x, y),
        );
    }
    for segment in 0..SEGMENTS {
        push_triangle(
            mesh,
            [
                start,
                start + 1 + segment,
                start + 1 + (segment + 1) % SEGMENTS,
            ],
            material,
            part,
        );
    }
}

fn append_eyeliner(
    mesh: &mut GlbMeshData,
    eye: &crate::dsl::EyeNode,
    liner: &crate::dsl::EyelinerNode,
    images: &HashMap<String, GlbTextureData>,
    surface: &impl Fn(f32, f32) -> f32,
) {
    let material = if liner.texture.is_some() {
        component_material(
            mesh,
            &liner.id,
            liner.texture.as_ref(),
            images.get(&liner.id),
        )
    } else {
        solid_component_material(mesh, &liner.id, [30, 22, 42, 255])
    };
    let part = begin_part(mesh, &liner.id);
    let start = mesh.positions.len() as u32;
    const SEGMENTS: u32 = 32;
    for segment in 0..=SEGMENTS {
        let s = segment as f32 / SEGMENTS as f32;
        let local = liner_point(eye, liner, s);
        let before = liner_point(eye, liner, (s - 0.002).max(0.0));
        let after = liner_point(eye, liner, (s + 0.002).min(1.0));
        let tangent = normalize2([after[0] - before[0], after[1] - before[1]]);
        let outward = if liner.edge == "upper" {
            [-tangent[1], tangent[0]]
        } else {
            [tangent[1], -tangent[0]]
        };
        let taper = endpoint_taper(s, liner.taper);
        for row in 0..=2 {
            let across = row as f32 / 2.0;
            let distance = (across - 0.20) * liner.thickness * taper;
            let point = [
                local[0] + outward[0] * distance,
                local[1] + outward[1] * distance,
            ];
            let [x, y] = rotate_local(eye.position, point, eye.tilt);
            let uv = transformed_uv([s, 1.0 - across], liner.texture.as_ref());
            push_vertex(
                mesh,
                [x, y, surface(x, y) + 0.008],
                uv,
                surface_normal(surface, x, y),
            );
        }
    }
    for segment in 0..SEGMENTS {
        let a = start + segment * 3;
        for row in 0..2 {
            push_triangle(mesh, [a + row, a + 3 + row, a + row + 1], material, part);
            push_triangle(
                mesh,
                [a + row + 1, a + 3 + row, a + 4 + row],
                material,
                part,
            );
        }
    }
}

fn liner_point(eye: &crate::dsl::EyeNode, liner: &crate::dsl::EyelinerNode, s: f32) -> [f32; 2] {
    let t = liner.span[0] + (liner.span[1] - liner.span[0]) * s;
    let q = t * 2.0 - 1.0;
    let sign = if liner.edge == "upper" { 1.0 } else { -1.0 };
    let x = q * eye.width * 0.5 - liner.extension[0] * (1.0 - s) + liner.extension[1] * s;
    let arch = (1.0 - q * q).max(0.0).sqrt();
    let y = sign * eye.opening * 0.5 * arch + liner.tip_lift[0] * (1.0 - s) + liner.tip_lift[1] * s;
    [x, y]
}

fn endpoint_taper(s: f32, taper: [f32; 2]) -> f32 {
    if s < 0.2 {
        (1.0 - taper[0]) + taper[0] * s / 0.2
    } else if s > 0.8 {
        1.0 - taper[1] * (s - 0.8) / 0.2
    } else {
        1.0
    }
}

fn normalize2(value: [f32; 2]) -> [f32; 2] {
    let length = (value[0] * value[0] + value[1] * value[1]).sqrt().max(1e-8);
    [value[0] / length, value[1] / length]
}

fn transformed_uv(uv: [f32; 2], texture: Option<&FaceTextureNode>) -> [f32; 2] {
    let Some(texture) = texture else { return uv };
    let x = uv[0] - 0.5;
    let y = uv[1] - 0.5;
    let (sin, cos) = texture.rotation.to_radians().sin_cos();
    [
        (x * cos + y * sin) / texture.scale[0] + 0.5 - texture.offset[0],
        (-x * sin + y * cos) / texture.scale[1] + 0.5 - texture.offset[1],
    ]
}

fn component_material(
    mesh: &mut GlbMeshData,
    id: &str,
    texture: Option<&FaceTextureNode>,
    image: Option<&GlbTextureData>,
) -> usize {
    let Some(_) = texture else { return 0 };
    let material = mesh.materials.len();
    let tex_index = mesh.textures.len();
    mesh.textures.push(image.cloned());
    mesh.materials.push(GlbMaterialData {
        name: Some(id.into()),
        base_color_texture: Some(tex_index),
        metallic_factor: 0.0,
        roughness_factor: 0.62,
        specular_factor: 0.08,
        alpha_mode: GlbAlphaMode::Mask,
        alpha_cutoff: 0.1,
        double_sided: true,
        ..Default::default()
    });
    material
}

/// A one-pixel texture preserves semantic component colors on primitive actors.
fn solid_component_material(mesh: &mut GlbMeshData, id: &str, rgba: [u8; 4]) -> usize {
    let material = mesh.materials.len();
    let texture = mesh.textures.len();
    mesh.textures.push(Some(GlbTextureData {
        width: 1,
        height: 1,
        rgba: Arc::new(rgba.to_vec()),
    }));
    mesh.materials.push(GlbMaterialData {
        name: Some(id.into()),
        base_color_texture: Some(texture),
        metallic_factor: 0.0,
        roughness_factor: 0.72,
        specular_factor: 0.08,
        double_sided: true,
        ..Default::default()
    });
    material
}

fn begin_part(mesh: &mut GlbMeshData, id: &str) -> usize {
    let part = mesh.mesh_names.len();
    mesh.mesh_names.push(Some(id.into()));
    part
}

fn surface_normal(surface: &impl Fn(f32, f32) -> f32, x: f32, y: f32) -> [f32; 3] {
    let epsilon = 0.0001;
    let nx = -(surface(x + epsilon, y) - surface(x - epsilon, y)) / (2.0 * epsilon);
    let ny = -(surface(x, y + epsilon) - surface(x, y - epsilon)) / (2.0 * epsilon);
    normalize([nx, ny, 1.0])
}

fn push_vertex(mesh: &mut GlbMeshData, position: [f32; 3], uv: [f32; 2], normal: [f32; 3]) {
    mesh.positions.push(position);
    mesh.normals.push(Some(normal));
    mesh.texcoords.push(Some(uv));
    mesh.colors.push(Some([1.0; 4]));
    mesh.joints.push(None);
    mesh.weights.push(None);
    for (axis, value) in position.iter().enumerate() {
        mesh.bounds_min[axis] = mesh.bounds_min[axis].min(*value);
        mesh.bounds_max[axis] = mesh.bounds_max[axis].max(*value);
    }
}

fn push_triangle(mesh: &mut GlbMeshData, indices: [u32; 3], material: usize, part: usize) {
    mesh.indices.extend(indices);
    mesh.triangles.push(GlbTriangle {
        indices,
        material: Some(material),
        mesh: Some(part),
        mesh_node: None,
    });
}

/// Project onto actual procedural geometry when no analytic section surface exists.
fn sample(triangles: &[[[f32; 3]; 3]], x: f32, y: f32, axis: usize, front: bool) -> f32 {
    let horizontal = if axis == 2 { 0 } else { 2 };
    let mut result: Option<f32> = None;
    for t in triangles {
        let [a, b, c] = *t;
        let det = (b[1] - c[1]) * (a[horizontal] - c[horizontal])
            + (c[horizontal] - b[horizontal]) * (a[1] - c[1]);
        if det.abs() < 1e-10 {
            continue;
        }
        let u = ((b[1] - c[1]) * (x - c[horizontal])
            + (c[horizontal] - b[horizontal]) * (y - c[1]))
            / det;
        let v = ((c[1] - a[1]) * (x - c[horizontal])
            + (a[horizontal] - c[horizontal]) * (y - c[1]))
            / det;
        if u < -1e-5 || v < -1e-5 || u + v > 1.00001 {
            continue;
        }
        let depth = u * a[axis] + v * b[axis] + (1.0 - u - v) * c[axis];
        result = Some(result.map_or(depth, |old| {
            if front {
                old.max(depth)
            } else {
                old.min(depth)
            }
        }));
    }
    result.unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments)]
fn patch(
    mesh: &mut GlbMeshData,
    id: &str,
    center: [f32; 3],
    size: [f32; 2],
    tilt: f32,
    texture: &FaceTextureNode,
    image: Option<&GlbTextureData>,
    surface: &impl Fn(f32, f32) -> f32,
) {
    surface_patch(
        mesh,
        id,
        center,
        size,
        tilt,
        Some(texture),
        image,
        surface,
        None,
    );
}

/// Brows keep their ribbon shape while image transforms operate in local UV space.
#[allow(clippy::too_many_arguments)]
fn surface_patch(
    mesh: &mut GlbMeshData,
    id: &str,
    center: [f32; 3],
    size: [f32; 2],
    tilt: f32,
    binding: Option<&FaceTextureNode>,
    image: Option<&GlbTextureData>,
    surface: &impl Fn(f32, f32) -> f32,
    arch: Option<f32>,
) {
    let defaults = FaceTextureNode {
        asset: String::new(),
        source: None,
        offset: [0.0; 2],
        scale: [1.0; 2],
        rotation: 0.0,
    };
    let texture = binding.unwrap_or(&defaults);
    // Untextured eyebrows use the head's existing material, including clay shading.
    let material = if binding.is_some() {
        mesh.materials.len()
    } else {
        0
    };
    if binding.is_some() {
        let tex_index = mesh.textures.len();
        mesh.textures.push(image.cloned());
        mesh.materials.push(GlbMaterialData {
            name: Some(id.into()),
            base_color_texture: Some(tex_index),
            metallic_factor: 0.0,
            roughness_factor: 0.62,
            specular_factor: 0.08,
            alpha_mode: GlbAlphaMode::Mask,
            alpha_cutoff: 0.1,
            double_sided: true,
            ..Default::default()
        });
    }
    let part = mesh.mesh_names.len();
    mesh.mesh_names.push(Some(id.into()));
    let start = mesh.positions.len() as u32;
    let (sin, cos) = tilt.to_radians().sin_cos();
    let (ts, tc) = texture.rotation.to_radians().sin_cos();
    const N: u32 = 24;
    let rows = if arch.is_some() { 2 } else { N };
    for row in 0..=rows {
        for col in 0..=N {
            let uv = [col as f32 / N as f32, row as f32 / rows as f32];
            let u = (uv[0] - 0.5) * texture.scale[0].abs();
            let v = (uv[1] - 0.5) * texture.scale[1].abs();
            let (dx, dy) = if let Some(arch) = arch {
                (
                    (uv[0] - 0.5) * size[0],
                    (0.5 - uv[1]) * size[1] + 4.0 * arch * uv[0] * (1.0 - uv[0]),
                )
            } else {
                (
                    (u * tc - v * ts + texture.offset[0]) * size[0],
                    -(u * ts + v * tc + texture.offset[1]) * size[1],
                )
            };
            let x = center[0] + dx * cos - dy * sin;
            let y = center[1] + dx * sin + dy * cos;
            let z = surface(x, y) + center[2] + 0.003;
            mesh.positions.push([x, y, z]);
            let epsilon = 0.0001;
            let nx = -(surface(x + epsilon, y) - surface(x - epsilon, y)) / (2.0 * epsilon);
            let ny = -(surface(x, y + epsilon) - surface(x, y - epsilon)) / (2.0 * epsilon);
            mesh.normals.push(Some(normalize([nx, ny, 1.0])));
            let mapped = if arch.is_some() {
                // Invert placement: positive scale enlarges the image over the ribbon.
                let x = uv[0] - 0.5 - texture.offset[0];
                let y = uv[1] - 0.5 - texture.offset[1];
                [
                    (x * tc + y * ts) / texture.scale[0] + 0.5,
                    (-x * ts + y * tc) / texture.scale[1] + 0.5,
                ]
            } else {
                [
                    if texture.scale[0] < 0.0 {
                        1.0 - uv[0]
                    } else {
                        uv[0]
                    },
                    if texture.scale[1] < 0.0 {
                        1.0 - uv[1]
                    } else {
                        uv[1]
                    },
                ]
            };
            mesh.texcoords.push(Some(mapped));
            mesh.colors.push(Some([1.0; 4]));
            mesh.joints.push(None);
            mesh.weights.push(None);
            for axis in 0..3 {
                mesh.bounds_min[axis] = mesh.bounds_min[axis].min([x, y, z][axis]);
                mesh.bounds_max[axis] = mesh.bounds_max[axis].max([x, y, z][axis]);
            }
        }
    }
    for row in 0..rows {
        for col in 0..N {
            let a = start + row * (N + 1) + col;
            for indices in [[a, a + N + 1, a + 1], [a + 1, a + N + 1, a + N + 2]] {
                mesh.indices.extend(indices);
                mesh.triangles.push(GlbTriangle {
                    indices,
                    material: Some(material),
                    mesh: Some(part),
                    mesh_node: None,
                });
            }
        }
    }
}
