// =========================================
// =========================================
// crates/motionloom/src/world/primitive/facial_cage_mesh.rs

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

use super::profiled_surface::{CrossSection, front_patch_bounds, sample_catmull_rom};
use crate::dsl::{ControlCageNode, FaceLayoutNode, FacialCageNode, HeadDomeNode, HeadSectionNode};

#[derive(Clone, Copy)]
struct RingStep {
    blend: f64,
    scale: f64,
    depth: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PatchKind {
    Eye,
    Mouth,
}

struct Patch {
    bounds: (usize, usize, usize, usize),
    center: [f64; 2],
    radii: [f64; 2],
    outer: [f64; 2],
    kind: PatchKind,
    eye: Option<crate::dsl::EyeNode>,
    offset_z: f64,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum LayoutError {
    #[error(
        "facial component opening is outside the sampled head surface or too small for the cage resolution"
    )]
    Bounds,
    #[error(
        "facial component socket patches overlap; reduce socket dimensions or separate the components"
    )]
    Overlap,
}

/// Reject invalid patches before the generator allocates and connects their rings.
pub(crate) fn validate_layout(
    settings: &FacialCageNode,
    profile: &[HeadSectionNode],
    dome: Option<&HeadDomeNode>,
    face: &FaceLayoutNode,
) -> Result<(), LayoutError> {
    let rows = sampled_rows(settings, profile, dome);
    let n = settings.segments as usize;
    let positions: Vec<_> = rows
        .iter()
        .flat_map(|row| {
            (0..n).map(move |i| {
                let source = i as f64 * settings.profile_segments as f64 / n as f64;
                let low = source.floor();
                let fraction = source - low;
                let cosine = (1.0 - fraction)
                    * (TAU * low / settings.profile_segments as f64).cos()
                    + fraction * (TAU * (low + 1.0) / settings.profile_segments as f64).cos();
                [row.width * 0.5 * cosine, row.at, 0.0]
            })
        })
        .collect();
    let mut bounds = Vec::new();
    for (center, radius) in face
        .eyes
        .iter()
        .map(|e| {
            (
                [e.position[0] as f64, e.position[1] as f64],
                [e.socket_width as f64 * 0.5, e.socket_height as f64 * 0.5],
            )
        })
        .chain(face.mouths.iter().map(|m| {
            (
                [m.position[0] as f64, m.position[1] as f64],
                [
                    (m.width as f64 * 0.5 * 3.488372).max(0.08),
                    (m.opening as f64 * 0.5 * 14.285714).max(0.04),
                ],
            )
        }))
    {
        let b =
            front_patch_bounds(&rows, &positions, n, center, radius).ok_or(LayoutError::Bounds)?;
        if b.0 >= b.1 || b.2 >= b.3 {
            return Err(LayoutError::Bounds);
        }
        if bounds
            .iter()
            .any(|&(a, c, d, e)| b.0 < c && a < b.1 && b.2 < e && d < b.3)
        {
            return Err(LayoutError::Overlap);
        }
        bounds.push(b);
    }
    Ok(())
}

/// Build the deterministic semantic cage used by both native and WASM renderers.
pub(super) fn build(
    settings: &FacialCageNode,
    profile: &[HeadSectionNode],
    dome: Option<&HeadDomeNode>,
    face: &FaceLayoutNode,
) -> ControlCageNode {
    let rows = sampled_rows(settings, profile, dome);
    let n = settings.segments as usize;
    let mut positions = Vec::<[f64; 3]>::new();
    let mut pinned = Vec::<bool>::new();
    let mut faces = Vec::<Vec<u32>>::new();
    let surface = |x: f64, y: f64| face_surface(&rows, face, x, y);

    for row in &rows {
        for segment in 0..n {
            let t = TAU * segment as f64 / n as f64;
            let source = segment as f64 * settings.profile_segments as f64 / n as f64;
            let low = source.floor();
            let fraction = source - low;
            let a = TAU * low / settings.profile_segments as f64;
            let b = TAU * (low + 1.0) / settings.profile_segments as f64;
            let cosine = (1.0 - fraction) * a.cos() + fraction * b.cos();
            let sine = (1.0 - fraction) * a.sin() + fraction * b.sin();
            let x = row.width * 0.5 * cosine;
            let mut z = (row.front + row.back) * 0.5 + (row.front - row.back) * 0.5 * sine;
            let editable_nose = t.sin() > 0.0
                && face.noses.iter().any(|nose| {
                    (x - nose.position[0] as f64).abs() < nose.width as f64 * 1.4
                        && y_in_nose(row.at, nose)
                });
            if editable_nose {
                z = surface(x, row.at);
            }
            let editable_nose = t.sin() > 0.0
                && face.noses.iter().any(|nose| {
                    (x - nose.position[0] as f64).abs() < nose.width as f64 * 1.4
                        && row.at > nose.position[1] as f64 + 0.25 - nose.length as f64 * 1.93
                        && row.at < nose.position[1] as f64 + 0.25 - nose.length as f64 * 0.04
                });
            positions.push([x, row.at, z]);
            pinned.push(settings.preserve_profile && !editable_nose);
        }
    }

    let index = |row: usize, segment: usize| (row * n + segment % n) as u32;
    let mut patches = Vec::new();
    for eye in &face.eyes {
        let mut patch = patch_for(
            &rows,
            &positions,
            n,
            [eye.position[0] as f64, eye.position[1] as f64],
            [
                eye.socket_width as f64 * 0.5,
                eye.socket_height as f64 * 0.5,
            ],
            [eye.width as f64 * 0.5, eye.opening as f64 * 0.5],
            PatchKind::Eye,
        );
        patch.eye = Some(eye.clone());
        patches.push(patch);
    }
    for mouth in &face.mouths {
        let inner = [mouth.width as f64 * 0.5, mouth.opening as f64 * 0.5];
        let mut patch = patch_for(
            &rows,
            &positions,
            n,
            [mouth.position[0] as f64, mouth.position[1] as f64],
            [
                (inner[0] * 3.488372).max(0.08),
                (inner[1] * 14.285714).max(0.04),
            ],
            inner,
            PatchKind::Mouth,
        );
        patch.offset_z = mouth.position[2] as f64;
        patches.push(patch);
    }

    for row in 0..rows.len() - 1 {
        for segment in 0..n {
            let removed = patches.iter().any(|patch| {
                let (a, b, c, d) = patch.bounds;
                a <= row && row < b && c <= segment && segment < d
            });
            if !removed {
                faces.push(vec![
                    index(row, segment),
                    index(row + 1, segment),
                    index(row + 1, segment + 1),
                    index(row, segment + 1),
                ]);
            }
        }
    }

    for patch in &patches {
        let (a, b, c, d) = patch.bounds;
        let boundary = (c + 1..=d)
            .rev()
            .map(|s| index(a, s))
            .chain((a..b).map(|r| index(r, c)))
            .chain((c..d).map(|s| index(b, s)))
            .chain((a + 1..=b).rev().map(|r| index(r, d)))
            .collect::<Vec<_>>();
        let outer = boundary
            .iter()
            .map(|&i| positions[i as usize])
            .collect::<Vec<_>>();
        let angles = outer
            .iter()
            .map(|p| {
                ((p[1] - patch.center[1]) / patch_outer(patch)[1])
                    .atan2((p[0] - patch.center[0]) / patch_outer(patch)[0])
            })
            .collect::<Vec<_>>();
        let steps = resampled_profile(
            if patch.kind == PatchKind::Eye {
                eye_profile()
            } else {
                mouth_profile()
            },
            if patch.kind == PatchKind::Eye {
                settings.orbital_rings
            } else {
                settings.mouth_rings
            },
        );
        let mut previous = boundary;
        for step in steps {
            let mut ring = Vec::with_capacity(outer.len());
            for (point, angle) in outer.iter().zip(&angles) {
                let target_x = patch.center[0] + patch.radii[0] * step.scale * angle.cos();
                let taper = if patch.kind == PatchKind::Eye {
                    angle.sin().abs().powf(0.20)
                } else {
                    1.0
                };
                let target_y = patch.center[1] + patch.radii[1] * step.scale * angle.sin() * taper;
                let (target_x, target_y) = if let Some(eye) = &patch.eye {
                    let (sin, cos) = (eye.tilt as f64).to_radians().sin_cos();
                    let dx = target_x - patch.center[0];
                    let dy = target_y - patch.center[1];
                    (
                        patch.center[0] + dx * cos - dy * sin,
                        patch.center[1] + dx * sin + dy * cos,
                    )
                } else {
                    (target_x, target_y)
                };
                let x = point[0] * (1.0 - step.blend) + target_x * step.blend;
                let y = point[1] * (1.0 - step.blend) + target_y * step.blend;
                let profile_relief =
                    if patch.kind == PatchKind::Eye && angle.sin() > 0.0 && step.depth > 0.0 {
                        step.depth * 1.15
                    } else {
                        step.depth
                    };
                // Eye profile depths were authored around the legacy 0.139 default. Scale the
                // complete orbital profile, rather than only its centre vertex, so
                // `eyeSocketDepth` actually controls the visible socket depth.
                let relief = if patch.kind == PatchKind::Eye {
                    profile_relief * patch.eye.as_ref().unwrap().socket_depth as f64 / 0.139
                } else {
                    profile_relief
                };
                ring.push(positions.len() as u32);
                positions.push([x, y, surface(x, y) + relief + patch.offset_z]);
                pinned.push(false);
            }
            for j in 0..ring.len() {
                let k = (j + 1) % ring.len();
                faces.push(vec![previous[j], previous[k], ring[k], ring[j]]);
            }
            previous = ring;
        }
        let center_depth = if patch.kind == PatchKind::Eye {
            patch.eye.as_ref().unwrap().socket_depth as f64 + 0.004
        } else {
            0.05
        };
        let center = positions.len() as u32;
        positions.push([
            patch.center[0],
            patch.center[1],
            surface(patch.center[0], patch.center[1]) - center_depth + patch.offset_z,
        ]);
        pinned.push(false);
        for j in 0..previous.len() {
            faces.push(vec![
                previous[j],
                previous[(j + 1) % previous.len()],
                center,
            ]);
        }
    }

    for (row, top) in [(0, false), (rows.len() - 1, true)] {
        let center = positions.len() as u32;
        positions.push([0.0, rows[row].at, (rows[row].front + rows[row].back) * 0.5]);
        pinned.push(true);
        for segment in 0..n {
            faces.push(if top {
                vec![center, index(row, segment + 1), index(row, segment)]
            } else {
                vec![center, index(row, segment), index(row, segment + 1)]
            });
        }
    }

    // Ear relief attaches to the lateral surface without imposing a two-ear limit.
    for (position, pin) in positions.iter_mut().zip(&mut pinned) {
        for ear in &face.ears {
            let side = if ear.position[0] < 0.0 { -1.0 } else { 1.0 };
            if position[0] * side <= 0.0 {
                continue;
            }
            let weight = bump(
                position[2],
                position[1],
                ear.position[2] as f64,
                ear.position[1] as f64,
                ear.width as f64,
                ear.height as f64 * 0.5,
            );
            if weight > 0.0 {
                position[0] += side * ear.depth as f64 * weight;
                *pin = false;
            }
        }
    }
    compact(settings, &rows, positions, pinned, faces)
}

pub(super) fn sampled_rows(
    settings: &FacialCageNode,
    profile: &[HeadSectionNode],
    dome: Option<&HeadDomeNode>,
) -> Vec<CrossSection> {
    let source = profile
        .iter()
        .map(|p| CrossSection {
            at: p.at as f64,
            width: p.width as f64,
            front: p.front_depth as f64,
            back: p.back_depth as f64,
        })
        .collect::<Vec<_>>();
    let mut rows = sample_catmull_rom(&source, settings.samples_per_section);
    if let Some(dome) = dome {
        let dome_width = source
            .iter()
            .filter(|row| row.at <= dome.start as f64)
            .max_by(|a, b| a.at.total_cmp(&b.at))
            .map(|row| row.width)
            .unwrap_or_else(|| source.last().unwrap().width);
        rows.retain(|row| row.at < dome.start as f64);
        for sample in 0..=dome.samples {
            let t = sample as f64 / dome.samples as f64;
            let s = (0.00002_f64.max(1.0 - t * t)).sqrt();
            rows.push(CrossSection {
                at: dome.start as f64 + (dome.top - dome.start) as f64 * t,
                width: dome_width * s,
                front: dome.center_depth as f64 + dome.front_radius as f64 * s,
                back: dome.center_depth as f64 - dome.back_radius as f64 * s,
            });
        }
    }
    rows
}

pub(super) fn face_surface(rows: &[CrossSection], face: &FaceLayoutNode, x: f64, y: f64) -> f64 {
    let base = rows
        .windows(2)
        .find(|pair| pair[0].at <= y && y <= pair[1].at)
        .map(|pair| {
            let t = (y - pair[0].at) / (pair[1].at - pair[0].at);
            let width = pair[0].width + t * (pair[1].width - pair[0].width);
            let front = pair[0].front + t * (pair[1].front - pair[0].front);
            let back = pair[0].back + t * (pair[1].back - pair[0].back);
            (front + back) * 0.5
                + (front - back) * 0.5 * (0.0_f64.max(1.0 - (2.0 * x / width).powi(2))).sqrt()
        })
        .unwrap_or(0.0);
    base + face
        .noses
        .iter()
        .map(|nose| nose_relief(nose, x - nose.position[0] as f64, y))
        .sum::<f64>()
}

fn nose_relief(nose: &crate::dsl::NoseNode, x: f64, y: f64) -> f64 {
    let bridge = nose.projection as f64
        * 0.3666666667
        * bump(
            x,
            y,
            0.0,
            nose.position[1] as f64 + 0.078,
            nose.width as f64,
            nose.length as f64,
        );
    let tip = nose.projection as f64
        * bump(
            x,
            y,
            0.0,
            nose.position[1] as f64,
            nose.width as f64 * 0.85,
            nose.length as f64 * 0.3777777778,
        );
    let wings = nose.projection as f64 / 6.0
        * (bump(
            x,
            y,
            -nose.width as f64 * 0.52,
            nose.position[1] as f64 - 0.026,
            nose.width as f64 * 0.5,
            nose.length as f64 * 0.2111111111,
        ) + bump(
            x,
            y,
            nose.width as f64 * 0.52,
            nose.position[1] as f64 - 0.026,
            nose.width as f64 * 0.5,
            nose.length as f64 * 0.2111111111,
        ));
    bridge
        + tip
        + wings
        + nose.position[2] as f64
            * bump(
                x,
                y,
                0.0,
                nose.position[1] as f64,
                nose.width as f64,
                nose.length as f64,
            )
}

fn bump(x: f64, y: f64, cx: f64, cy: f64, rx: f64, ry: f64) -> f64 {
    let d = ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2);
    0.0_f64.max(1.0 - d).powi(3)
}

fn y_in_nose(y: f64, nose: &crate::dsl::NoseNode) -> bool {
    y > nose.position[1] as f64 + 0.25 - nose.length as f64 * 1.9833333333
        && y < nose.position[1] as f64 + 0.25 + nose.length as f64 * 0.0722222222
}

fn patch_for(
    rows: &[CrossSection],
    positions: &[[f64; 3]],
    segments: usize,
    center: [f64; 2],
    outer: [f64; 2],
    inner: [f64; 2],
    kind: PatchKind,
) -> Patch {
    Patch {
        bounds: front_patch_bounds(rows, positions, segments, center, outer)
            .expect("parser validated patch bounds"),
        center,
        radii: inner,
        outer,
        kind,
        eye: None,
        offset_z: 0.0,
    }
}

fn patch_outer(patch: &Patch) -> [f64; 2] {
    patch.outer
}

fn eye_profile() -> &'static [RingStep] {
    &[
        RingStep {
            blend: 0.28,
            scale: 1.30,
            depth: 0.0,
        },
        RingStep {
            blend: 0.58,
            scale: 1.25,
            depth: 0.002,
        },
        RingStep {
            blend: 0.83,
            scale: 1.17,
            depth: 0.010,
        },
        RingStep {
            blend: 1.0,
            scale: 1.07,
            depth: 0.018,
        },
        RingStep {
            blend: 1.0,
            scale: 1.0,
            depth: 0.012,
        },
        RingStep {
            blend: 1.0,
            scale: 0.965,
            depth: -0.008,
        },
        RingStep {
            blend: 1.0,
            scale: 0.94,
            depth: -0.046,
        },
        RingStep {
            blend: 1.0,
            scale: 0.85,
            depth: -0.092,
        },
        RingStep {
            blend: 1.0,
            scale: 0.58,
            depth: -0.125,
        },
        RingStep {
            blend: 1.0,
            scale: 0.25,
            depth: -0.139,
        },
    ]
}

fn mouth_profile() -> &'static [RingStep] {
    &[
        RingStep {
            blend: 0.3,
            scale: 1.6,
            depth: 0.0,
        },
        RingStep {
            blend: 0.6,
            scale: 1.45,
            depth: 0.004,
        },
        RingStep {
            blend: 0.85,
            scale: 1.25,
            depth: 0.010,
        },
        RingStep {
            blend: 1.0,
            scale: 1.1,
            depth: 0.016,
        },
        RingStep {
            blend: 1.0,
            scale: 1.0,
            depth: 0.012,
        },
        RingStep {
            blend: 1.0,
            scale: 0.96,
            depth: -0.006,
        },
        RingStep {
            blend: 1.0,
            scale: 0.85,
            depth: -0.025,
        },
        RingStep {
            blend: 1.0,
            scale: 0.45,
            depth: -0.045,
        },
    ]
}

fn resampled_profile(source: &[RingStep], count: u32) -> Vec<RingStep> {
    if count as usize == source.len() {
        return source.to_vec();
    }
    (0..count)
        .map(|i| {
            let x = i as f64 * (source.len() - 1) as f64 / (count - 1) as f64;
            let a = x.floor() as usize;
            let b = (a + 1).min(source.len() - 1);
            let t = x - a as f64;
            RingStep {
                blend: source[a].blend + t * (source[b].blend - source[a].blend),
                scale: source[a].scale + t * (source[b].scale - source[a].scale),
                depth: source[a].depth + t * (source[b].depth - source[a].depth),
            }
        })
        .collect()
}

fn compact(
    settings: &FacialCageNode,
    rows: &[CrossSection],
    positions: Vec<[f64; 3]>,
    pinned: Vec<bool>,
    faces: Vec<Vec<u32>>,
) -> ControlCageNode {
    let used = faces
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mapping = used
        .iter()
        .enumerate()
        .map(|(next, old)| (*old, next as u32))
        .collect::<BTreeMap<_, _>>();
    let rounded = |v: f64| ((v * 1_000_000.0).round() / 1_000_000.0) as f32;
    let final_positions = used
        .iter()
        .map(|&i| positions[i as usize].map(rounded))
        .collect::<Vec<_>>();
    let uvs = final_positions
        .iter()
        .map(|p| {
            if settings.uv_mode == "fallbackxy" {
                [p[0], p[1]]
            } else {
                let min_y = rows.first().unwrap().at;
                let max_y = rows.last().unwrap().at;
                let v = ((max_y - p[1] as f64) / (max_y - min_y)).clamp(0.002, 0.998);
                let mid = rows
                    .windows(2)
                    .find(|pair| pair[0].at <= p[1] as f64 && p[1] as f64 <= pair[1].at)
                    .map(|pair| {
                        (pair[0].front + pair[0].back + pair[1].front + pair[1].back) * 0.25
                    })
                    .unwrap_or(0.0);
                if p[2] as f64 >= mid {
                    [0.25 + 0.18 * p[0], rounded(v)]
                } else {
                    [0.75, rounded(v)]
                }
            }
        })
        .collect();
    ControlCageNode {
        positions: final_positions,
        uvs,
        pinned: used.iter().map(|&i| pinned[i as usize]).collect(),
        faces: faces
            .into_iter()
            .map(|face| face.into_iter().map(|i| mapping[&i]).collect())
            .collect(),
        subdivision: settings.subdivision,
    }
}
