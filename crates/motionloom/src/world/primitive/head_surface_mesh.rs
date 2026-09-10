// =========================================
// =========================================
// crates/motionloom/src/world/primitive/head_surface_mesh.rs

use std::f32::consts::{PI, TAU};

use crate::dsl::{FaceLayoutNode, HeadFeatureNode, HeadMorphNode, HeadShapeNode};

use super::MeshBuilder;

/// Tessellate a closed head cage and deform its shared surface with semantic
/// fields. Features remain continuous instead of attaching visible primitives.
#[allow(clippy::too_many_arguments)]
pub(super) fn generate(
    builder: &mut MeshBuilder,
    segments: u32,
    rings: u32,
    shape: &HeadShapeNode,
    face_layout: Option<&FaceLayoutNode>,
    features: &[HeadFeatureNode],
    morph: &HeadMorphNode,
) {
    let segments = segments.max(12) as usize;
    let rings = rings.max(8) as usize;
    let half = [
        shape.size[0] * 0.5 * morph.head_width,
        shape.size[1] * 0.5 * morph.head_height,
        shape.size[2] * 0.5 * morph.head_depth,
    ];
    let automatic = face_layout
        .map(|layout| layout_features(layout, morph))
        .unwrap_or_default();
    let mut vertices = Vec::with_capacity((rings + 1) * (segments + 1));
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = v * PI;
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let theta = u * TAU;
            let unit = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            let position = surface_position(unit, half, shape, morph, &automatic, features);
            let normal = ellipsoid_normal(position, half);
            vertices.push(builder.vertex(position, normal, [u, v]));
        }
    }
    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = vertices[ring * stride + segment];
            let b = vertices[(ring + 1) * stride + segment];
            let c = vertices[(ring + 1) * stride + segment + 1];
            let d = vertices[ring * stride + segment + 1];
            builder.triangle(a, c, b);
            builder.triangle(a, d, c);
        }
    }
    // Semantic fields displace the shared cage after the analytic ellipsoid
    // normal is known. Rebuild smooth normals from the deformed surface so
    // sockets, lips, cheek planes, and similar relief respond to lighting.
    builder.recalculate_face_normals();
    builder.smooth_normals(180.0, 1.0, false);
}

fn surface_position(
    unit: [f32; 3],
    half: [f32; 3],
    shape: &HeadShapeNode,
    morph: &HeadMorphNode,
    automatic: &[HeadFeatureNode],
    features: &[HeadFeatureNode],
) -> [f32; 3] {
    let y = unit[1];
    let front = unit[2].max(0.0);
    let forehead_weight = smoothstep(0.15, 0.78, y);
    let cheek_weight = (1.0 - (y / 0.56).abs()).clamp(0.0, 1.0) * front.powf(0.35);
    let jaw_weight = smoothstep(-0.05, -0.82, y);
    let mut width = 1.0;
    width *= mix(1.0, shape.forehead, forehead_weight);
    width *= mix(1.0, shape.cheek_width * morph.face_width, cheek_weight);
    width *= mix(
        1.0,
        shape.jaw_width * morph.jaw_width,
        jaw_weight * (0.65 + shape.chin_roundness * 0.35),
    );
    let face_height = mix(1.0, morph.face_height, front);
    let mut position = [
        unit[0] * half[0] * width,
        unit[1] * half[1] * face_height,
        unit[2] * half[2],
    ];
    let chin_weight = smoothstep(-0.45, -0.94, y) * front.powf(1.5);
    position[1] -= shape.chin_length * half[1] * chin_weight;

    let normal = ellipsoid_normal(position, half);
    for feature in automatic.iter().chain(features) {
        apply_feature(
            &mut position,
            normal,
            unit,
            half,
            feature,
            morph.feature_scale,
        );
        if feature.mirror_x && feature.center[0].abs() > 1.0e-5 {
            let mut mirrored = feature.clone();
            mirrored.center[0] = -mirrored.center[0];
            apply_feature(
                &mut position,
                normal,
                unit,
                half,
                &mirrored,
                morph.feature_scale,
            );
        }
    }
    position
}

fn apply_feature(
    position: &mut [f32; 3],
    normal: [f32; 3],
    unit: [f32; 3],
    half: [f32; 3],
    feature: &HeadFeatureNode,
    feature_scale: f32,
) {
    let distance = (0..3)
        .map(|axis| ((unit[axis] - feature.center[axis]) / feature.size[axis]).powi(2))
        .sum::<f32>()
        .sqrt();
    if distance >= 1.0 {
        return;
    }
    let base = 1.0 - distance;
    let weight = match feature.falloff.as_str() {
        "linear" => base,
        "sharp" => base.powi(4),
        _ => base * base * (3.0 - 2.0 * base),
    };
    let scale = half[0].min(half[1]).min(half[2]);
    let amount = feature.amount * feature_scale * scale * weight;
    for axis in 0..3 {
        position[axis] +=
            normal[axis] * amount + feature.offset[axis] * feature_scale * scale * weight;
    }
}

fn layout_features(layout: &FaceLayoutNode, morph: &HeadMorphNode) -> Vec<HeadFeatureNode> {
    let mut fields = Vec::new();
    let mut add = |id: &str, kind: &str, p: [f32; 3], size, amount, falloff: &str| {
        fields.push(HeadFeatureNode {
            id: id.into(),
            kind: kind.into(),
            center: [p[0] * morph.face_width, p[1] * morph.face_height, p[2]],
            size,
            amount,
            offset: [0.0; 3],
            falloff: falloff.into(),
            mirror_x: false,
        });
    };
    for eye in &layout.eyes {
        // Eye Z moves the generated eyeball component; socket relief belongs to
        // the head surface and remains anchored to the facial layout X/Y.
        let p = [eye.position[0], eye.position[1], 0.90];
        add(
            &eye.id,
            "socket",
            p,
            [eye.width.max(0.12) * 0.65, eye.opening.max(0.04), 0.24],
            -eye.socket_depth.abs(),
            "smooth",
        );
        add(
            &format!("{}_fold", eye.id),
            "eyelid",
            [p[0], p[1] + eye.opening * 0.34, p[2] + 0.02],
            [
                eye.width.max(0.12) * 0.62,
                eye.opening.max(0.04) * 0.65,
                0.18,
            ],
            0.03 * 0.18,
            "smooth",
        );
    }
    for nose in &layout.noses {
        add(
            &nose.id,
            "nose",
            [nose.position[0], nose.position[1], 0.96 + nose.position[2]],
            [nose.width.max(0.04), nose.length.max(0.08), 0.20],
            nose.projection,
            "sharp",
        );
    }
    for mouth in &layout.mouths {
        add(
            &mouth.id,
            "mouth",
            [
                mouth.position[0],
                mouth.position[1],
                0.94 + mouth.position[2],
            ],
            [mouth.width.max(0.08), mouth.opening.max(0.02) + 0.04, 0.18],
            -mouth.opening.max(0.008) * 0.5,
            "sharp",
        );
        add(
            &format!("{}_lip", mouth.id),
            "lip",
            [
                mouth.position[0],
                mouth.position[1],
                0.96 + mouth.position[2],
            ],
            [
                mouth.width.max(0.08),
                (mouth.opening + 0.08).max(0.08),
                0.18,
            ],
            (mouth.upper_lip + mouth.lower_lip) * 0.11,
            "smooth",
        );
        add(
            &format!("{}_muzzle", mouth.id),
            "muzzle",
            [
                mouth.position[0],
                mouth.position[1] + 0.13,
                0.82 + mouth.position[2],
            ],
            [mouth.muzzle_width.max(0.08), 0.34, 0.34],
            mouth.muzzle_length * morph.muzzle_length,
            "smooth",
        );
    }
    for ear in &layout.ears {
        add(
            &ear.id,
            "ear",
            ear.position,
            [ear.width.max(0.04), ear.height.max(0.08), 0.30],
            ear.depth,
            "smooth",
        );
    }
    fields
}

fn ellipsoid_normal(position: [f32; 3], half: [f32; 3]) -> [f32; 3] {
    normalize([
        position[0] / (half[0] * half[0]),
        position[1] / (half[1] * half[1]),
        position[2] / (half[2] * half[2]),
    ])
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = value.iter().map(|part| part * part).sum::<f32>().sqrt();
    if length > 1.0e-8 {
        value.map(|part| part / length)
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let denominator = edge1 - edge0;
    let amount = if denominator.abs() > 1.0e-8 {
        ((value - edge0) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    amount * amount * (3.0 - 2.0 * amount)
}

fn mix(start: f32, end: f32, amount: f32) -> f32 {
    start + (end - start) * amount
}
