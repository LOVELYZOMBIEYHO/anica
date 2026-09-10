// =========================================
// =========================================
// crates/motionloom/src/world/primitive/profiled_surface.rs

/// One horizontal cross-section shared by semantic cage generators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CrossSection {
    pub at: f64,
    pub width: f64,
    pub front: f64,
    pub back: f64,
}

/// Sample an ordered section stack without assigning anatomy-specific meaning.
pub(super) fn sample_catmull_rom(
    source: &[CrossSection],
    samples_per_section: u32,
) -> Vec<CrossSection> {
    let mut result = Vec::new();
    for index in 0..source.len() - 1 {
        let a = source[index.saturating_sub(1)];
        let b = source[index];
        let c = source[index + 1];
        let d = source[(index + 2).min(source.len() - 1)];
        for sample in 0..samples_per_section {
            let t = sample as f64 / samples_per_section as f64;
            result.push(CrossSection {
                at: b.at + t * (c.at - b.at),
                width: catmull(a.width, b.width, c.width, d.width, t),
                front: catmull(a.front, b.front, c.front, d.front, t),
                back: catmull(a.back, b.back, c.back, d.back, t),
            });
        }
    }
    result.push(*source.last().expect("validated section stack"));
    result
}

/// Find a rectangular patch on the front half of a ring stack.
pub(super) fn front_patch_bounds(
    rows: &[CrossSection],
    positions: &[[f64; 3]],
    segments: usize,
    center: [f64; 2],
    radius: [f64; 2],
) -> Option<(usize, usize, usize, usize)> {
    let selected_rows = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| center[1] - radius[1] <= row.at && row.at <= center[1] + radius[1])
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let middle = rows
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1.at - center[1])
                .abs()
                .total_cmp(&(b.1.at - center[1]).abs())
        })?
        .0;
    let selected_columns = (1..segments / 2)
        .filter(|&segment| {
            let x = positions[middle * segments + segment][0];
            center[0] - radius[0] <= x && x <= center[0] + radius[0]
        })
        .collect::<Vec<_>>();
    Some((
        *selected_rows.first()?,
        *selected_rows.last()?,
        *selected_columns.first()?,
        *selected_columns.last()?,
    ))
}

fn catmull(a: f64, b: f64, c: f64, d: f64, t: f64) -> f64 {
    0.5 * ((2.0 * b)
        + (-a + c) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t
        + (-a + 3.0 * b - 3.0 * c + d) * t * t * t)
}
