// =========================================
// =========================================
// crates/motionloom/src/weaver/camera/mod.rs

use crate::weaver::{RenderJob, WeaverError};
use crate::world::WorldCamera;

fn number(s: &str) -> Result<f32, WeaverError> {
    s.parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| WeaverError::Scene(format!("unevaluated camera value: {s}")))
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.map(|x| x / n)
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Match the existing vertical FOV and orbit camera, then derive a physical lens.
pub(crate) fn configure(
    p: &mut crate::weaver::backend::wgpu::CameraParams,
    c: &WorldCamera,
    job: &RenderJob,
) -> Result<(), WeaverError> {
    if c.projection != crate::world::WorldCameraProjection::Perspective {
        return Err(WeaverError::Unsupported("orthographic camera".into()));
    }
    let target = [
        number(&c.target_x)?,
        number(&c.target_y)?,
        number(&c.target_z)?,
    ];
    let yaw = number(&c.yaw)?.to_radians();
    let pitch = number(&c.pitch)?.to_radians();
    let distance = number(&c.distance)?;
    let fov = number(&c.fov)?;
    if distance <= 0.0 || !(0.0..180.0).contains(&fov) || fov == 0.0 {
        return Err(WeaverError::Invalid(
            "camera distance/FOV outside physical range".into(),
        ));
    }
    let offset = [
        yaw.sin() * pitch.cos() * distance,
        pitch.sin() * distance,
        yaw.cos() * pitch.cos() * distance,
    ];
    let origin = std::array::from_fn::<_, 3, _>(|i| target[i] + offset[i]);
    let forward = normalize(offset.map(|v| -v));
    let up = normalize([number(&c.up_x)?, number(&c.up_y)?, number(&c.up_z)?]);
    let right = normalize(cross(forward, up));
    let up = normalize(cross(right, forward));
    let roll = number(&c.roll)?.to_radians();
    let r = std::array::from_fn::<_, 3, _>(|i| right[i] * roll.cos() + up[i] * roll.sin());
    let u = std::array::from_fn::<_, 3, _>(|i| up[i] * roll.cos() - right[i] * roll.sin());
    let tangent = (fov.to_radians() * 0.5).tan();
    let aspect = job.resolution[0] as f32 / job.resolution[1] as f32;
    let lens_radius = if job.lens.enabled {
        job.lens.sensor_width_mm * 0.001 / (2.0 * tangent * aspect) / (2.0 * job.lens.f_stop)
    } else {
        0.0
    };
    p[2] = [origin[0], origin[1], origin[2], 0.0];
    p[3] = [forward[0], forward[1], forward[2], tangent];
    p[4] = [r[0], r[1], r[2], aspect];
    p[5] = [u[0], u[1], u[2], lens_radius];
    p[6] = [
        job.lens.focus_distance,
        job.lens.aperture_blades as f32,
        0.0,
        0.0,
    ];

    if p[2..7].iter().flatten().any(|v| !v.is_finite()) {
        return Err(WeaverError::Invalid("degenerate camera basis".into()));
    }
    Ok(())
}
