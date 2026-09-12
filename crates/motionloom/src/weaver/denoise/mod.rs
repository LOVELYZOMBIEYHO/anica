// =========================================
// =========================================
// crates/motionloom/src/weaver/denoise/mod.rs

//! Optional OIDN C API adapter. No dependency on another application's renderer.
use crate::weaver::WeaverError;
use std::{
    ffi::{CStr, c_char, c_void},
    path::Path,
};
type Handle = *mut c_void;
type Release = unsafe extern "C" fn(Handle);
struct Owned {
    handle: Handle,
    release: Release,
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { (self.release)(self.handle) }
    }
}

/// The caller chooses a trusted native library. CPU-owned buffers remain alive
/// until synchronous execution completes and all filter/device handles release.
pub(crate) fn run(library: &Path, size: [u32; 2], film: &[f32]) -> Result<Vec<f32>, WeaverError> {
    let fail = |s: String| WeaverError::Gpu(format!("denoiser: {s}"));
    let mut color = Vec::new();
    let mut albedo = Vec::new();
    let mut normal = Vec::new();
    for f in film.chunks_exact(16) {
        let n = f[3].max(1.0);
        color.extend(f[..3].iter().map(|v| v / n));
        albedo.extend(f[8..11].iter().map(|v| v / n));
        let len = f[12..15]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        normal.extend(f[12..15].iter().map(|v| v / len));
    }
    let mut output = vec![0.0f32; color.len()];
    // ABI declarations follow OIDN's public C header. FLOAT3=3, CPU=1.
    unsafe {
        let lib = libloading::Library::new(library).map_err(|e| fail(e.to_string()))?;
        macro_rules! symbol {
            ($name:literal,$ty:ty) => {
                *lib.get::<$ty>(concat!($name, "\0").as_bytes())
                    .map_err(|e| fail(e.to_string()))?
            };
        }
        let new_device = symbol!("oidnNewDevice", unsafe extern "C" fn(i32) -> Handle);
        let release_device = symbol!("oidnReleaseDevice", Release);
        let commit_device = symbol!("oidnCommitDevice", Release);
        let error = symbol!(
            "oidnGetDeviceError",
            unsafe extern "C" fn(Handle, *mut *const c_char) -> i32
        );
        let new_filter = symbol!(
            "oidnNewFilter",
            unsafe extern "C" fn(Handle, *const c_char) -> Handle
        );
        let release_filter = symbol!("oidnReleaseFilter", Release);
        let commit_filter = symbol!("oidnCommitFilter", Release);
        let execute = symbol!("oidnExecuteFilter", Release);
        let set_image = symbol!(
            "oidnSetSharedFilterImage",
            unsafe extern "C" fn(
                Handle,
                *const c_char,
                *mut c_void,
                i32,
                usize,
                usize,
                usize,
                usize,
                usize,
            )
        );
        let set_bool = symbol!(
            "oidnSetFilterBool",
            unsafe extern "C" fn(Handle, *const c_char, bool)
        );
        let check = |handle| -> Result<(), WeaverError> {
            let mut message = std::ptr::null();
            if error(handle, &mut message) != 0 {
                return Err(fail(if message.is_null() {
                    "unknown error".into()
                } else {
                    CStr::from_ptr(message).to_string_lossy().into_owned()
                }));
            }
            Ok(())
        };
        let device = new_device(1);
        if device.is_null() {
            check(device)?;
            return Err(fail("null device".into()));
        }
        let device = Owned {
            handle: device,
            release: release_device,
        };
        check(device.handle)?;
        commit_device(device.handle);
        check(device.handle)?;
        let filter = new_filter(device.handle, c"RT".as_ptr());
        if filter.is_null() {
            check(device.handle)?;
            return Err(fail("null filter".into()));
        }
        let filter = Owned {
            handle: filter,
            release: release_filter,
        };
        check(device.handle)?;
        for (name, buffer) in [
            (c"color", &mut color),
            (c"albedo", &mut albedo),
            (c"normal", &mut normal),
            (c"output", &mut output),
        ] {
            set_image(
                filter.handle,
                name.as_ptr(),
                buffer.as_mut_ptr().cast(),
                3,
                size[0] as usize,
                size[1] as usize,
                0,
                0,
                0,
            );
        }
        set_bool(filter.handle, c"hdr".as_ptr(), true);
        // Auxiliary passes are noisy; never claim cleanAux without prefiltering.
        set_bool(filter.handle, c"cleanAux".as_ptr(), false);
        commit_filter(filter.handle);
        check(device.handle)?;
        execute(filter.handle);
        check(device.handle)?;
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Err(fail("non-finite output".into()));
    }
    Ok(output)
}
