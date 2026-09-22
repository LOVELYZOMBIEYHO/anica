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
type NewDevice = unsafe extern "C" fn(i32) -> Handle;
type GetError = unsafe extern "C" fn(Handle, *mut *const c_char) -> i32;
type NewFilter = unsafe extern "C" fn(Handle, *const c_char) -> Handle;
type SetImage = unsafe extern "C" fn(
    Handle,
    *const c_char,
    *mut c_void,
    i32,
    usize,
    usize,
    usize,
    usize,
    usize,
);
type SetBool = unsafe extern "C" fn(Handle, *const c_char, bool);

struct Owned {
    handle: Handle,
    release: Release,
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { (self.release)(self.handle) }
    }
}

/// Cached OIDN device and entry points.
///
/// The one-shot `run` helper reloads the library for every call, which is fine
/// for a final frame. A preview keeps this handle alive so a device is created
/// once and reused for the whole session.
pub(crate) struct Denoiser {
    // Keeps the symbols below valid for the lifetime of the handle.
    _library: libloading::Library,
    device: Owned,
    error: GetError,
    new_filter: NewFilter,
    release_filter: Release,
    commit_filter: Release,
    execute: Release,
    set_image: SetImage,
    set_bool: SetBool,
}

impl Denoiser {
    /// The caller chooses a trusted native library.
    pub(crate) fn new(library: &Path) -> Result<Self, WeaverError> {
        let fail = |s: String| WeaverError::Gpu(format!("denoiser: {s}"));
        unsafe {
            let lib = libloading::Library::new(library).map_err(|e| fail(e.to_string()))?;
            macro_rules! symbol {
                ($name:literal,$ty:ty) => {
                    *lib.get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|e| fail(e.to_string()))?
                };
            }
            let new_device = symbol!("oidnNewDevice", NewDevice);
            let release_device = symbol!("oidnReleaseDevice", Release);
            let commit_device = symbol!("oidnCommitDevice", Release);
            let error = symbol!("oidnGetDeviceError", GetError);
            let new_filter = symbol!("oidnNewFilter", NewFilter);
            let release_filter = symbol!("oidnReleaseFilter", Release);
            let commit_filter = symbol!("oidnCommitFilter", Release);
            let execute = symbol!("oidnExecuteFilter", Release);
            let set_image = symbol!("oidnSetSharedFilterImage", SetImage);
            let set_bool = symbol!("oidnSetFilterBool", SetBool);
            // ABI declarations follow OIDN's public C header. CPU=1.
            let device = new_device(1);
            if device.is_null() {
                return Err(fail("null device".into()));
            }
            let device = Owned {
                handle: device,
                release: release_device,
            };
            let denoiser = Self {
                _library: lib,
                device,
                error,
                new_filter,
                release_filter,
                commit_filter,
                execute,
                set_image,
                set_bool,
            };
            denoiser.check(denoiser.device.handle)?;
            commit_device(denoiser.device.handle);
            denoiser.check(denoiser.device.handle)?;
            Ok(denoiser)
        }
    }

    fn check(&self, handle: Handle) -> Result<(), WeaverError> {
        let mut message = std::ptr::null();
        if unsafe { (self.error)(handle, &mut message) } != 0 {
            return Err(WeaverError::Gpu(format!(
                "denoiser: {}",
                if message.is_null() {
                    "unknown error".to_string()
                } else {
                    unsafe { CStr::from_ptr(message) }
                        .to_string_lossy()
                        .into_owned()
                }
            )));
        }
        Ok(())
    }

    /// Denoise one HDR film buffer in place. Reuses the warm device and only
    /// creates a fresh filter for the requested size.
    pub(crate) fn run(&self, size: [u32; 2], film: &[f32]) -> Result<Vec<f32>, WeaverError> {
        let mut color = Vec::new();
        let mut albedo = Vec::new();
        let mut normal = Vec::new();
        for f in film.chunks_exact(super::backend::wgpu::FILM_FLOATS_PER_PIXEL) {
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
        unsafe {
            let filter = (self.new_filter)(self.device.handle, c"RT".as_ptr());
            if filter.is_null() {
                self.check(self.device.handle)?;
                return Err(WeaverError::Gpu("denoiser: null filter".into()));
            }
            let filter = Owned {
                handle: filter,
                release: self.release_filter,
            };
            self.check(self.device.handle)?;
            for (name, buffer) in [
                (c"color", &mut color),
                (c"albedo", &mut albedo),
                (c"normal", &mut normal),
                (c"output", &mut output),
            ] {
                (self.set_image)(
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
            (self.set_bool)(filter.handle, c"hdr".as_ptr(), true);
            // Auxiliary passes are noisy; never claim cleanAux without prefiltering.
            (self.set_bool)(filter.handle, c"cleanAux".as_ptr(), false);
            (self.commit_filter)(filter.handle);
            self.check(self.device.handle)?;
            (self.execute)(filter.handle);
            self.check(self.device.handle)?;
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(WeaverError::Gpu("denoiser: non-finite output".into()));
        }
        Ok(output)
    }
}

/// The caller chooses a trusted native library. CPU-owned buffers remain alive
/// until synchronous execution completes and all filter/device handles release.
pub(crate) fn run(library: &Path, size: [u32; 2], film: &[f32]) -> Result<Vec<f32>, WeaverError> {
    Denoiser::new(library)?.run(size, film)
}
