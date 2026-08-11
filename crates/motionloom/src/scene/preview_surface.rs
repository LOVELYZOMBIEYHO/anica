// =========================================
// =========================================
// crates/motionloom/src/scene/preview_surface.rs

use std::sync::Arc;

/// GPU-native output of a scene frame render.
///
/// The texture is returned without CPU readback. Callers can sample or blit it
/// directly in their own wgpu pipeline, or extract the native backend handle
/// (e.g. `MTLTexture` on Metal) for zero-copy display.
#[derive(Debug, Clone)]
pub struct SceneGpuTexture {
    pub texture: Arc<wgpu::Texture>,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
}

/// Preferred engine-wide name for a frame that remains resident on the GPU.
///
/// `SceneGpuTexture` remains available as a compatibility name. New integrations
/// should use this neutral name because the same texture can contain 2D, 3D, or
/// mixed Scene output.
pub type GpuFrameTexture = SceneGpuTexture;

/// Pixel formats used by preview surfaces exposed to host applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePreviewPixelFormat {
    Rgba8Unorm,
    Bgra8Unorm,
}

/// Preferred preview backend for `render_frame_to_preview_surface`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePreviewBackend {
    /// Prefer the most efficient displayable surface and fall back safely.
    Auto,
    /// Return a wgpu texture without CPU readback.
    WgpuTexture,
    /// Return a platform display surface when the bridge is implemented.
    PlatformSurface,
    /// Return CPU BGRA bytes for compatibility with existing UI paths.
    CpuBgra,
}

/// Options for high-level preview surface rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenePreviewSurfaceOptions {
    pub backend: ScenePreviewBackend,
    pub preferred_format: ScenePreviewPixelFormat,
}

impl Default for ScenePreviewSurfaceOptions {
    fn default() -> Self {
        Self {
            backend: ScenePreviewBackend::Auto,
            preferred_format: ScenePreviewPixelFormat::Bgra8Unorm,
        }
    }
}

/// One plane of a Linux DMA-BUF descriptor.
///
/// The `fd` field is currently part of the public descriptor shape but its
/// ownership policy is not final. Real DMA-BUF export is not implemented yet;
/// when it is, this will likely switch to `OwnedFd` or document a clear
/// borrow/close contract.
#[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
#[derive(Debug, Clone)]
pub struct DmabufPlane {
    pub fd: std::os::fd::RawFd,
    pub stride: u32,
    pub offset: u32,
}

/// DXGI shared handle returned by a Windows D3D11 preview surface.
///
/// This is a legacy global shared handle obtained through
/// `IDXGIResource::GetSharedHandle`. The host must open it on a D3D10/11
/// device on the same adapter; the handle itself is owned by the OS and does
/// not need to be closed by the caller.
#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowsD3DSharedHandle(pub isize);

/// Windows D3D11 BGRA texture that backs a shared preview surface.
///
/// The `ID3D11Texture2D` is kept alive here so that the legacy DXGI shared
/// handle remains valid for as long as the host may hold it. Cloning this
/// struct clones the COM reference, not the underlying GPU resource.
#[cfg(target_os = "windows")]
#[derive(Clone)]
pub struct WindowsD3DSharedSurface {
    // Held only to keep the DXGI shared handle valid; never read directly.
    #[allow(dead_code)]
    texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
    pub width: u32,
    pub height: u32,
    pub format: ScenePreviewPixelFormat,
    pub handle: WindowsD3DSharedHandle,
    pub adapter_luid: i64,
    pub stride: u32,
}

#[cfg(target_os = "windows")]
impl std::fmt::Debug for WindowsD3DSharedSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WindowsD3DSharedSurface")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("handle", &self.handle)
            .field("adapter_luid", &self.adapter_luid)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

/// Platform-specific display surface for zero-copy preview bridges.
///
/// Each variant contains the real handle or descriptor a host application
/// needs to import and display the surface. Placeholder variants without
/// usable handles are intentionally not exposed.
#[derive(Debug, Clone)]
pub enum ScenePlatformPreviewSurface {
    /// macOS: a Metal-compatible `CVPixelBuffer` in BGRA format.
    #[cfg(target_os = "macos")]
    MacOs {
        surface: core_video::pixel_buffer::CVPixelBuffer,
        width: u32,
        height: u32,
        format: ScenePreviewPixelFormat,
    },
    /// Windows: a shareable D3D11 BGRA texture.
    ///
    /// The contained `WindowsD3DSharedSurface` keeps the texture alive so the
    /// legacy DXGI shared handle remains valid.
    #[cfg(target_os = "windows")]
    WindowsD3D(WindowsD3DSharedSurface),
    /// Linux: DMA-BUF descriptor for direct scanout/import.
    #[cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
    LinuxDmabuf {
        width: u32,
        height: u32,
        format: ScenePreviewPixelFormat,
        fourcc: u32,
        modifier: u64,
        planes: Vec<DmabufPlane>,
    },
}

/// High-level preview output for apps that want the fastest available path.
#[derive(Debug, Clone)]
pub enum ScenePreviewSurface {
    WgpuTexture(SceneGpuTexture),
    PlatformSurface(ScenePlatformPreviewSurface),
    CpuBgra {
        width: u32,
        height: u32,
        data: Arc<Vec<u8>>,
        format: ScenePreviewPixelFormat,
    },
}

// =========================================
// Windows D3D11 shared surface helpers.
// =========================================

#[cfg(target_os = "windows")]
pub(crate) struct WindowsD3D11Context {
    pub device: windows::Win32::Graphics::Direct3D11::ID3D11Device,
    pub context: windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
}

#[cfg(target_os = "windows")]
impl WindowsD3D11Context {
    pub fn new() -> Option<Self> {
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::Graphics::Direct3D::*;
        use windows::Win32::Graphics::Direct3D11::*;
        unsafe {
            let mut device = None;
            let mut context = None;
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .ok()?;
            Some(Self {
                device: device?,
                context: context?,
            })
        }
    }
}

#[cfg(target_os = "windows")]
fn create_windows_shared_bgra_texture(
    device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
    width: u32,
    height: u32,
) -> Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D> {
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: D3D11_RESOURCE_MISC_SHARED.0 as u32,
        ..Default::default()
    };
    unsafe {
        let mut texture = None;
        device
            .CreateTexture2D(&desc, None, Some(&mut texture))
            .ok()?;
        texture
    }
}

#[cfg(target_os = "windows")]
fn windows_texture_shared_handle(
    texture: &windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
) -> Option<isize> {
    use windows::Win32::Graphics::Dxgi::*;
    use windows::core::Interface;
    unsafe {
        let resource: IDXGIResource = texture.cast().ok()?;
        let handle = resource.GetSharedHandle().ok()?;
        Some(handle.0 as isize)
    }
}

#[cfg(target_os = "windows")]
fn upload_bgra_to_d3d11_texture(
    context: &windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
    texture: &windows::Win32::Graphics::Direct3D11::ID3D11Texture2D,
    width: u32,
    height: u32,
    data: &[u8],
) -> bool {
    let row_pitch = (width as usize).saturating_mul(4);
    let expected_len = row_pitch.saturating_mul(height as usize);
    if data.len() < expected_len {
        return false;
    }
    unsafe {
        context.UpdateSubresource(
            texture,
            0,
            None,
            data.as_ptr() as *const _,
            row_pitch as u32,
            row_pitch.saturating_mul(height as usize) as u32,
        );
        // Make the upload visible to any device that opens the shared handle.
        context.Flush();
    }
    true
}

#[cfg(target_os = "windows")]
fn d3d11_adapter_luid(device: &windows::Win32::Graphics::Direct3D11::ID3D11Device) -> Option<i64> {
    use windows::Win32::Graphics::Dxgi::*;
    use windows::core::Interface;
    unsafe {
        let dxgi_device: IDXGIDevice = device.cast().ok()?;
        let adapter = dxgi_device.GetAdapter().ok()?;
        let desc = adapter.GetDesc().ok()?;
        // Cast LowPart through u32 first to avoid sign extension from i32.
        let low = desc.AdapterLuid.LowPart as u32 as i64;
        Some(((desc.AdapterLuid.HighPart as i64) << 32) | low)
    }
}

#[cfg(target_os = "windows")]
impl WindowsD3DSharedSurface {
    pub fn new(
        device: &windows::Win32::Graphics::Direct3D11::ID3D11Device,
        context: &windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext,
        width: u32,
        height: u32,
        bgra: &[u8],
    ) -> Option<Self> {
        if std::env::var_os("ANICA_DEBUG_PREVIEW_BGRA").is_some() && !bgra.is_empty() {
            eprintln!(
                "[motionloom] WindowsD3DSharedSurface first pixel BGRA=({}, {}, {}, {})",
                bgra[0],
                bgra[1],
                bgra[2],
                bgra.get(3).copied().unwrap_or(0)
            );
        }
        let texture = create_windows_shared_bgra_texture(device, width, height)?;
        if !upload_bgra_to_d3d11_texture(context, &texture, width, height, bgra) {
            return None;
        }
        let shared_handle = windows_texture_shared_handle(&texture)?;
        let adapter_luid = d3d11_adapter_luid(device).unwrap_or(0);
        let stride = width.checked_mul(4)?;
        Some(Self {
            texture,
            width,
            height,
            format: ScenePreviewPixelFormat::Bgra8Unorm,
            handle: WindowsD3DSharedHandle(shared_handle),
            adapter_luid,
            stride,
        })
    }
}

// =========================================
// macOS CVPixelBuffer helpers.
// =========================================

#[cfg(target_os = "macos")]
pub(crate) fn create_macos_bgra_surface(
    width: u32,
    height: u32,
) -> Option<core_video::pixel_buffer::CVPixelBuffer> {
    use core_foundation::base::{CFType, TCFType};
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::string::CFString;
    use core_video::pixel_buffer::{CVPixelBuffer, CVPixelBufferKeys};

    if width == 0 || height == 0 {
        return None;
    }
    let iosurface_props: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[]);
    let cv_options: CFDictionary<CFString, CFType> = CFDictionary::from_CFType_pairs(&[
        (
            CFString::from(CVPixelBufferKeys::MetalCompatibility),
            CFBoolean::true_value().as_CFType(),
        ),
        (
            CFString::from(CVPixelBufferKeys::IOSurfaceProperties),
            iosurface_props.as_CFType(),
        ),
    ]);
    CVPixelBuffer::new(
        core_video::pixel_buffer::kCVPixelFormatType_32BGRA,
        width as usize,
        height as usize,
        Some(&cv_options),
    )
    .or_else(|_| {
        CVPixelBuffer::new(
            core_video::pixel_buffer::kCVPixelFormatType_32BGRA,
            width as usize,
            height as usize,
            None,
        )
    })
    .ok()
}

#[cfg(target_os = "macos")]
pub(crate) fn copy_bgra_into_macos_surface(
    pixel_buffer: &core_video::pixel_buffer::CVPixelBuffer,
    width: u32,
    height: u32,
    src: &[u8],
) -> bool {
    use core_video::r#return::kCVReturnSuccess;

    let w = width as usize;
    let h = height as usize;
    if w == 0
        || h == 0
        || pixel_buffer.get_pixel_format() != core_video::pixel_buffer::kCVPixelFormatType_32BGRA
        || pixel_buffer.get_width() < w
        || pixel_buffer.get_height() < h
    {
        return false;
    }
    let Some(src_stride) = w.checked_mul(4) else {
        return false;
    };
    if src.len() < src_stride.saturating_mul(h) {
        return false;
    }
    if pixel_buffer.lock_base_address(0) != kCVReturnSuccess {
        return false;
    }
    let copied = (|| {
        let dst_stride = pixel_buffer.get_bytes_per_row();
        let dst_height = pixel_buffer.get_height();
        if dst_height < h || dst_stride < src_stride {
            return None;
        }
        let dst_ptr = unsafe { pixel_buffer.get_base_address() as *mut u8 };
        if dst_ptr.is_null() {
            return None;
        }
        let dst_len = dst_stride.checked_mul(dst_height)?;
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) };
        for row in 0..h {
            let src_off = row * src_stride;
            let dst_off = row * dst_stride;
            dst[dst_off..(dst_off + src_stride)]
                .copy_from_slice(&src[src_off..(src_off + src_stride)]);
        }
        Some(())
    })()
    .is_some();
    let _ = pixel_buffer.unlock_base_address(0);
    copied
}
