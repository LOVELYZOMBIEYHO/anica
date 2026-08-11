// =========================================
// =========================================
// crates/motionloom/src/asset.rs

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Platform-neutral source for a media or model asset.
///
/// Native builds usually load from the filesystem via `Path`. WASM builds can
/// receive assets as in-memory bytes, URLs, or Blobs so that no filesystem
/// access is required.
///
/// Currently wired into:
/// - Scene image/SVG loading (`load_rgba_image_source`, `load_svg_source`)
/// - World background image, directional character sprite, and GLB mesh loading
///
/// Remote Scene images, SVGs, and 3D model assets are fetched with `ureq` on
/// native hosts. Browser hosts fetch asynchronously and register the resulting
/// bytes in a `MemoryAssetResolver` under the unchanged DSL URL.
#[derive(Debug, Clone)]
pub enum AssetSource {
    /// Native filesystem path.
    Path(PathBuf),
    /// In-memory bytes (e.g. a loaded file or Blob).
    Bytes(Vec<u8>),
    /// Remote URL. Native renderers fetch with `ureq`; WASM hosts preload with
    /// `fetch` and register the bytes under this URL.
    Url(String),
}

impl AssetSource {
    /// Create a path-based source.
    pub fn path<P: Into<PathBuf>>(path: P) -> Self {
        Self::Path(path.into())
    }

    /// Create an in-memory source.
    pub fn bytes(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }

    /// Create a URL source.
    pub fn url(url: String) -> Self {
        Self::Url(url)
    }

    /// Return the raw bytes if this source is already in memory.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }
}

/// Resolves an asset identifier such as `<Image src="...">` into an
/// `AssetSource`. Implementations are provided for filesystem paths (native)
/// and in-memory maps (WASM).
///
/// Resolvers are owned by renderers rather than stored in a global static,
/// so multiple projects or WASM renderers can coexist without interfering
/// with each other's asset lookup.
pub trait AssetResolver: Send + Sync {
    /// Resolve `src` to an asset source. The returned source may still need
    /// network or filesystem access to obtain bytes.
    fn resolve(&self, src: &str) -> Result<AssetSource, String>;
}

/// Native filesystem resolver. Searches configured scene asset roots and
/// falls back to treating `src` as an absolute or relative path.
pub struct PathAssetResolver;

impl AssetResolver for PathAssetResolver {
    fn resolve(&self, src: &str) -> Result<AssetSource, String> {
        Ok(AssetSource::Path(PathBuf::from(src)))
    }
}

/// In-memory resolver keyed by the `src` string used in the MotionLoom
/// script. Useful for WASM hosts that preload assets.
pub struct MemoryAssetResolver {
    assets: Mutex<HashMap<String, Vec<u8>>>,
}

impl MemoryAssetResolver {
    pub fn new() -> Self {
        Self {
            assets: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert(&self, src: String, bytes: Vec<u8>) {
        self.assets
            .lock()
            .expect("memory asset lock")
            .insert(src, bytes);
    }

    pub fn clear(&self) {
        self.assets.lock().expect("memory asset lock").clear();
    }

    pub fn with_asset(self, src: String, bytes: Vec<u8>) -> Self {
        self.insert(src, bytes);
        self
    }
}

impl Default for MemoryAssetResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetResolver for MemoryAssetResolver {
    fn resolve(&self, src: &str) -> Result<AssetSource, String> {
        self.assets
            .lock()
            .expect("memory asset lock")
            .get(src)
            .cloned()
            .map(AssetSource::Bytes)
            .ok_or_else(|| format!("asset not found in memory resolver: {src}"))
    }
}

impl Default for PathAssetResolver {
    fn default() -> Self {
        Self
    }
}
