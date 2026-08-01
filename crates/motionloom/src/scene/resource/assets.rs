// =========================================
// =========================================
// crates/motionloom/src/scene/resource/assets.rs

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{OnceLock, RwLock},
};

#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;

use base64::Engine;
use image::RgbaImage;

use crate::asset::AssetResolver;
use crate::scene::render::MotionLoomSceneRenderError;

#[cfg(not(target_arch = "wasm32"))]
const MAX_REMOTE_ASSET_BYTES: u64 = 64 * 1024 * 1024;

static SCENE_ASSET_ROOTS: OnceLock<RwLock<Vec<PathBuf>>> = OnceLock::new();

/// Set project-specific asset roots used by scene `<Image>` and `<Svg>` nodes.
///
/// Roots are searched before the built-in Anica public/sample asset fallbacks.
pub fn set_scene_asset_roots(roots: Vec<PathBuf>) {
    let roots = roots
        .into_iter()
        .filter(|path| !path.as_os_str().is_empty())
        .collect::<Vec<_>>();
    if let Ok(mut configured) = SCENE_ASSET_ROOTS.get_or_init(Default::default).write() {
        *configured = roots;
    }
}

/// Clear project-specific scene asset roots and return to built-in fallbacks.
pub fn clear_scene_asset_roots() {
    set_scene_asset_roots(Vec::new());
}

pub(crate) fn load_rgba_image_source(
    src: &str,
    resolver: &dyn AssetResolver,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    if is_raster_data_uri(src) {
        let bytes = decode_raster_data_uri(src)?;
        return decode_image_bytes(src, &bytes);
    }

    if is_remote_image_source(src) {
        // Browser hosts preload remote assets and register their bytes under
        // the original DSL URL. Consult that renderer-owned resolver before
        // attempting a platform fetch; WASM deliberately has no synchronous
        // network loader of its own.
        if let Ok(crate::asset::AssetSource::Bytes(bytes)) = resolver.resolve(src) {
            return decode_image_bytes(src, &bytes);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let bytes = fetch_remote_asset_bytes(src)?;
            return decode_image_bytes(src, &bytes);
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Err(MotionLoomSceneRenderError::FetchAsset {
                url: src.to_string(),
                message: "remote asset fetching is not implemented for WASM".to_string(),
            });
        }
    }

    match resolver.resolve(src) {
        Ok(crate::asset::AssetSource::Bytes(bytes)) => decode_image_bytes(src, &bytes),
        Ok(crate::asset::AssetSource::Path(path)) => {
            let path = if path.exists() {
                path
            } else {
                resolve_local_scene_asset_path(src)
            };
            image::open(&path)
                .map_err(|source| MotionLoomSceneRenderError::OpenImage { path, source })
                .map(|decoded| decoded.to_rgba8())
        }
        Ok(crate::asset::AssetSource::Url(url)) => Err(MotionLoomSceneRenderError::FetchAsset {
            url,
            message: "URL asset source requires a fetch implementation".to_string(),
        }),
        Err(message) => Err(MotionLoomSceneRenderError::FetchAsset {
            url: src.to_string(),
            message,
        }),
    }
}

fn decode_image_bytes(src: &str, bytes: &[u8]) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    image::load_from_memory(bytes)
        .map_err(|source| MotionLoomSceneRenderError::DecodeImage {
            source_ref: src.to_string(),
            source,
        })
        .map(|decoded| decoded.to_rgba8())
}

fn is_raster_data_uri(src: &str) -> bool {
    let lower = src.trim_start().to_ascii_lowercase();
    lower.starts_with("data:image/")
        && !lower.starts_with("data:image/svg+xml")
        && lower.contains(";base64,")
}

fn decode_raster_data_uri(src: &str) -> Result<Vec<u8>, MotionLoomSceneRenderError> {
    let trimmed = src.trim_start();
    let Some(comma_ix) = trimmed.find(',') else {
        return Err(MotionLoomSceneRenderError::InvalidImageDataUri {
            source_ref: src.to_string(),
            message: "missing data payload separator ','".to_string(),
        });
    };
    let (header, payload) = trimmed.split_at(comma_ix);
    let header_lower = header.to_ascii_lowercase();

    // Raster textures use data URIs so examples can stay self-contained.
    if !header_lower.starts_with("data:image/") || header_lower.starts_with("data:image/svg+xml") {
        return Err(MotionLoomSceneRenderError::InvalidImageDataUri {
            source_ref: src.to_string(),
            message: "expected raster data:image media type".to_string(),
        });
    }
    if !header_lower.contains(";base64") {
        return Err(MotionLoomSceneRenderError::InvalidImageDataUri {
            source_ref: src.to_string(),
            message: "raster image data URIs must use base64 encoding".to_string(),
        });
    }

    base64::engine::general_purpose::STANDARD
        .decode(&payload[1..])
        .map_err(|err| MotionLoomSceneRenderError::InvalidImageDataUri {
            source_ref: src.to_string(),
            message: format!("base64 decode failed: {err}"),
        })
}

pub(crate) fn load_svg_source(
    src: &str,
    resolver: &dyn AssetResolver,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    let (bytes, resources_dir) = if is_svg_data_uri(src) {
        (decode_svg_data_uri(src)?, None)
    } else if is_remote_image_source(src) {
        // Match raster-image handling: a WASM host can fetch asynchronously,
        // then expose the bytes through the renderer's MemoryAssetResolver.
        // The URL remains the stable DSL key.
        if let Ok(crate::asset::AssetSource::Bytes(bytes)) = resolver.resolve(src) {
            return render_svg_bytes(src, &bytes, None);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            (fetch_remote_asset_bytes(src)?, None)
        }
        #[cfg(target_arch = "wasm32")]
        {
            return Err(MotionLoomSceneRenderError::FetchAsset {
                url: src.to_string(),
                message: "remote asset fetching is not implemented for WASM".to_string(),
            });
        }
    } else {
        match resolver.resolve(src) {
            Ok(crate::asset::AssetSource::Bytes(bytes)) => Ok((bytes, None)),
            Ok(crate::asset::AssetSource::Path(path)) => {
                let bytes =
                    fs::read(&path).map_err(|source| MotionLoomSceneRenderError::ReadSvg {
                        path: path.clone(),
                        source,
                    })?;
                Ok((bytes, path.parent().map(Path::to_path_buf)))
            }
            Ok(crate::asset::AssetSource::Url(url)) => {
                Err(MotionLoomSceneRenderError::FetchAsset {
                    url,
                    message: "URL asset source requires a fetch implementation".to_string(),
                })
            }
            Err(message) => Err(MotionLoomSceneRenderError::FetchAsset {
                url: src.to_string(),
                message,
            }),
        }?
    };

    render_svg_bytes(src, &bytes, resources_dir)
}

fn is_svg_data_uri(src: &str) -> bool {
    src.trim_start()
        .to_ascii_lowercase()
        .starts_with("data:image/svg+xml")
}

fn decode_svg_data_uri(src: &str) -> Result<Vec<u8>, MotionLoomSceneRenderError> {
    let trimmed = src.trim_start();
    let Some(comma_ix) = trimmed.find(',') else {
        return Err(MotionLoomSceneRenderError::InvalidSvgDataUri {
            source_ref: src.to_string(),
            message: "missing data payload separator ','".to_string(),
        });
    };
    let (header, payload) = trimmed.split_at(comma_ix);
    let payload = &payload[1..];
    let header_lower = header.to_ascii_lowercase();

    if !header_lower.starts_with("data:image/svg+xml") {
        return Err(MotionLoomSceneRenderError::InvalidSvgDataUri {
            source_ref: src.to_string(),
            message: "expected data:image/svg+xml media type".to_string(),
        });
    }

    if header_lower.contains(";base64") {
        return base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|err| MotionLoomSceneRenderError::InvalidSvgDataUri {
                source_ref: src.to_string(),
                message: format!("base64 decode failed: {err}"),
            });
    }

    percent_decode_bytes(payload.as_bytes()).map_err(|message| {
        MotionLoomSceneRenderError::InvalidSvgDataUri {
            source_ref: src.to_string(),
            message,
        }
    })
}

fn percent_decode_bytes(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0usize;
    while i < input.len() {
        let ch = input[i];
        if ch == b'%' {
            if i + 2 >= input.len() {
                return Err("truncated percent escape".to_string());
            }
            let hi = decode_hex_nibble(input[i + 1])
                .ok_or_else(|| "invalid percent escape".to_string())?;
            let lo = decode_hex_nibble(input[i + 2])
                .ok_or_else(|| "invalid percent escape".to_string())?;
            out.push((hi << 4) | lo);
            i += 3;
        } else {
            out.push(ch);
            i += 1;
        }
    }
    Ok(out)
}

fn decode_hex_nibble(ch: u8) -> Option<u8> {
    match ch {
        b'0'..=b'9' => Some(ch - b'0'),
        b'a'..=b'f' => Some(ch - b'a' + 10),
        b'A'..=b'F' => Some(ch - b'A' + 10),
        _ => None,
    }
}

fn render_svg_bytes(
    source_ref: &str,
    bytes: &[u8],
    resources_dir: Option<PathBuf>,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    #[cfg(not(target_arch = "wasm32"))]
    let mut options = resvg::usvg::Options {
        resources_dir,
        ..Default::default()
    };
    #[cfg(not(target_arch = "wasm32"))]
    options.fontdb_mut().load_system_fonts();
    #[cfg(target_arch = "wasm32")]
    let options = resvg::usvg::Options {
        resources_dir,
        ..Default::default()
    };

    let tree = resvg::usvg::Tree::from_data(bytes, &options).map_err(|source| {
        MotionLoomSceneRenderError::ParseSvg {
            source_ref: source_ref.to_string(),
            source,
        }
    })?;
    let svg_size = tree.size();
    let width = svg_size.width().ceil().max(1.0) as u32;
    let height = svg_size.height().ceil().max(1.0) as u32;
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
        MotionLoomSceneRenderError::RenderSvg {
            source_ref: source_ref.to_string(),
        }
    })?;
    let transform = resvg::tiny_skia::Transform::from_scale(
        width as f32 / svg_size.width(),
        height as f32 / svg_size.height(),
    );
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    rgba_image_from_pixmap(pixmap, source_ref)
}

fn rgba_image_from_pixmap(
    pixmap: resvg::tiny_skia::Pixmap,
    source_ref: &str,
) -> Result<RgbaImage, MotionLoomSceneRenderError> {
    let width = pixmap.width();
    let height = pixmap.height();
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for pixel in pixmap.pixels() {
        let color = pixel.demultiply();
        rgba.extend_from_slice(&[color.red(), color.green(), color.blue(), color.alpha()]);
    }
    RgbaImage::from_raw(width, height, rgba).ok_or_else(|| MotionLoomSceneRenderError::RenderSvg {
        source_ref: source_ref.to_string(),
    })
}

pub(crate) fn default_world_asset_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/motionloom/world")
}

pub(crate) fn resolve_local_scene_asset_path(src: &str) -> PathBuf {
    let path = Path::new(src);
    if path.is_absolute() || path.exists() {
        return path.to_path_buf();
    }

    for root in local_scene_asset_roots() {
        for candidate in scene_asset_candidates(&root, path) {
            if candidate.exists() {
                return candidate;
            }
        }
    }

    path.to_path_buf()
}

fn local_scene_asset_roots() -> Vec<PathBuf> {
    let mut roots = Vec::<PathBuf>::new();

    if let Some(configured) = SCENE_ASSET_ROOTS.get().and_then(|roots| roots.read().ok()) {
        for root in configured.iter() {
            push_unique_path(&mut roots, root.clone());
        }
    }

    if let Some(root) = documents_anica_public_dir() {
        push_unique_path(&mut roots, root);
    }
    if let Some(root) = app_support_anica_public_dir() {
        push_unique_path(&mut roots, root);
    }
    if let Some(root) = bundle_motionloom_public_dir() {
        push_unique_path(&mut roots, root);
    }

    if let Ok(cwd) = std::env::current_dir() {
        push_unique_path(&mut roots, cwd.join("anica/examples/motionloom"));
        push_unique_path(&mut roots, cwd.join("examples/motionloom"));
        push_unique_path(
            &mut roots,
            cwd.join("anica/examples/motionloom/sample_assets"),
        );
        push_unique_path(&mut roots, cwd.join("examples/motionloom/sample_assets"));
    }

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    push_unique_path(&mut roots, crate_root.join("../../examples/motionloom"));
    push_unique_path(
        &mut roots,
        crate_root
            .join("../../..")
            .join("anica/examples/motionloom"),
    );
    push_unique_path(
        &mut roots,
        crate_root.join("../../examples/motionloom/sample_assets"),
    );
    push_unique_path(
        &mut roots,
        crate_root
            .join("../../..")
            .join("anica/examples/motionloom/sample_assets"),
    );

    roots
}

fn scene_asset_candidates(root: &Path, path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    push_unique_path(&mut candidates, root.join(path));

    for suffix in scene_asset_relative_suffixes(path) {
        push_unique_path(&mut candidates, root.join(suffix));
    }

    candidates
}

pub(crate) fn scene_asset_relative_suffixes(path: &Path) -> Vec<PathBuf> {
    let components = path
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part.to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut suffixes = Vec::new();
    for marker in ["public", "sample_assets", "motionloom"] {
        if let Some(index) = components
            .iter()
            .position(|part| part.to_string_lossy() == marker)
        {
            let suffix = pathbuf_from_components(&components[index + 1..]);
            if !suffix.as_os_str().is_empty() {
                push_unique_path(&mut suffixes, suffix);
            }
        }
    }

    suffixes
}

fn pathbuf_from_components(components: &[std::ffi::OsString]) -> PathBuf {
    let mut path = PathBuf::new();
    for component in components {
        path.push(component);
    }
    path
}

fn documents_anica_public_dir() -> Option<PathBuf> {
    Some(home_dir()?.join("Documents/AnicaProjects/public"))
}

fn app_support_anica_public_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        Some(home_dir()?.join("Library/Application Support/Anica/public"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn bundle_motionloom_public_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let contents_dir = exe.parent()?.parent()?;
    Some(contents_dir.join("Resources/motionloom/public"))
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn is_remote_image_source(src: &str) -> bool {
    url::Url::parse(src)
        .map(|url| matches!(url.scheme(), "http" | "https"))
        .unwrap_or(false)
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_remote_asset_bytes(src: &str) -> Result<Vec<u8>, MotionLoomSceneRenderError> {
    let response = ureq::get(src)
        .call()
        .map_err(|err| MotionLoomSceneRenderError::FetchAsset {
            url: src.to_string(),
            message: format_ureq_error(err),
        })?;
    let mut reader = response.into_reader().take(MAX_REMOTE_ASSET_BYTES);
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|source| MotionLoomSceneRenderError::FetchAsset {
            url: src.to_string(),
            message: source.to_string(),
        })?;
    if bytes.is_empty() {
        return Err(MotionLoomSceneRenderError::FetchAsset {
            url: src.to_string(),
            message: "response body was empty".to_string(),
        });
    }
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn format_ureq_error(err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, response) => {
            format!("HTTP {code} {}", response.status_text())
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{load_rgba_image_source, load_svg_source};
    use crate::asset::MemoryAssetResolver;
    use base64::Engine;

    #[test]
    fn remote_raster_url_uses_prefetched_resolver_bytes() {
        let src = "https://example.test/pixel.png";
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            )
            .expect("valid PNG fixture");
        let resolver = MemoryAssetResolver::new().with_asset(src.to_string(), bytes);

        let image = load_rgba_image_source(src, &resolver).expect("prefetched PNG should decode");

        assert_eq!((image.width(), image.height()), (1, 1));
    }

    #[test]
    fn remote_svg_url_uses_prefetched_resolver_bytes() {
        let src = "https://example.test/mark.svg";
        let bytes = br##"<svg xmlns="http://www.w3.org/2000/svg" width="3" height="2"><rect width="3" height="2" fill="#00ffcc"/></svg>"##.to_vec();
        let resolver = MemoryAssetResolver::new().with_asset(src.to_string(), bytes);

        let image = load_svg_source(src, &resolver).expect("prefetched SVG should render");

        assert_eq!((image.width(), image.height()), (3, 2));
    }
}
