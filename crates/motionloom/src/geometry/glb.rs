// =========================================
// =========================================
// crates/motionloom/src/geometry/glb.rs

use super::*;
use serde_json::{Value, json};
use std::collections::BTreeSet;

/// Static, world-space export. Cameras and lights cannot enter this data model.
pub fn export_scene_glb(snapshot: &GeometrySnapshot) -> Result<Vec<u8>, GeometryError> {
    if snapshot.meshes.is_empty() {
        return Err(GeometryError::Invalid("empty snapshot".into()));
    }
    let mut writer = Writer::default();
    let mut meshes = Vec::new();
    let mut nodes = Vec::new();
    let mut materials = Vec::new();
    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut extensions = BTreeSet::new();
    let mut texture_cache = std::collections::HashMap::<Vec<u8>, usize>::new();
    for mesh in &snapshot.meshes {
        let n = mesh.positions.len();
        if n == 0
            || mesh.indices.is_empty()
            || mesh.indices.len() % 3 != 0
            || mesh.indices.iter().any(|&i| i as usize >= n)
            || mesh.normals.len() != n
            || mesh.uvs.len() != n
            || mesh.colors.len() != n
            || mesh.tangents.len() != n
            || mesh
                .positions
                .iter()
                .flatten()
                .chain(mesh.normals.iter().flatten())
                .chain(mesh.uvs.iter().flatten())
                .chain(mesh.colors.iter().flatten())
                .chain(mesh.tangents.iter().flatten())
                .any(|x| !x.is_finite())
        {
            return Err(GeometryError::Invalid(format!(
                "invalid vertex data: {}",
                mesh.name
            )));
        }
        let p = writer.floats(&mesh.positions, "VEC3");
        let min: [f32; 3] = std::array::from_fn(|i| {
            mesh.positions
                .iter()
                .map(|p| p[i])
                .fold(f32::INFINITY, f32::min)
        });
        let max: [f32; 3] = std::array::from_fn(|i| {
            mesh.positions
                .iter()
                .map(|p| p[i])
                .fold(f32::NEG_INFINITY, f32::max)
        });
        writer.accessors[p]["min"] = json!(min);
        writer.accessors[p]["max"] = json!(max);
        let normal = writer.floats(&mesh.normals, "VEC3");
        let tangent = writer.floats(&mesh.tangents, "VEC4");
        let uv = writer.floats(&mesh.uvs, "VEC2");
        let color = writer.floats(&mesh.colors, "VEC4");
        let indices = writer.accessor(
            mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
            mesh.indices.len(),
            "SCALAR",
            5125,
            34963,
        );
        let mut texture_indices = Vec::new();
        for t in &mesh.textures {
            let image = image::RgbaImage::from_raw(t.width, t.height, t.rgba.as_ref().clone())
                .ok_or_else(|| GeometryError::Invalid("invalid texture".into()))?;
            let mut png = std::io::Cursor::new(Vec::new());
            image.write_to(&mut png, image::ImageFormat::Png)?;
            let bytes = png.into_inner();
            let index = if let Some(&i) = texture_cache.get(&bytes) {
                i
            } else {
                let view = writer.view(bytes.clone(), None);
                let index = textures.len();
                images.push(json!({"bufferView":view,"mimeType":"image/png"}));
                textures.push(json!({"source":images.len()-1,"sampler":0}));
                texture_cache.insert(bytes, index);
                index
            };
            texture_indices.push(index);
        }
        let m = &mesh.material;
        if [
            m.base_color_texture,
            m.normal_texture,
            m.metallic_roughness_texture,
            m.emissive_texture,
            m.occlusion_texture,
        ]
        .into_iter()
        .flatten()
        .any(|i| i >= texture_indices.len())
        {
            return Err(GeometryError::Invalid(format!(
                "invalid material texture index: {}",
                mesh.name
            )));
        }
        let mut material = json!({"name":m.name.as_deref().unwrap_or(&mesh.name),"pbrMetallicRoughness":{
            "baseColorFactor":m.base_color_factor,"metallicFactor":m.metallic_factor,"roughnessFactor":m.roughness_factor},
            "alphaMode":match m.alpha_mode {crate::world::gltf_loader::GlbAlphaMode::Opaque=>"OPAQUE",crate::world::gltf_loader::GlbAlphaMode::Mask=>"MASK",crate::world::gltf_loader::GlbAlphaMode::Blend=>"BLEND"},
            "doubleSided":m.double_sided,"emissiveFactor":m.emissive_factor});
        if m.alpha_mode == crate::world::gltf_loader::GlbAlphaMode::Mask {
            material["alphaCutoff"] = json!(m.alpha_cutoff);
        }
        for (path, slot) in [
            ("baseColorTexture", m.base_color_texture),
            ("metallicRoughnessTexture", m.metallic_roughness_texture),
        ] {
            if let Some(i) = slot {
                material["pbrMetallicRoughness"][path] = json!({"index":texture_indices[i]});
            }
        }
        for (path, slot) in [
            ("normalTexture", m.normal_texture),
            ("occlusionTexture", m.occlusion_texture),
            ("emissiveTexture", m.emissive_texture),
        ] {
            if let Some(i) = slot {
                material[path] = json!({"index":texture_indices[i]});
            }
        }
        if m.normal_texture.is_some() {
            material["normalTexture"]["scale"] = json!(m.normal_scale);
        }
        if m.occlusion_texture.is_some() {
            material["occlusionTexture"]["strength"] = json!(m.occlusion_strength);
        }
        let mut ext = serde_json::Map::new();
        ext.insert("KHR_materials_specular".into(),json!({"specularFactor":m.specular_factor.clamp(0.0,1.0),"specularColorFactor":m.specular_color_factor.map(|v|v.clamp(0.0,1.0))}));
        if m.emissive_strength != 1.0 {
            ext.insert(
                "KHR_materials_emissive_strength".into(),
                json!({"emissiveStrength":m.emissive_strength}),
            );
        }
        if m.unlit {
            ext.insert("KHR_materials_unlit".into(), json!({}));
        }
        if m.transmission_factor > 0.0 {
            ext.insert(
                "KHR_materials_transmission".into(),
                json!({"transmissionFactor":m.transmission_factor}),
            );
            ext.insert("KHR_materials_ior".into(), json!({"ior":m.ior}));
            ext.insert("KHR_materials_volume".into(),json!({"thicknessFactor":m.thickness_factor,"attenuationColor":m.attenuation_color,"attenuationDistance":m.attenuation_distance}));
        }
        extensions.extend(ext.keys().cloned());
        material["extensions"] = Value::Object(ext);
        let index = meshes.len();
        meshes.push(json!({"name":mesh.name,"primitives":[{"attributes":{"POSITION":p,"NORMAL":normal,"TANGENT":tangent,"TEXCOORD_0":uv,"COLOR_0":color},"indices":indices,"material":materials.len(),"mode":4}]}));
        nodes.push(json!({"name":mesh.name,"mesh":index}));
        materials.push(material);
    }
    let mut document = json!({"asset":{"version":"2.0","generator":"MotionLoom geometry snapshot"},
        "scene":0,"scenes":[{"nodes":(0..nodes.len()).collect::<Vec<_>>()}],"nodes":nodes,"meshes":meshes,
        "buffers":[{"byteLength":writer.bin.len()}],"bufferViews":writer.views,"accessors":writer.accessors,
        "materials":materials,"extensionsUsed":extensions,"extras":{"topologySignature":snapshot.topology_signature,"uvSignature":snapshot.uv_signature}});
    if !textures.is_empty() {
        document["images"] = json!(images);
        document["textures"] = json!(textures);
        document["samplers"] =
            json!([{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}]);
    }
    let mut json_bytes = serde_json::to_vec(&document)?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    while writer.bin.len() % 4 != 0 {
        writer.bin.push(0);
    }
    let length = 12 + 8 + json_bytes.len() + 8 + writer.bin.len();
    let length =
        u32::try_from(length).map_err(|_| GeometryError::Invalid("GLB exceeds 4 GiB".into()))?;
    let mut bytes = Vec::new();
    bytes.extend(b"glTF");
    bytes.extend(2u32.to_le_bytes());
    bytes.extend(length.to_le_bytes());
    bytes.extend((json_bytes.len() as u32).to_le_bytes());
    bytes.extend(0x4e4f534au32.to_le_bytes());
    bytes.extend(json_bytes);
    bytes.extend((writer.bin.len() as u32).to_le_bytes());
    bytes.extend(0x004e4942u32.to_le_bytes());
    bytes.extend(writer.bin);
    Ok(bytes)
}

#[derive(Default)]
struct Writer {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}
impl Writer {
    fn view(&mut self, bytes: Vec<u8>, target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let mut view = json!({"buffer":0,"byteOffset":self.bin.len(),"byteLength":bytes.len()});
        if let Some(t) = target {
            view["target"] = json!(t);
        }
        let index = self.views.len();
        self.views.push(view);
        self.bin.extend(bytes);
        index
    }
    fn accessor(
        &mut self,
        bytes: Vec<u8>,
        count: usize,
        kind: &str,
        component: u32,
        target: u32,
    ) -> usize {
        let view = self.view(bytes, Some(target));
        let index = self.accessors.len();
        self.accessors
            .push(json!({"bufferView":view,"componentType":component,"count":count,"type":kind}));
        index
    }
    fn floats<const N: usize>(&mut self, data: &[[f32; N]], kind: &str) -> usize {
        self.accessor(
            data.iter()
                .flatten()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            data.len(),
            kind,
            5126,
            34962,
        )
    }
}
