// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/lighting.wgsl

fn distribution_ggx(normal: vec3<f32>, halfway: vec3<f32>, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let n_dot_h = max(dot(normal, halfway), 0.0);
    let denominator = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(3.14159265 * denominator * denominator, 0.000001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let k = ((roughness + 1.0) * (roughness + 1.0)) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.000001);
}

fn geometry_smith(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, roughness: f32) -> f32 {
    return geometry_schlick_ggx(max(dot(normal, view), 0.0), roughness) *
        geometry_schlick_ggx(max(dot(normal, light), 0.0), roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn direct_pbr(
    normal: vec3<f32>,
    view: vec3<f32>,
    light: vec3<f32>,
    radiance: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    f0: vec3<f32>,
    face_band: f32,
) -> vec3<f32> {
    let halfway = normalize(view + light);
    let n_dot_l = max(dot(normal, light), 0.0);
    let n_dot_v = max(dot(normal, view), 0.0);
    let distribution = distribution_ggx(normal, halfway, roughness);
    let geometry = geometry_smith(normal, view, light, roughness);
    let fresnel = fresnel_schlick(max(dot(halfway, view), 0.0), f0);
    let specular = (distribution * geometry * fresnel) / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let diffuse_weight = (vec3<f32>(1.0) - fresnel) * (1.0 - metallic);
    let diffuse = diffuse_weight * base_color / 3.14159265;
    if (lighting.surface0.x < -0.5) {
        // The filmic path uses correlated Smith visibility and Lambert diffuse for glTF.
        let a2 = pow(roughness, 4.0);
        let gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
        let gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
        let visibility = 0.5 / max(gv + gl, 0.000001);
        return (base_color * (1.0 - metallic) / 3.14159265 + distribution * visibility * fresnel) * radiance * n_dot_l;
    }
    // Art-directed diffuse is calculated per light, not by posterizing pixels.
    if (lighting.surface0.x > 3.5) {
        return shade_cel(
            normal, light, radiance, base_color, specular, n_dot_l, face_band,
        );
    }
    if (lighting.surface0.x > 0.5) {
        var intensity = stylized_intensity(normal, light);
        if (lighting.surface0.x > 1.5 && lighting.surface0.x < 2.5) {
            intensity = toon_intensity(intensity);
        }
        return shade_stylized(base_color, specular, radiance, intensity, n_dot_l);
    }
    if (lighting.surface1.y != 1.0 || lighting.surface0.z != 0.0) {
        return shade_wrapped_physical(normal, light, diffuse, specular, radiance, n_dot_l);
    }
    return shade_physical(diffuse, specular, radiance, n_dot_l);
}
