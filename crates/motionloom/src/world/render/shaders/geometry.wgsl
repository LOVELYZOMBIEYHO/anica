// =========================================
// =========================================
// crates/motionloom/src/world/render/shaders/geometry.wgsl

fn bone_transform(joint: f32, position: vec3<f32>) -> vec3<f32> {
    let joint_index = u32(max(joint + 0.5, 0.0));
    return (bones.matrices[joint_index] * vec4<f32>(position, 1.0)).xyz;
}

fn bone_transform_vector(joint: f32, vector: vec3<f32>) -> vec3<f32> {
    let joint_index = u32(max(joint + 0.5, 0.0));
    return (bones.matrices[joint_index] * vec4<f32>(vector, 0.0)).xyz;
}

fn previous_bone_transform(joint: f32, position: vec3<f32>) -> vec3<f32> {
    let joint_index = u32(max(joint + 0.5, 0.0)) + u32(max(params.motion0.x, 0.0));
    return (bones.matrices[joint_index] * vec4<f32>(position, 1.0)).xyz;
}

fn cel_basis(input: VertexIn, axis: vec3<f32>) -> vec3<f32> {
    let sum = dot(input.weights, vec4<f32>(1.0));
    if (sum < 0.000001) { return normalize(actor_rotate(axis)); }
    let v = bone_transform_vector(input.joints.x, axis) * input.weights.x
        + bone_transform_vector(input.joints.y, axis) * input.weights.y
        + bone_transform_vector(input.joints.z, axis) * input.weights.z
        + bone_transform_vector(input.joints.w, axis) * input.weights.w;
    return normalize(actor_rotate(v / sum));
}

fn rotate_quaternion(authored_quaternion: vec4<f32>, vector: vec3<f32>) -> vec3<f32> {
    let quaternion = normalize(authored_quaternion);
    let doubled_cross = 2.0 * cross(quaternion.xyz, vector);
    return vector + quaternion.w * doubled_cross + cross(quaternion.xyz, doubled_cross);
}

fn actor_rotate(vector: vec3<f32>) -> vec3<f32> {
    return rotate_quaternion(params.actor_rotation, vector);
}

fn camera_hidden_joint(joint: f32) -> f32 {
    let encoded = joint + 1.0;
    let candidate = vec4<f32>(encoded);
    if (any(abs(params.hidden0 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden1 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden2 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden3 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden4 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden5 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden6 - candidate) < vec4<f32>(0.25)) ||
        any(abs(params.hidden7 - candidate) < vec4<f32>(0.25))) {
        return 1.0;
    }
    return 0.0;
}

fn vegetation_deform_at(position: vec3<f32>, vegetation: vec4<f32>) -> vec3<f32> {
    if (vegetation.x < 0.5) {
        return position;
    }
    let asset_height = max(vegetation.y, 0.001);
    let weight = smoothstep(0.04, 1.0, clamp(position.y / asset_height, 0.0, 1.0));
    let phase = vegetation.z + vegetation.w * 1.35 + position.y * 0.73;
    let sway = vec3<f32>(sin(phase), 0.0, cos(phase * 0.83))
        * weight * weight * asset_height * 0.026;
    return position + sway;
}

fn vegetation_deform(position: vec3<f32>) -> vec3<f32> {
    return vegetation_deform_at(position, params.vegetation);
}

fn cel_shared_vertex(input: VertexIn, instance_id: u32) -> VertexOut {
    params = instance_params[instance_id];
    let weight_sum = input.weights.x + input.weights.y + input.weights.z + input.weights.w;
    var skinned = vegetation_deform(input.position);
    var previous_skinned = vegetation_deform_at(input.position, params.previous_vegetation);
    var skinned_normal = input.normal;
    var skinned_tangent = input.tangent;
    var skinned_bitangent = input.bitangent;
    if (weight_sum > 0.000001) {
        skinned =
            bone_transform(input.joints.x, input.position) * (input.weights.x / weight_sum) +
            bone_transform(input.joints.y, input.position) * (input.weights.y / weight_sum) +
            bone_transform(input.joints.z, input.position) * (input.weights.z / weight_sum) +
            bone_transform(input.joints.w, input.position) * (input.weights.w / weight_sum);
        previous_skinned =
            previous_bone_transform(input.joints.x, input.position) * (input.weights.x / weight_sum) +
            previous_bone_transform(input.joints.y, input.position) * (input.weights.y / weight_sum) +
            previous_bone_transform(input.joints.z, input.position) * (input.weights.z / weight_sum) +
            previous_bone_transform(input.joints.w, input.position) * (input.weights.w / weight_sum);
        skinned_normal =
            bone_transform_vector(input.joints.x, input.normal) * (input.weights.x / weight_sum) +
            bone_transform_vector(input.joints.y, input.normal) * (input.weights.y / weight_sum) +
            bone_transform_vector(input.joints.z, input.normal) * (input.weights.z / weight_sum) +
            bone_transform_vector(input.joints.w, input.normal) * (input.weights.w / weight_sum);
        skinned_tangent =
            bone_transform_vector(input.joints.x, input.tangent) * (input.weights.x / weight_sum) +
            bone_transform_vector(input.joints.y, input.tangent) * (input.weights.y / weight_sum) +
            bone_transform_vector(input.joints.z, input.tangent) * (input.weights.z / weight_sum) +
            bone_transform_vector(input.joints.w, input.tangent) * (input.weights.w / weight_sum);
        skinned_bitangent =
            bone_transform_vector(input.joints.x, input.bitangent) * (input.weights.x / weight_sum) +
            bone_transform_vector(input.joints.y, input.bitangent) * (input.weights.y / weight_sum) +
            bone_transform_vector(input.joints.z, input.bitangent) * (input.weights.z / weight_sum) +
            bone_transform_vector(input.joints.w, input.bitangent) * (input.weights.w / weight_sum);
    }

    let local = vec3<f32>(
        skinned.x - params.model.x,
        skinned.y - params.model.y,
        skinned.z - params.model.z,
    ) * params.model.w;
    let rotated = actor_rotate(local);
    let world = params.actor.xyz + rotated;
    let previous_local = vec3<f32>(
        previous_skinned.x - params.previous_model.x,
        previous_skinned.y - params.previous_model.y,
        previous_skinned.z - params.previous_model.z,
    ) * params.previous_model.w;
    let previous_world = params.previous_actor.xyz
        + rotate_quaternion(params.previous_actor_rotation, previous_local);

    let normal_world = normalize(actor_rotate(skinned_normal));
    let tangent_world = normalize(actor_rotate(skinned_tangent));
    let bitangent_world = normalize(actor_rotate(skinned_bitangent));
    let right = params.camera1.xyz;
    let up = params.camera2.xyz;
    let forward = params.camera3.xyz;
    let rel = world - params.camera0.xyz;
    let view_x = dot(rel, right);
    let view_y = dot(rel, up);
    let view_z = dot(rel, forward);
    let near = params.camera1.w;
    let far = max(params.camera2.w, params.camera1.w + 0.001);
    var clip_x = (2.0 * params.canvas.z / params.canvas.x - 1.0) * view_z + 2.0 * view_x * params.camera0.w / params.canvas.x;
    var clip_y = (1.0 - 2.0 * params.canvas.w / params.canvas.y) * view_z + 2.0 * view_y * params.camera0.w / params.canvas.y;
    // TAA jitters projection by a sub-pixel Halton sequence. The unjittered
    // camera is retained in the lighting block for history reprojection.
    clip_x += 2.0 * lighting.preview1.x * view_z / params.canvas.x;
    clip_y -= 2.0 * lighting.preview1.y * view_z / params.canvas.y;
    let clip_z = near * (far - view_z) / (far - near);

    var out: VertexOut;
    out.instance_id = instance_id;
    // Signed camera depth allows homogeneous near-plane clipping, including
    // triangles crossing behind the camera, and perspective-correct varyings.
    out.pos = vec4<f32>(clip_x, clip_y, clip_z, view_z);
    out.current_clip = out.pos;
    let previous_relative = previous_world - lighting.previous_camera0.xyz;
    let previous_view_x = dot(previous_relative, lighting.previous_camera1.xyz);
    let previous_view_y = dot(previous_relative, lighting.previous_camera2.xyz);
    let previous_view_z = dot(previous_relative, lighting.previous_camera3.xyz);
    var previous_clip_x = 2.0 * previous_view_x * lighting.previous_camera0.w / params.canvas.x;
    var previous_clip_y = 2.0 * previous_view_y * lighting.previous_camera0.w / params.canvas.y;
    previous_clip_x += 2.0 * lighting.preview1.z * previous_view_z / params.canvas.x;
    previous_clip_y -= 2.0 * lighting.preview1.w * previous_view_z / params.canvas.y;
    let previous_near = lighting.previous_camera1.w;
    let previous_far = max(lighting.previous_camera2.w, previous_near + 0.001);
    let previous_clip_z = previous_near * (previous_far - previous_view_z)
        / (previous_far - previous_near);
    out.previous_clip = vec4<f32>(
        previous_clip_x,
        previous_clip_y,
        previous_clip_z,
        previous_view_z,
    );
    out.color = input.color;
    // Seeded variation is per instance. Keeping it out of authored vertices
    // lets every CompoundAsset child reuse the same retained geometry.
    let scaled_uv = input.uv * params.material5.xy;
    let rotated_uv = vec2<f32>(
        scaled_uv.x * params.material3.z - scaled_uv.y * params.material3.w,
        scaled_uv.x * params.material3.w + scaled_uv.y * params.material3.z,
    );
    out.uv = rotated_uv + params.material5.zw + params.material3.xy;
    out.world_position = world;
    out.normal = normal_world;
    out.tangent = tangent_world;
    out.bitangent = bitangent_world;
    out.face_forward = vec3<f32>(0.0,0.0,1.0);
    out.face_right = vec3<f32>(1.0,0.0,0.0);
    if (params.cel_material0.y > 2.5) {
        out.face_forward = cel_basis(input, vec3<f32>(0.0,0.0,1.0));
        out.face_right = cel_basis(input, vec3<f32>(1.0,0.0,0.0));
    }
    var hidden_weight = params.style.w;
    if (hidden_weight <= 0.01 && weight_sum > 0.000001) {
        hidden_weight = (camera_hidden_joint(input.joints.x) * input.weights.x +
            camera_hidden_joint(input.joints.y) * input.weights.y +
            camera_hidden_joint(input.joints.z) * input.weights.z +
            camera_hidden_joint(input.joints.w) * input.weights.w) / weight_sum;
    }
    out.hidden_weight = hidden_weight;
    return out;
}
