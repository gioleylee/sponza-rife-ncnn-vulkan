#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in uvec4 inJoints;
layout(location = 4) in vec4 inWeights;

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 2) uniform SkinUBO {
    mat4 jointMatrices[64];
} skin;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;

void main() {
    mat4 skinMatrix =
        inWeights.x * skin.jointMatrices[inJoints.x] +
        inWeights.y * skin.jointMatrices[inJoints.y] +
        inWeights.z * skin.jointMatrices[inJoints.z] +
        inWeights.w * skin.jointMatrices[inJoints.w];

    vec4 skinnedPosition = skinMatrix * vec4(inPosition, 1.0);
    vec3 skinnedNormal = normalize(mat3(skinMatrix) * inNormal);

    vec4 worldPos = ubo.model * skinnedPosition;
    fragWorldPos = worldPos.xyz;
    fragNormal = normalize(mat3(ubo.model) * skinnedNormal);
    fragTexCoord = inTexCoord;
    gl_Position = ubo.proj * ubo.view * worldPos;
}
