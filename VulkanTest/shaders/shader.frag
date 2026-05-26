#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outNormal;   // G-buffer: normal
layout(location = 1) out vec4 outAlbedo;   // G-buffer: albedo
layout(location = 2) out vec4 outPosition; // G-buffer: position

void main() {
    vec4 albedo = texture(texSampler, fragTexCoord);

    outNormal = vec4(normalize(fragNormal), 0.0);
    outAlbedo = albedo;
    outPosition = vec4(fragWorldPos, 1.0);
}
