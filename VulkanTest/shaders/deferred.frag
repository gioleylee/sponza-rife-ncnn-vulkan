#version 450

layout(input_attachment_index = 0, binding = 0) uniform subpassInput gNormal;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput gAlbedo;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput gPosition;

layout(push_constant) uniform LightingPushConstants {
    vec4 lightPos;
    vec4 lightColor;
    float showNormals;
    float showAlbedo;
    float showPosition;
    float showSpecular;
    vec4 cameraPos;
} lighting;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 positionData = subpassLoad(gPosition);
    vec3 normal = normalize(subpassLoad(gNormal).xyz);
    vec3 albedo = subpassLoad(gAlbedo).rgb;
    vec3 position = positionData.xyz;

    if (lighting.showNormals > 0.5) {
        // Visualize normals in 0-1 range for debug display.
        outColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0);
        return;
    }

    if (lighting.showAlbedo > 0.5) {
        outColor = vec4(albedo, 1.0);
        return;
    }

    if (lighting.showPosition > 0.5) {
        vec3 positionColor = normalize(position) * 0.5 + 0.5;
        outColor = vec4(positionColor, 1.0);
        return;
    }

    if (lighting.showSpecular > 0.5) {
        vec3 viewDir = normalize(lighting.cameraPos.xyz - position);
        vec3 lightDir = normalize(lighting.lightPos.xyz - position);
        vec3 halfDir = normalize(lightDir + viewDir);
        float specular = pow(max(dot(normal, halfDir), 0.0), 32.0);
        outColor = vec4(lighting.lightColor.rgb * specular, 1.0);
        return;
    }

    if (positionData.w == 0.0) {
        outColor = vec4(0.6, 0.8, 1.0, 1.0);
        return;
    }

    vec3 lightDir = normalize(lighting.lightPos.xyz - position);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = 0.02;
    vec3 color = albedo * (ambient + ndotl) * lighting.lightColor.rgb;

    outColor = vec4(color, 1.0);
}
