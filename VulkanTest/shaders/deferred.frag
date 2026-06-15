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
    vec4 debugPanelRect;
    float debugPanelEnabled;
    float debugPanelFrameId;
} lighting;

layout(location = 0) out vec4 outColor;

float box(vec2 p, vec2 origin, vec2 size) {
    vec2 insideMin = step(origin, p);
    vec2 insideMax = step(p, origin + size);
    return insideMin.x * insideMin.y * insideMax.x * insideMax.y;
}

int debugPanelGlyphRow(int glyph, int row) {
    if (glyph == 0) return row == 0 ? 6 : row == 1 ? 5 : row == 2 ? 5 : row == 3 ? 5 : 6; // D
    if (glyph == 1) return row == 0 ? 7 : row == 1 ? 4 : row == 2 ? 6 : row == 3 ? 4 : 7; // E
    if (glyph == 2) return row == 0 ? 6 : row == 1 ? 5 : row == 2 ? 6 : row == 3 ? 5 : 6; // B
    if (glyph == 3) return row == 0 ? 5 : row == 1 ? 5 : row == 2 ? 5 : row == 3 ? 5 : 7; // U
    if (glyph == 4) return row == 0 ? 7 : row == 1 ? 4 : row == 2 ? 5 : row == 3 ? 5 : 7; // G
    if (glyph == 5) return row == 0 ? 6 : row == 1 ? 5 : row == 2 ? 6 : row == 3 ? 4 : 4; // P
    if (glyph == 6) return row == 0 ? 2 : row == 1 ? 5 : row == 2 ? 7 : row == 3 ? 5 : 5; // A
    if (glyph == 7) return row == 0 ? 5 : row == 1 ? 7 : row == 2 ? 7 : row == 3 ? 7 : 5; // N
    if (glyph == 8) return row == 0 ? 4 : row == 1 ? 4 : row == 2 ? 4 : row == 3 ? 4 : 7; // L
    if (glyph == 9) return row == 0 ? 0 : row == 1 ? 2 : row == 2 ? 0 : row == 3 ? 2 : 0; // :
    if (glyph == 10) return row == 0 ? 7 : row == 1 ? 5 : row == 2 ? 5 : row == 3 ? 5 : 7; // 0
    if (glyph == 11) return row == 0 ? 2 : row == 1 ? 6 : row == 2 ? 2 : row == 3 ? 2 : 7; // 1
    if (glyph == 12) return row == 0 ? 7 : row == 1 ? 1 : row == 2 ? 7 : row == 3 ? 4 : 7; // 2
    if (glyph == 13) return row == 0 ? 7 : row == 1 ? 1 : row == 2 ? 7 : row == 3 ? 1 : 7; // 3
    if (glyph == 14) return row == 0 ? 5 : row == 1 ? 5 : row == 2 ? 7 : row == 3 ? 1 : 1; // 4
    if (glyph == 15) return row == 0 ? 7 : row == 1 ? 4 : row == 2 ? 7 : row == 3 ? 1 : 7; // 5
    if (glyph == 16) return row == 0 ? 7 : row == 1 ? 4 : row == 2 ? 7 : row == 3 ? 5 : 7; // 6
    if (glyph == 17) return row == 0 ? 7 : row == 1 ? 1 : row == 2 ? 2 : row == 3 ? 4 : 4; // 7
    if (glyph == 18) return row == 0 ? 7 : row == 1 ? 5 : row == 2 ? 7 : row == 3 ? 5 : 7; // 8
    if (glyph == 19) return row == 0 ? 7 : row == 1 ? 5 : row == 2 ? 7 : row == 3 ? 1 : 7; // 9
    if (glyph == 20) return row == 0 ? 7 : row == 1 ? 2 : row == 2 ? 2 : row == 3 ? 2 : 7; // I
    return 0;
}

int digitGlyph(int digit) {
    return 10 + clamp(digit, 0, 9);
}

float drawGlyph(vec2 p, int glyph) {
    float m = 0.0;
    const float cell = 3.0;
    for (int row = 0; row < 5; ++row) {
        int rowMask = debugPanelGlyphRow(glyph, row);
        for (int col = 0; col < 3; ++col) {
            int bitValue = col == 0 ? 4 : col == 1 ? 2 : 1;
            if ((rowMask / bitValue) % 2 == 1) {
                m = max(m, box(p, vec2(float(col) * cell, float(row) * cell), vec2(cell)));
            }
        }
    }
    return m;
}

float drawFrameIdText(vec2 p, int frameId) {
    float m = 0.0;
    int labelGlyphs[11] = int[](0, 1, 2, 3, 4, -1, 5, 6, 7, 1, 8);
    for (int i = 0; i < 11; ++i) {
        if (labelGlyphs[i] >= 0) {
            m = max(m, drawGlyph(p - vec2(14.0 + float(i) * 12.0, 40.0), labelGlyphs[i]));
        }
    }

    int idGlyphs[3] = int[](20, 0, 9);
    for (int i = 0; i < 3; ++i) {
        m = max(m, drawGlyph(p - vec2(14.0 + float(i) * 12.0, 16.0), idGlyphs[i]));
    }

    int value = frameId % 100000;
    int divisor = 10000;
    for (int i = 0; i < 5; ++i) {
        int digit = (value / divisor) % 10;
        vec2 digitPos = p - vec2(56.0 + float(i) * 12.0, 16.0);
        m = max(m, drawGlyph(digitPos, digitGlyph(digit)));
        divisor = divisor / 10;
    }
    return m;
}

vec3 applyInterpolationDebugPanel(vec3 baseColor) {
    if (lighting.debugPanelEnabled < 0.5) {
        return baseColor;
    }

    vec2 local = gl_FragCoord.xy - lighting.debugPanelRect.xy;
    vec2 size = lighting.debugPanelRect.zw;
    float inside = box(local, vec2(0.0), size);
    if (inside < 0.5) {
        return baseColor;
    }

    vec3 panelColor = mix(baseColor, vec3(0.02, 0.08, 0.12), 0.5);
    float border = step(local.x, 2.0) + step(local.y, 2.0) +
        step(size.x - 2.0, local.x) + step(size.y - 2.0, local.y);
    float grid = 0.0;
    grid = max(grid, 1.0 - step(1.0, mod(local.x, 18.0)));
    grid = max(grid, 1.0 - step(1.0, mod(local.y, 18.0)));
    float text = drawFrameIdText(local, int(lighting.debugPanelFrameId + 0.5));

    panelColor = mix(panelColor, vec3(0.35, 0.95, 1.0), grid * 0.32);
    panelColor = mix(panelColor, vec3(1.0), clamp(border, 0.0, 1.0));
    panelColor = mix(panelColor, vec3(1.0, 0.95, 0.2), text);
    return panelColor;
}

void main() {
    vec4 positionData = subpassLoad(gPosition);
    vec3 normal = normalize(subpassLoad(gNormal).xyz);
    vec3 albedo = subpassLoad(gAlbedo).rgb;
    vec3 position = positionData.xyz;

    if (lighting.showNormals > 0.5) {
        // Visualize normals in 0-1 range for debug display.
        outColor = vec4(applyInterpolationDebugPanel(normalize(normal) * 0.5 + 0.5), 1.0);
        return;
    }

    if (lighting.showAlbedo > 0.5) {
        outColor = vec4(applyInterpolationDebugPanel(albedo), 1.0);
        return;
    }

    if (lighting.showPosition > 0.5) {
        vec3 positionColor = normalize(position) * 0.5 + 0.5;
        outColor = vec4(applyInterpolationDebugPanel(positionColor), 1.0);
        return;
    }

    if (lighting.showSpecular > 0.5) {
        vec3 viewDir = normalize(lighting.cameraPos.xyz - position);
        vec3 lightDir = normalize(lighting.lightPos.xyz - position);
        vec3 halfDir = normalize(lightDir + viewDir);
        float specular = pow(max(dot(normal, halfDir), 0.0), 32.0);
        outColor = vec4(applyInterpolationDebugPanel(lighting.lightColor.rgb * specular), 1.0);
        return;
    }

    if (positionData.w == 0.0) {
        outColor = vec4(applyInterpolationDebugPanel(vec3(0.6, 0.8, 1.0)), 1.0);
        return;
    }

    vec3 lightDir = normalize(lighting.lightPos.xyz - position);
    float ndotl = max(dot(normal, lightDir), 0.0);
    float ambient = 0.02;
    vec3 color = albedo * (ambient + ndotl) * lighting.lightColor.rgb;

    outColor = vec4(applyInterpolationDebugPanel(color), 1.0);
}
