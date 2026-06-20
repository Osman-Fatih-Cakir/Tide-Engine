#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require

#include "pbr.glsl"

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 0) out vec4 outColor;

struct GpuMaterial {
    vec4  baseColorFactor;
    int   baseColorTexture;
    int   normalTexture;
    int   metalRoughTexture;
    int   occlusionTexture;
    float metallicFactor;
    float roughnessFactor;
    float occlusionStrength;
    float alphaCutoff;
    int   alphaMode;
    int   pad0;
    int   pad1;
    int   pad2;
};

layout(std430, set = 0, binding = 0) readonly buffer Materials { GpuMaterial materials[]; };
layout(set = 0, binding = 1) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    vec4 cameraPos;
    vec4 sunDir;     // xyz = dir to sun, w = ambient
    vec4 sunColor;   // rgb = radiance
    uint materialIndex;
} pc;

void main() {
    GpuMaterial m = materials[pc.materialIndex];

    vec4 base = m.baseColorFactor;
    if (m.baseColorTexture >= 0)
        base *= texture(textures[nonuniformEXT(m.baseColorTexture)], vUV);
    vec3 albedo = base.rgb;
    float opacity = base.a;

    float metallic  = m.metallicFactor;
    float roughness = m.roughnessFactor;
    if (m.metalRoughTexture >= 0) {
        vec4 mr = texture(textures[nonuniformEXT(m.metalRoughTexture)], vUV);
        roughness *= mr.g;
        metallic  *= mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);

    vec3 N = normalize(vNormal);
    vec3 V = normalize(pc.cameraPos.xyz - vWorldPos);
    if (dot(N, V) < 0.0) N = -N;
    vec3 L = normalize(pc.sunDir.xyz);

    vec3 color = cookTorrance(N, V, L, albedo, metallic, roughness, pc.sunColor.rgb);
    color += pc.sunDir.w * albedo; // flat ambient

    outColor = vec4(color, opacity);
}
