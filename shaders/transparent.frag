#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require

#include "pbr.glsl"

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUV;
layout(location = 3) in vec4 vTangent;
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
layout(set = 0, binding = 5) uniform accelerationStructureEXT topLevelAS;

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    vec4 cameraPos;  // w = materialIndex (float)
    vec4 sunDir;     // xyz = dir to sun, w = ambient
    vec4 sunColor;   // rgb = radiance
    vec4 shadowCfg;  // x=sunConeRad y=samples z=shadowsOn w=frameIndex
} pc;

#include "rtshadow.glsl"

void main() {
    GpuMaterial m = materials[uint(pc.cameraPos.w)];

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

    // Baked ambient occlusion (glTF occlusion: R). Reuse ORM fetch when shared.
    float ao = 1.0;
    if (m.occlusionTexture >= 0) {
        float occ = (m.occlusionTexture == m.metalRoughTexture)
            ? texture(textures[nonuniformEXT(m.metalRoughTexture)], vUV).r
            : texture(textures[nonuniformEXT(m.occlusionTexture)],  vUV).r;
        ao = mix(1.0, occ, m.occlusionStrength);
    }

    // Two-sided: face the geometric normal toward the viewer BEFORE normal
    // mapping (flipping the perturbed normal afterwards causes dark patches).
    vec3 N = normalize(vNormal);
    vec3 V = normalize(pc.cameraPos.xyz - vWorldPos);
    float faceSign = (dot(N, V) < 0.0) ? -1.0 : 1.0;
    N *= faceSign;

    // Tangent-space normal mapping (vertex tangent; .w = handedness).
    if (m.normalTexture >= 0) {
        vec3 T = normalize(vTangent.xyz - N * dot(N, vTangent.xyz));
        vec3 B = cross(N, T) * vTangent.w * faceSign;
        vec3 nt = texture(textures[nonuniformEXT(m.normalTexture)], vUV).xyz * 2.0 - 1.0;
        N = normalize(mat3(T, B, N) * nt);
    }

    vec3 L = normalize(pc.sunDir.xyz);

    float shadow = 1.0;
    if (pc.shadowCfg.z > 0.5)
        shadow = traceSunShadow(vWorldPos, N, L, pc.shadowCfg.x,
                                int(pc.shadowCfg.y), uint(pc.shadowCfg.w), gl_FragCoord.xy);

    vec3 color = cookTorrance(N, V, L, albedo, metallic, roughness, pc.sunColor.rgb) * shadow;
    color += pc.sunDir.w * albedo * ao; // ambient occluded by baked AO

    outColor = vec4(color, opacity);
}
