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
    vec4 sunColor;   // rgb = radiance, w = glass Fresnel-opacity strength
    vec4 shadowCfg;  // x=sunConeRad y=samples z=shadowsOn w=frameIndex
} pc;

#include "rtshadow.glsl"

// Probe relocation offsets (set 1 b3), before the include so ddgiProbePos agrees with
// the relocated positions the DDGI trace used.
layout(std430, set = 1, binding = 3) readonly buffer ProbeOffsets { vec4 probeOffsets[]; };
#define DDGI_PROBE_OFFSET(idx) probeOffsets[idx].xyz

#include "ddgi.glsl"

// DDGI sample set (set 1): indirect irradiance for the glass body + environment
// reflection.
layout(std140, set = 1, binding = 0) uniform DdgiUBO { DdgiParams ddgi; };
layout(set = 1, binding = 1) uniform sampler2D ddgiIrradiance;
layout(set = 1, binding = 2) uniform sampler2D ddgiDepth;

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

    // Direct sun (specular + diffuse), RT-shadowed.
    vec3 color = cookTorrance(N, V, L, albedo, metallic, roughness, pc.sunColor.rgb) * shadow;

    float ndv = max(dot(N, V), 1e-4);
    vec3  F0 = mix(vec3(0.04), albedo, metallic);

    // Indirect: DDGI irradiance fills the glass body; a second sample along the
    // reflection direction acts as a (blurry) environment for the specular reflection.
    // Falls back to the flat ambient when GI is off.
    vec3 diffuseIndirect, envSpec;
    if (ddgi.misc.x > 0.5) {
        diffuseIndirect = ddgiSampleIrradiance(ddgiIrradiance, ddgiDepth, ddgi, vWorldPos, N) * ddgi.params.y;
        // The atlas stores irradiance (E); the specular reflection wants radiance, so
        // divide by PI before using it as the environment along the reflection ray.
        vec3 R = reflect(-V, N);
        envSpec = ddgiSampleIrradiance(ddgiIrradiance, ddgiDepth, ddgi, vWorldPos, R)
                  * ddgi.params.y / PBR_PI;
    } else {
        diffuseIndirect = vec3(pc.sunDir.w);
        envSpec = vec3(pc.sunDir.w) / PBR_PI;
    }
    color += diffuseIndirect * albedo * ao * (1.0 - metallic);

    // Fresnel environment reflection, weighted by the split-sum environment BRDF.
    vec3 F   = fresnelSchlickRoughness(ndv, F0, roughness);
    vec2 ab  = envBRDFApprox(ndv, roughness);
    color   += envSpec * (F0 * ab.x + ab.y);

    // Fresnel-driven opacity: glass turns more mirror-like (opaque) at grazing angles
    // and stays transmissive head-on. Use the dielectric grazing term (not the full F,
    // which can saturate) and ease it so only the silhouette firms up, no white rim.
    float grazing = pow(1.0 - ndv, 5.0);
    float alpha = clamp(opacity + (1.0 - opacity) * grazing * pc.sunColor.w, 0.0, 1.0);

    outColor = vec4(color, alpha);
}
