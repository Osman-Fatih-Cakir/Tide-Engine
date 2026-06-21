#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Depth-only shadow caster. Opaque writes depth unconditionally; alpha-masked
// (MASK) geometry discards texels below the cutoff so holes (e.g. blind slats)
// let light through. BLEND geometry never reaches this pass (CPU skips it).
layout(location = 0) in vec2 vUV;

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
    int   alphaMode;   // 0 OPAQUE, 1 MASK, 2 BLEND
    int   pad0;
    int   pad1;
    int   pad2;
};

layout(std430, set = 0, binding = 0) readonly buffer Materials { GpuMaterial materials[]; };
layout(set = 0, binding = 1) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    mat4 lightViewProj;
    mat4 model;
    uint materialIndex;
} pc;

void main() {
    GpuMaterial m = materials[pc.materialIndex];
    if (m.alphaMode == 1) { // MASK
        float a = m.baseColorFactor.a;
        if (m.baseColorTexture >= 0)
            a *= texture(textures[nonuniformEXT(m.baseColorTexture)], vUV).a;
        if (a < m.alphaCutoff) discard;
    }
}
