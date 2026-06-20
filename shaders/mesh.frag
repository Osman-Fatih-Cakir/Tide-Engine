#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vUV;
layout(location = 0) out vec4 outColor;

struct GpuMaterial {
    vec4  baseColorFactor;
    int   baseColorTexture;
    int   normalTexture;
    int   metalRoughTexture;
    float metallicFactor;
    float roughnessFactor;
    int   pad0;
    int   pad1;
    int   pad2;
};

layout(std430, set = 0, binding = 0) readonly buffer Materials {
    GpuMaterial materials[];
};
layout(set = 0, binding = 1) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    uint materialIndex;
} pc;

void main() {
    GpuMaterial m = materials[pc.materialIndex];
    vec3 albedo = m.baseColorFactor.rgb;
    if (m.baseColorTexture >= 0)
        albedo *= texture(textures[nonuniformEXT(m.baseColorTexture)], vUV).rgb;

    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(N, L), 0.0) * 0.8 + 0.2;
    outColor = vec4(albedo * diff, 1.0);
}
