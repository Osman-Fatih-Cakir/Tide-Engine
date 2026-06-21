#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) out vec4 vTangent;

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    vec4 cameraPos;  // w = materialIndex (float)
    vec4 sunDir;     // xyz = dir to sun, w = ambient
    vec4 sunColor;   // rgb = radiance
    vec4 shadowCfg;
} pc;

void main() {
    vec4 wp = pc.model * vec4(inPos, 1.0);
    vWorldPos = wp.xyz;
    vNormal = mat3(transpose(inverse(pc.model))) * inNormal;
    vTangent = vec4(mat3(pc.model) * inTangent.xyz, inTangent.w);
    vUV = inUV;
    gl_Position = pc.viewProj * wp;
}
