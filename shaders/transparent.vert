#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec2 vUV;

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    vec4 cameraPos;
    vec4 sunDir;     // xyz = dir to sun, w = ambient
    vec4 sunColor;   // rgb = radiance
    uint materialIndex;
} pc;

void main() {
    vec4 wp = pc.model * vec4(inPos, 1.0);
    vWorldPos = wp.xyz;
    vNormal = mat3(transpose(inverse(pc.model))) * inNormal;
    vUV = inUV;
    gl_Position = pc.viewProj * wp;
}
