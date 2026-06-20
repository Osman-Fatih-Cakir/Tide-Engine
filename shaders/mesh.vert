#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
} pc;

layout(location = 0) out vec3 vNormal;

void main() {
    gl_Position = pc.viewProj * pc.model * vec4(inPos, 1.0);
    vNormal = mat3(pc.model) * inNormal;
}
