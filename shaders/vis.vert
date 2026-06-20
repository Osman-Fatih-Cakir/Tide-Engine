#version 450

layout(location = 0) in vec3 inPos;

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    uint drawID;
} pc;

void main() {
    gl_Position = pc.viewProj * pc.model * vec4(inPos, 1.0);
}
