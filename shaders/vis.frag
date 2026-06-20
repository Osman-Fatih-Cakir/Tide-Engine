#version 450

layout(location = 0) out uint outVis;

layout(push_constant) uniform Push {
    mat4 viewProj;
    mat4 model;
    uint drawID;
} pc;

void main() {
    // Pack (drawID << 20) | primitiveID. drawID < 4096, primID < 1M.
    outVis = (pc.drawID << 20) | (uint(gl_PrimitiveID) & 0xFFFFFu);
}
