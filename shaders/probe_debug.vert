#version 460
#extension GL_GOOGLE_include_directive : require

// DDGI probe debug viz: draws one small icosahedron sphere per probe, instanced.
// No vertex/index buffers — the icosahedron is generated from gl_VertexIndex.
// gl_InstanceIndex = probe index; position comes from the DDGI grid params (UBO).

#include "ddgi.glsl"

layout(std140, set = 0, binding = 0) uniform DdgiUBO { DdgiParams ddgi; };
layout(push_constant) uniform Push { mat4 viewProj; float radius; } pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out flat int vProbe;

const float T = 1.618034; // golden ratio (icosahedron)
const vec3 ICO[12] = vec3[12](
    vec3(-1,  T, 0), vec3( 1, T, 0), vec3(-1,-T, 0), vec3( 1,-T, 0),
    vec3( 0, -1, T), vec3( 0, 1, T), vec3( 0,-1,-T), vec3( 0, 1,-T),
    vec3( T,  0,-1), vec3( T, 0, 1), vec3(-T, 0,-1), vec3(-T, 0, 1));
const int IDX[60] = int[60](
    0,11,5,  0,5,1,  0,1,7,  0,7,10, 0,10,11,
    1,5,9,   5,11,4, 11,10,2,10,7,6, 7,1,8,
    3,9,4,   3,4,2,  3,2,6,  3,6,8,  3,8,9,
    4,9,5,   2,4,11, 6,2,10, 8,6,7,  9,8,1);

void main() {
    int probe = gl_InstanceIndex;
    vec3 v = normalize(ICO[IDX[gl_VertexIndex]]);
    vec3 probePos = ddgiProbePos(ddgi, ddgiProbeCoord(ddgi, probe));
    vNormal = v;
    vProbe = probe;
    gl_Position = pc.viewProj * vec4(probePos + v * pc.radius, 1.0);
}
