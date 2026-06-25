#version 460

// Fog box debug viz: draws the 12 edges of the local fog box as a line list.
// No vertex/index buffers — the 24 edge endpoints (12 edges x 2) are generated
// from gl_VertexIndex by selecting box corners. Corner i picks min/max per axis
// from bit (i.x, i.y, i.z); push constant carries the box bounds and viewProj.

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec4 boxMin; // xyz
    vec4 boxMax; // xyz
} pc;

// 12 edges as pairs of corner indices (0..7, bit0=x bit1=y bit2=z).
const int EDGES[24] = int[24](
    0,1, 1,3, 3,2, 2,0,   // bottom (z = min)
    4,5, 5,7, 7,6, 6,4,   // top    (z = max)
    0,4, 1,5, 2,6, 3,7);  // verticals

void main() {
    int corner = EDGES[gl_VertexIndex];
    vec3 t = vec3(float(corner & 1), float((corner >> 1) & 1), float((corner >> 2) & 1));
    vec3 p = mix(pc.boxMin.xyz, pc.boxMax.xyz, t);
    gl_Position = pc.viewProj * vec4(p, 1.0);
}
