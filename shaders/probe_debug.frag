#version 460
#extension GL_GOOGLE_include_directive : require

// Colors each probe sphere by the irradiance it stores in the surface direction
// (an "irradiance ball"). Invalid probes (inside/behind geometry, classified out)
// are tinted red so leak sources stand out. Cheap; only drawn when debug is on.

#include "ddgi.glsl"

layout(std140, set = 0, binding = 0) uniform DdgiUBO { DdgiParams ddgi; };
layout(set = 0, binding = 1) uniform sampler2D ddgiIrradiance;

layout(location = 0) in vec3 vNormal;
layout(location = 1) in flat int vProbe;
layout(location = 0) out vec4 outColor;

void main() {
    ivec3 c = ddgiProbeCoord(ddgi, vProbe);
    vec3  n = normalize(vNormal);
    vec4  irr = texture(ddgiIrradiance, ddgiAtlasUV(ddgi, c, n, DDGI_IRR_RES));
    vec3  col = irr.rgb * max(ddgi.params.y, 1.0); // irradiance, kept visible
    if (irr.a < 0.5) col = vec3(0.5, 0.0, 0.0);    // classified invalid -> red
    outColor = vec4(col, 1.0);
}
