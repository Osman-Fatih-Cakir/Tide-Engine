#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;

layout(push_constant) uniform Push {
    float exposure;
} pc;

// ACES filmic tonemap (Narkowicz approximation).
vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdrTex, vUV).rgb * pc.exposure;
    vec3 ldr = acesFilm(hdr);
    // Swapchain is sRGB; hardware applies the OETF, so write linear.
    outColor = vec4(ldr, 1.0);
}
