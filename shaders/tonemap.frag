#version 450

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;

layout(push_constant) uniform Push {
    float exposure;
    int   mode;       // 0 = ACES, 1 = AgX
} pc;

// ----- ACES filmic (Narkowicz approximation) -----
// Punchy, contrasty "cinematic" look. Hue shifts at very bright/saturated values
// (blue->purple, highlights skew). Long the AAA default.
vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ----- AgX (Troy Sobotka; Blender 4.0 default) -----
// Preserves hue under heavy exposure and desaturates highlights toward white
// ("path to white") instead of clipping/shifting color. Slightly flatter out of
// the box, so a small look (contrast/saturation push) is applied after.
// Minimal 6th-order polynomial fit of the AgX sigmoid (Filament/Three.js variant).
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return  - 17.86     * x4 * x2 * x
            + 78.01     * x4 * x2
            - 126.7     * x4 * x
            + 92.06     * x4
            - 28.72     * x2 * x
            + 4.361     * x2
            - 0.1718    * x
            + 0.002857;
}

vec3 agx(vec3 val) {
    // sRGB -> AgX base (input transform).
    const mat3 agxIn = mat3(
        0.842479062253094,  0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772,  0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 agxOut = mat3(
         1.19687900512017,  -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
        -0.0990297440797205,-0.0989611768448433,  1.15107367264116);

    val = agxIn * val;
    // Log2 encoding over the AgX dynamic range, then the sigmoid contrast curve.
    const float minEv = -12.47393, maxEv = 4.026069;
    val = clamp(log2(val), minEv, maxEv);
    val = (val - minEv) / (maxEv - minEv);
    val = agxDefaultContrastApprox(val);

    // Look: punch up contrast + saturation a touch so AgX isn't washed out.
    const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    float luma = dot(val, lw);
    val = pow(val, vec3(1.0));               // (gamma hook; identity here)
    val = luma + 1.18 * (val - luma);        // saturation 1.18
    val = mix(vec3(luma), val, 1.0);

    val = agxOut * val;
    val = clamp(val, 0.0, 1.0);
    // The fit outputs sRGB-gamma (display-referred); our swapchain is sRGB and the
    // hardware re-applies the OETF, so linearize here to avoid double gamma.
    return pow(val, vec3(2.2));
}

void main() {
    vec3 hdr = texture(hdrTex, vUV).rgb * pc.exposure;
    vec3 ldr = (pc.mode == 1) ? agx(hdr) : acesFilm(hdr);
    // Swapchain is sRGB; hardware applies the OETF, so write linear.
    outColor = vec4(ldr, 1.0);
}
