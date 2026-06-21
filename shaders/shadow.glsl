// PCSS soft directional shadow, shared by resolve.comp and transparent.frag.
// Manual depth comparison against a plain (non-comparison) depth shadow map.
// shadowParams = vec4(lightSizeUV, normalBias, depthBias, invShadowDim)
//   lightSizeUV : search/penumbra radius scale in shadow-map UV space (from the
//                 sun angular size); normalBias/depthBias fight acne; invDim is
//                 1/shadowResolution (one texel in UV).
#ifndef SHADOW_GLSL
#define SHADOW_GLSL

// 16-tap Poisson disk (unit radius).
const vec2 POISSON16[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845), vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554), vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507), vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367), vec2( 0.14383161, -0.14100790));

// Interleaved gradient noise -> per-pixel rotation angle (kills banding).
float ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float pcssShadow(vec3 worldPos, vec3 N, vec3 L, mat4 lightVP,
                 sampler2D shadowMap, vec4 sp, vec2 pixel) {
    // Offset along the surface normal before projecting (cheap acne fix).
    vec4 lc = lightVP * vec4(worldPos + N * sp.y, 1.0);
    vec3 ndc = lc.xyz / lc.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    float refDepth = ndc.z - sp.z;

    // Outside the light frustum -> fully lit.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0)
        return 1.0;

    // Per-pixel rotated Poisson kernel.
    float ang = ign(pixel) * 6.2831853;
    float ca = cos(ang), sa = sin(ang);
    mat2 rot = mat2(ca, -sa, sa, ca);

    float searchR = sp.x;

    // --- Blocker search ---
    float blockerSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < 16; i++) {
        vec2 o = rot * POISSON16[i] * searchR;
        float d = texture(shadowMap, uv + o).r;
        if (d < refDepth) { blockerSum += d; blockerCount++; }
    }
    if (blockerCount == 0) return 1.0; // no occluder -> lit

    float dBlocker = blockerSum / float(blockerCount);

    // --- Penumbra estimate (parallel-planes approximation) ---
    float penumbra = max((refDepth - dBlocker) / max(dBlocker, 1e-4), 0.0) * sp.x;
    float filterR = max(penumbra, sp.w); // at least one texel

    // --- PCF ---
    float sum = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 o = rot * POISSON16[i] * filterR;
        float d = texture(shadowMap, uv + o).r;
        sum += (refDepth <= d) ? 1.0 : 0.0;
    }
    return sum / 16.0;
}

#endif
