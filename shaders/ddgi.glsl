// Shared DDGI helpers: probe grid math, octahedral encode/decode, atlas mapping,
// and irradiance sampling. Included by ddgi_trace.comp, ddgi_update.comp and
// resolve.comp. Atlases live in one big 2D image, tiled per probe:
//   probe (x,y,z) -> tile (x, y + Ny*z);  irradiance tile = 8x8, depth tile = 16x16.
// Tiles have NO border: bilinear lookups clamp to the tile interior, and the
// 8-probe trilinear blend hides the residual faceting (low-frequency signal).
#ifndef DDGI_GLSL
#define DDGI_GLSL

const int  DDGI_IRR_RES   = 8;   // octahedral irradiance resolution per probe
const int  DDGI_DEPTH_RES = 16;  // octahedral depth/moments resolution per probe
const int  DDGI_MAX_RAYS  = 128; // SSBO stride per probe
const float DDGI_PI = 3.14159265359;

// Matches the C++ DdgiParams UBO (std140; all members 16-byte aligned).
struct DdgiParams {
    vec4  gridOrigin;   // xyz = world pos of probe (0,0,0)
    vec4  gridSpacing;  // xyz = spacing between probes
    ivec4 gridCounts;   // xyz = Nx,Ny,Nz ; w = raysPerProbe
    vec4  sunDir;       // xyz = dir to sun ; w = sky/ambient intensity
    vec4  sunColor;     // rgb = sun radiance
    vec4  params;       // x=hysteresis y=intensity z=normalBias w=frame
    vec4  shadowCfg;    // x=coneRad y=samples z=shadowsOn w=maxRayDist
    vec4  misc;         // x=use GI in resolve (0/1)
};

// ---- probe <-> world / atlas ----
int  ddgiProbeCount(DdgiParams p) { return p.gridCounts.x * p.gridCounts.y * p.gridCounts.z; }

vec3 ddgiProbePos(DdgiParams p, ivec3 c) {
    return p.gridOrigin.xyz + p.gridSpacing.xyz * vec3(c);
}
ivec3 ddgiProbeCoord(DdgiParams p, int idx) {
    int nx = p.gridCounts.x, ny = p.gridCounts.y;
    return ivec3(idx % nx, (idx / nx) % ny, idx / (nx * ny));
}
int ddgiProbeIndex(DdgiParams p, ivec3 c) {
    return c.x + p.gridCounts.x * (c.y + p.gridCounts.y * c.z);
}
ivec2 ddgiTile(DdgiParams p, ivec3 c) { return ivec2(c.x, c.y + p.gridCounts.y * c.z); }

// Atlas dimensions in tiles (so atlasUV cancels the per-tile resolution).
vec2 ddgiTilesDim(DdgiParams p) { return vec2(p.gridCounts.x, p.gridCounts.y * p.gridCounts.z); }

// ---- octahedral mapping (Cigolle et al.) ----
vec2 octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 o = (n.z >= 0.0) ? n.xy : (1.0 - abs(n.yx)) * sign(n.xy);
    return o * 0.5 + 0.5; // [0,1]
}
vec3 octDecode(vec2 f) {
    f = f * 2.0 - 1.0;
    vec3 n = vec3(f.xy, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

// Atlas UV for probe tile `c`, looking in direction `dir`, with `res` per tile.
vec2 ddgiAtlasUV(DdgiParams p, ivec3 c, vec3 dir, int res) {
    vec2 uv = octEncode(dir);
    float ht = 0.5 / float(res);
    uv = clamp(uv, vec2(ht), vec2(1.0 - ht)); // stay inside the tile (no border)
    return (vec2(ddgiTile(p, c)) + uv) / ddgiTilesDim(p);
}

// ---- per-frame ray direction set (trace & update MUST agree) ----
vec3 ddgiFibonacci(int i, int n) {
    const float GA = 2.39996323; // golden angle
    float k  = float(i) + 0.5;
    float ct = 1.0 - 2.0 * k / float(n);
    float st = sqrt(max(1.0 - ct * ct, 0.0));
    float ph = GA * float(i);
    return vec3(cos(ph) * st, sin(ph) * st, ct);
}
float ddgiHash(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
mat3 ddgiFrameRotation(float frame) {
    float a = ddgiHash(frame * 3.0 + 0.0) * 6.2831853;
    float b = ddgiHash(frame * 3.0 + 1.0) * 6.2831853;
    float c = ddgiHash(frame * 3.0 + 2.0) * 6.2831853;
    mat3 rx = mat3(1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a));
    mat3 ry = mat3(cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b));
    mat3 rz = mat3(cos(c), -sin(c), 0, sin(c), cos(c), 0, 0, 0, 1);
    return rz * ry * rx;
}
vec3 ddgiRayDir(DdgiParams p, int ray) {
    return normalize(ddgiFrameRotation(p.params.w) *
                     ddgiFibonacci(ray, p.gridCounts.w));
}

// ---- irradiance lookup at a shading point ----
// Trilinear over the 8 surrounding probes, weighted by a normal-facing term and
// Chebyshev (variance) visibility from the depth atlas to suppress light leaking.
vec3 ddgiSampleIrradiance(sampler2D irrAtlas, sampler2D depthAtlas,
                          DdgiParams p, vec3 P, vec3 N) {
    vec3 biasP = P + N * p.params.z; // normalBias: lift off the surface
    vec3 base  = (P - p.gridOrigin.xyz) / p.gridSpacing.xyz;
    ivec3 b0   = ivec3(floor(base));
    vec3 frac  = clamp(base - vec3(b0), 0.0, 1.0);

    vec3  sumIrr = vec3(0.0);
    float sumW   = 0.0;
    for (int i = 0; i < 8; i++) {
        ivec3 off = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 c   = clamp(b0 + off, ivec3(0), p.gridCounts.xyz - 1);
        vec3  pp  = ddgiProbePos(p, c);

        vec3 tri3 = mix(1.0 - frac, frac, vec3(off));
        float trilinear = tri3.x * tri3.y * tri3.z;

        vec3 dirToProbe = normalize(pp - P);
        float wDir = max(dot(dirToProbe, N) * 0.5 + 0.5, 0.0);
        wDir *= wDir;

        // Chebyshev visibility (depth atlas stores mean, mean^2 of ray distance).
        // Softened vs reference RTXGI (v^2 not v^3, larger variance floor, a small
        // depth bias) so it kills genuine leaks without zeroing valid contributions.
        // Pre-cull negligible probes (also skips their visibility ray).
        float wpre = trilinear * wDir;
        if (wpre < 1e-3) continue;

        // Visibility: exact ray-traced (leak-free, needs TLAS) or Chebyshev (cheap).
        float vis;
#ifdef DDGI_HAS_TLAS
        if (p.misc.w > 0.5) {
            // Trace a short ray from the (biased) shading point to the probe. Any opaque
            // hit in between = occluded -> this probe can't light P. No leak, ever.
            vec3  toP = pp - biasP;
            float dP  = length(toP);
            rayQueryEXT vq;
            rayQueryInitializeEXT(vq, topLevelAS,
                gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFFu,
                biasP, 1e-2, toP / max(dP, 1e-4), dP - 2e-2);
            rayQueryProceedEXT(vq);
            if (rayQueryGetIntersectionTypeEXT(vq, true) !=
                gl_RayQueryCommittedIntersectionNoneEXT) continue; // occluded -> drop probe
            vis = 1.0;
        } else
#endif
        {
            vec3  pToProbe = biasP - pp;
            float dist = length(pToProbe);
            vec2  mom  = texture(depthAtlas, ddgiAtlasUV(p, c, normalize(pToProbe), DDGI_DEPTH_RES)).rg;
            float mean = mom.x;
            float varr = max(mom.y - mean * mean, 2e-3);
            float cheb = 1.0;
            if (dist > mean) { float d = dist - mean; cheb = varr / (varr + d * d); cheb = cheb*cheb*cheb; }
            if (cheb < 0.2) cheb *= cheb * cheb / 0.04; // crush residual leak
            vis = cheb;
        }

        float w = wpre * vis;
        vec3  irr = texture(irrAtlas, ddgiAtlasUV(p, c, N, DDGI_IRR_RES)).rgb;
        sumIrr += irr * w;
        sumW   += w;
    }
    // Fully enclosed/occluded point -> return 0 (dark) rather than an unweighted leak.
    return sumW > 1e-4 ? sumIrr / sumW : vec3(0.0);
}

#endif
