// Ray-traced sun shadow, shared by resolve.comp and transparent.frag.
// All occluders are opaque (transparent geometry is excluded from the TLAS), so
// the ray query auto-commits the first hit — no candidate loop, no alpha test.
// Requires (declared by the includer, scene set 0):
//   accelerationStructureEXT topLevelAS;     (b5)
// Needs: #extension GL_EXT_ray_query : require
#ifndef RTSHADOW_GLSL
#define RTSHADOW_GLSL

// Cheap hash RNG for per-sample cone jitter.
float rtHash(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return float((w >> 22u) ^ w) / 4294967295.0;
}

// True if the ray from origin in dir hits any opaque occluder within [tMin,tMax].
bool rtOccluded(vec3 origin, vec3 dir) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                          0xFFu, origin, 1e-3, dir, 1e4);
    rayQueryProceedEXT(rq); // opaque -> commits the first hit, no candidate handling
    return rayQueryGetIntersectionTypeEXT(rq, true) !=
           gl_RayQueryCommittedIntersectionNoneEXT;
}

// Returns sun visibility in [0,1]. coneHalfAngle in radians (0 = hard shadow).
float traceSunShadow(vec3 P, vec3 N, vec3 L, float coneHalfAngle,
                     int samples, uint frame, vec2 pixel) {
    if (dot(N, L) <= 0.0) return 0.0; // facing away from the sun -> no direct light

    vec3 origin = P + N * 1e-3;       // lift off the surface to avoid self-hit

    // Orthonormal basis around L for cone sampling.
    vec3 T = normalize(abs(L.x) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0));
    T = normalize(T - L * dot(T, L));
    vec3 B = cross(L, T);
    float tanA = tan(coneHalfAngle);

    int n = max(samples, 1);

    // Golden-angle (Fibonacci) disk: low-discrepancy radial layout, rotated by a
    // per-pixel + per-frame phase so frames decorrelate (temporal averages them).
    uint seed = uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + frame * 26699u + 1u;
    float rot = 6.2831853 * rtHash(seed);
    const float GOLDEN = 2.39996323; // radians

    float occluded = 0.0;
    for (int s = 0; s < n; s++) {
        vec3 dir = L;
        if (tanA > 0.0) {
            float r  = sqrt((float(s) + 0.5) / float(n)) * tanA;
            float ph = float(s) * GOLDEN + rot;
            dir = normalize(L + (T * (r * cos(ph)) + B * (r * sin(ph))));
        }
        if (rtOccluded(origin, dir)) occluded += 1.0;
    }
    return 1.0 - occluded / float(n);
}

#endif
