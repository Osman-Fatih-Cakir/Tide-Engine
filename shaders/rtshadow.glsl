// Ray-traced sun shadow, shared by resolve.comp and transparent.frag.
// Requires these to be declared by the includer (scene set 0):
//   accelerationStructureEXT topLevelAS;     (b5)
//   GpuMaterial materials[]; sampler2D textures[];
//   Vertex vertices[]; uint indices[]; GpuDraw draws[];
// Needs: #extension GL_EXT_ray_query : require
#ifndef RTSHADOW_GLSL
#define RTSHADOW_GLSL

// Cheap hash RNG for per-sample cone jitter.
float rtHash(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return float((w >> 22u) ^ w) / 4294967295.0;
}

// At a candidate triangle hit, decide whether it actually blocks light.
// OPAQUE/BLEND occluders block fully (glass is masked out by the ray cull mask,
// so a BLEND triangle only reaches here if explicitly included); MASK occluders
// block only where baseColor.a >= cutoff (holes let light through).
bool rtBlocks(uint drawID, uint primID, vec2 bary) {
    GpuDraw d = draws[drawID];
    GpuMaterial m = materials[d.materialIndex];
    if (m.alphaMode != 1) return true; // not MASK -> solid occluder

    uint base = d.firstIndex + primID * 3u;
    uint i0 = indices[base + 0u] + d.vertexOffset;
    uint i1 = indices[base + 1u] + d.vertexOffset;
    uint i2 = indices[base + 2u] + d.vertexOffset;
    vec2 uv = vertices[i0].uv * (1.0 - bary.x - bary.y)
            + vertices[i1].uv * bary.x
            + vertices[i2].uv * bary.y;

    float a = m.baseColorFactor.a;
    if (m.baseColorTexture >= 0)
        a *= textureLod(textures[nonuniformEXT(m.baseColorTexture)], uv, 0.0).a;
    return a >= m.alphaCutoff;
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

    uint seed = uint(pixel.x) * 1973u + uint(pixel.y) * 9277u + frame * 26699u + 1u;

    int n = max(samples, 1);
    float occluded = 0.0;
    for (int s = 0; s < n; s++) {
        vec3 dir = L;
        if (tanA > 0.0 && n > 1) {
            // Uniform disk sample -> perturb the direction within the cone.
            float r = sqrt(rtHash(seed)) * tanA;
            float ph = 6.2831853 * rtHash(seed);
            dir = normalize(L + (T * (r * cos(ph)) + B * (r * sin(ph))));
        }

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT,
                              0x01u, origin, 1e-3, dir, 1e4);
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) ==
                gl_RayQueryCandidateIntersectionTriangleEXT) {
                uint drawID = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false));
                uint primID = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, false));
                vec2 bary  = rayQueryGetIntersectionBarycentricsEXT(rq, false);
                if (rtBlocks(drawID, primID, bary))
                    rayQueryConfirmIntersectionEXT(rq);
            }
        }
        if (rayQueryGetIntersectionTypeEXT(rq, true) !=
            gl_RayQueryCommittedIntersectionNoneEXT)
            occluded += 1.0;
    }
    return 1.0 - occluded / float(n);
}

#endif
