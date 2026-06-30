// Trace one ray into the scene TLAS and return the incoming radiance: sky on miss,
// else direct sun (RT-shadowed) + DDGI indirect at the hit point. Used as the
// reflection fallback when the screen-space march finds nothing.
//
// Requires (declared by the includer): scene set 0 bindings (topLevelAS, materials,
// vertices, indices, instances, geometries, textures[]); the DDGI UBO + irradiance/
// depth atlases; and rtOccluded (rtshadow.glsl), skyColor (sky.glsl),
// ddgiSampleIrradiance (ddgi.glsl).
#ifndef RTHIT_GLSL
#define RTHIT_GLSL

vec3 traceReflection(vec3 origin, vec3 dir, float maxDist) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFFu,
                          origin, 1e-3, dir, maxDist);
    while (rayQueryProceedEXT(rq)) {} // opaque: just commits the nearest hit

    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT)
        return SKY_COLOR_UBO(dir, normalize(ddgi.sunDir.xyz), ddgi) * ddgi.misc.y;

    uint instID = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
    uint primID = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, true));
    vec2 bc = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    float t = rayQueryGetIntersectionTEXT(rq, true);
    vec3 w = vec3(1.0 - bc.x - bc.y, bc.x, bc.y);

    GpuInstance inst = instances[instID];
    GpuGeometry g = geometries[inst.geometryID];
    mat4 model = inst.transform;
    uint base = g.firstIndex + primID * 3u;
    Vertex a0 = vertices[indices[base + 0u] + g.vertexOffset];
    Vertex a1 = vertices[indices[base + 1u] + g.vertexOffset];
    Vertex a2 = vertices[indices[base + 2u] + g.vertexOffset];

    vec3 hitP = origin + dir * t;
    mat3 nm = transpose(inverse(mat3(model)));
    vec3 N = normalize(nm * (a0.normal * w.x + a1.normal * w.y + a2.normal * w.z));
    if (dot(N, dir) > 0.0) N = -N; // face the incoming ray
    vec2 uv = a0.uv * w.x + a1.uv * w.y + a2.uv * w.z;

    GpuMaterial m = materials[g.materialIndex];
    vec3 albedo = m.baseColorFactor.rgb;
    if (m.baseColorTexture >= 0)
        albedo *= textureLod(textures[nonuniformEXT(m.baseColorTexture)], uv, 0.0).rgb;

    // Direct sun (Lambert) with a hard RT shadow.
    vec3 L = normalize(ddgi.sunDir.xyz);
    float ndl = max(dot(N, L), 0.0);
    float vis = (ndl > 0.0 && ddgi.shadowCfg.z > 0.5)
              ? (rtOccluded(hitP + N * 1e-3, L) ? 0.0 : 1.0)
              : (ndl > 0.0 ? 1.0 : 0.0);
    vec3 radiance = albedo / PBR_PI * ddgi.sunColor.rgb * ndl * vis;
    // Indirect: the DDGI irradiance arriving at the hit (params.y = GI intensity).
    radiance += albedo * ddgiSampleIrradiance(ddgiIrradiance, ddgiDepth, ddgi, hitP, N)
              * ddgi.params.y;
    // Emissive: reflected emissive surfaces glow in the reflection (misc.w = intensity).
    vec3 emissive = m.emissiveFactor.rgb;
    if (m.emissiveTexture >= 0)
        emissive *= textureLod(textures[nonuniformEXT(m.emissiveTexture)], uv, 0.0).rgb;
    radiance += emissive * ddgi.misc.w;
    return radiance;
}

#endif
