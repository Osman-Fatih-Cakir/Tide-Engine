// Procedural sky (no RT), shared by resolve.comp (empty pixels) and ddgi_trace.comp
// (ray misses). Horizon->zenith gradient + a warm horizon band toward the sun's
// azimuth + an HDR sun disk/halo in direction L. dir = view/ray direction, L = dir to sun.
#ifndef SKY_GLSL
#define SKY_GLSL

vec3 skyColor(vec3 dir, vec3 L) {
    float h  = clamp(dir.y, -1.0, 1.0);
    float up = max(h, 0.0);
    vec3 zenith = vec3(0.10, 0.28, 0.65);
    vec3 ground = vec3(0.16, 0.30, 0.15); // natural green
    vec3 col = mix(ground, zenith, smoothstep(-0.01, 0.01, h));
    float azAlign = max(dot(normalize(vec3(dir.x, 0.0, dir.z)),
                            normalize(vec3(L.x, 0.0, L.z))), 0.0);
    col += vec3(0.45, 0.22, 0.08) * pow(azAlign, 4.0) * pow(1.0 - up, 6.0);

    float cosA = dot(dir, L);
    float ang  = acos(clamp(cosA, -1.0, 1.0));
    float disk = 1.0 - smoothstep(0.009, 0.013, ang);
    float halo = exp(-ang * ang / (2.0 * 0.05 * 0.05)) * 0.6
               + pow(max(cosA, 0.0), 8.0) * 0.12;
    col += vec3(1.0, 0.96, 0.88) * disk * 40.0;
    col += vec3(1.0, 0.85, 0.60) * halo;
    return col;
}

#endif
