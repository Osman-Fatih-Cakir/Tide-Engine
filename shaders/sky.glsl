// Procedural sky (no RT), shared by resolve.comp (empty pixels) and ddgi_trace.comp
// (ray misses). Horizon->zenith gradient + a warm horizon band toward the sun's
// azimuth + an HDR sun disk/halo in direction L. dir = view/ray direction, L = dir to sun.
#ifndef SKY_GLSL
#define SKY_GLSL

// zenith/ground/horizon = artist sky colors, intensity = overall sky brightness,
// sunCol = sun radiance (its hue tints the visible disk/halo).
vec3 skyColor(vec3 dir, vec3 L, vec3 zenith, vec3 ground, vec3 horizon, float intensity, vec3 sunCol) {
    float h  = clamp(dir.y, -1.0, 1.0);
    float up = max(h, 0.0);
    vec3 col = mix(ground, zenith, smoothstep(-0.01, 0.01, h));
    float azAlign = max(dot(normalize(vec3(dir.x, 0.0, dir.z)),
                            normalize(vec3(L.x, 0.0, L.z))), 0.0);
    col += horizon * pow(azAlign, 4.0) * pow(1.0 - up, 6.0);
    col *= intensity;

    float cosA = dot(dir, L);
    float ang  = acos(clamp(cosA, -1.0, 1.0));
    float disk = 1.0 - smoothstep(0.009, 0.013, ang);
    float halo = exp(-ang * ang / (2.0 * 0.05 * 0.05)) * 0.6
               + pow(max(cosA, 0.0), 8.0) * 0.12;
    // Tint the disk/halo by the sun's hue (normalize out its intensity magnitude).
    float sm = max(max(sunCol.r, sunCol.g), sunCol.b);
    vec3 sunHue = (sm > 1e-4) ? sunCol / sm : vec3(1.0);
    // Sun disk/halo scale with sky brightness so they fade out at night.
    col += sunHue * vec3(1.0, 0.96, 0.88) * disk * 40.0 * intensity;
    col += sunHue * vec3(1.0, 0.85, 0.60) * halo * intensity;
    return col;
}

// Convenience overload reading the sky params straight from the DDGI UBO.
#define SKY_COLOR_UBO(dir, L, u) skyColor((dir), (L), (u).skyZenith.rgb, (u).skyGround.rgb, (u).skyHorizon.rgb, (u).skyZenith.w, (u).sunColor.rgb)

#endif
