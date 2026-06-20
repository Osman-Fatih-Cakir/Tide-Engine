// Shared PBR lighting (Cook-Torrance, metallic-roughness). Included by
// resolve.comp (opaque) and transparent.frag.
#ifndef PBR_GLSL
#define PBR_GLSL

const float PBR_PI = 3.14159265359;

// Direct radiance from one directional light. sunColor = light radiance.
vec3 cookTorrance(vec3 N, vec3 V, vec3 L, vec3 albedo,
                  float metallic, float roughness, vec3 sunColor) {
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float a  = roughness * roughness;
    float a2 = a * a;
    float dd = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    float D  = a2 / max(PBR_PI * dd * dd, 1e-7);

    float k  = (roughness + 1.0); k = k * k / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    float G  = gv * gl;

    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    vec3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 1e-4);
    vec3 kd = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / PBR_PI;
    return (diffuse + spec) * sunColor * NdotL;
}

#endif
