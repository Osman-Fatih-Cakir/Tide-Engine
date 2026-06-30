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

// Roughness-aware Schlick Fresnel: rough surfaces lose their grazing rim (clamps the
// reflectance ceiling to max(1-roughness, F0) instead of 1).
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 fmax = max(vec3(1.0 - roughness), F0);
    return F0 + (fmax - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Split-sum environment BRDF (Karis/Lazarov analytic approximation of the DFG LUT).
// Returns (scale, bias): specular IBL = prefilteredColor * (F0 * scale + bias).
vec2 envBRDFApprox(float NoV, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

#endif
