// Shared froxel-volume helpers for the volumetric fog passes (Faz 7).
// The froxel grid is camera-frustum aligned: x,y span the screen, z is an
// exponential view-space depth slice (denser near the camera). Pass the grid
// dims, the (unjittered) inverse view-projection, camera position and the near/
// far fog extents; the helpers convert between froxel coords and world space.
#ifndef FROXEL_GLSL
#define FROXEL_GLSL

const float FR_PI = 3.14159265359;

// View-space depth (along the camera forward axis) of slice s in [0,1].
float froxelSliceToViewZ(float s, float zn, float zf) {
    return zn * pow(zf / zn, s);
}
// Inverse: froxel z-slice [0,1] for a given view-space depth.
float froxelViewZToSlice(float vz, float zn, float zf) {
    return log(max(vz, zn) / zn) / log(zf / zn);
}

// Henyey-Greenstein phase function. g>0 = forward scattering (sharp god rays).
float froxelHG(float cosTheta, float g) {
    float g2 = g * g;
    float d  = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * FR_PI * pow(max(d, 1e-4), 1.5));
}

// World position at fractional froxel cell `cell` (e.g. integer index + 0.5 +
// per-frame jitter) in a `dim` grid. Jittering the sample point each frame and
// letting the temporal pass average it removes the staircase/moiré beams that a
// fixed froxel-center shadow sample produces at grazing angles.
vec3 froxelWorldPos(vec3 cell, uvec3 dim, mat4 invViewProj, vec3 camPos,
                    float zn, float zf) {
    vec2 uv  = cell.xy / vec2(dim.xy);
    vec2 ndc = uv * 2.0 - 1.0;
    // Ray through the pixel center (ndc.z = 1 = far plane).
    vec4 f = invViewProj * vec4(ndc, 1.0, 1.0);
    vec3 dir = normalize(f.xyz / f.w - camPos);
    // Camera forward axis (center ray) to convert view-Z depth to ray length.
    vec4 cc = invViewProj * vec4(0.0, 0.0, 1.0, 1.0);
    vec3 fwd = normalize(cc.xyz / cc.w - camPos);
    float vz = froxelSliceToViewZ(cell.z / float(dim.z), zn, zf);
    float t  = vz / max(dot(dir, fwd), 1e-3);
    return camPos + dir * t;
}

#endif
