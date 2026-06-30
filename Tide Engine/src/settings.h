#pragma once
#include "pch.h"

// One stop on the camera flythrough path: a full camera transform.
struct CamWaypoint {
    glm::vec3 pos   = glm::vec3(0.0f);
    float     yaw   = 0.0f; // degrees (matches Camera)
    float     pitch = 0.0f; // degrees
};
constexpr int kMaxWaypoints = 32; // fixed array so Settings stays POD (binary state blob)

// Live-tweakable parameters (driven by the ImGui panel). Grows as features land.
struct Settings {
    float sunAzimuthDeg   = 193.0f;
    float sunElevationDeg = 28.0f;
    // Sun animation: when on, az/el sweep smoothly between the min/max below
    // (driven CPU-side each frame; reaches the GPU via the existing push constants).
    bool  sunAnimate      = false;
    float sunAzimuthMin   = 157.0f;
    float sunAzimuthMax   = 214.0f;
    float sunElevationMin = 29.0f;
    float sunElevationMax = 31.0f;
    float sunAnimSpeed    = 0.2f;   // radians/sec of the sweep phase (lower = slower)
    float ambient         = 0.0f;
    float sunIntensity    = 4.0f;  // directional sun radiance multiplier
    float exposure        = 30.0f;  // tonemap exposure (manual)
    int   tonemapper      = 1;      // 0 = ACES filmic, 1 = AgX

    // Bloom (physically-based, energy-conserving — COD/Jimenez dual-filter mip chain).
    bool  bloomEnabled    = true;
    float bloomIntensity  = 0.064f; // lerp(scene, bloom) blend factor (0..1; ~physical at 0.04)
    float bloomRadius     = 1.75f;  // upsample tent filter spread (mip blend "scatter")
    float bloomThreshold  = 0.04f;  // soft-knee bright-pass threshold (0 = thresholdless/pure PBR)
    float bloomKnee       = 0.5f;   // soft-knee width around the threshold

    // Shadows (ray traced).
    bool  shadowsEnabled  = true;
    float sunAngularSize  = 0.23f;   // sun cone angle (degrees; larger = softer penumbra)
    int   shadowSamples   = 1;      // rays per pixel (1 = hard shadow)
    // Denoiser / AA selector (single UI choice; maps to the fields below):
    //   0 Off, 1 Temporal, 2 SVGF, 3 DLSS Ray Reconstruction.
    int   denoiser = 3;
    // Derived from `denoiser` by the UI each frame (engine reads these):
    //   shadowDenoiseMode 0=Off 1=Temporal 2=SVGF (auto 0 while RR active).
    int   shadowDenoiseMode = 2;
    float shadowHistAlpha   = 0.02f; // temporal EMA blend (lower = smoother, more lag)
    int   svgfIterations    = 2;    // à-trous wavelet levels (step 1,2,4,...)
    float svgfPhiNormal     = 64.0f;// normal edge-stop sharpness (higher = more edge-preserving)
    float svgfPhiDepth      = 1.0f; // depth edge-stop tolerance (lower = more edge-preserving)

    // Ambient Occlusion — ray-traced AO against the scene TLAS. Modulates the
    // ambient/indirect term only (composite: hdr = ambient*ao + direct*shadow).
    // Shares the shadow denoise stack (temporal + à-trous), so few samples suffice.
    bool  aoEnabled   = true;   // off => ao=1.0
    int   aoSamples   = 4;      // cosine-weighted hemisphere rays per pixel
    float aoRadius    = 0.6f;   // world-space max occluder distance
    float aoIntensity = 1.0f;   // strength (0 = no effect, >1 exaggerates)
    float aoBias      = 0.02f;  // ray origin offset along N (self-occlusion fix, world units)

    // Reflections — screen-space reflections on opaque surfaces (transparents do not
    // reflect). Faded by roughness; near-mirror surfaces reflect strongest.
    int   reflectionsMode    = 2;     // 0 Off, 1 SSR, 2 SSR + RT fallback, 3 RT only
    int   ssrSteps           = 48;    // screen-space march samples
    float ssrMaxDistance     = 30.0f; // world-space march length
    float ssrThickness       = 0.5f;  // depth-test tolerance (linear depth units)
    float ssrMaxRoughness    = 0.6f;  // skip reflections above this (diffuse-dominated)
    float reflectionIntensity= 1.0f;  // master reflection multiplier
    float glassFresnel       = 0.3f;  // how much grazing Fresnel firms up glass opacity

    // TAA / DLSS pipeline foundation.
    bool  taaJitter       = false;  // sub-pixel Halton jitter — auto-forced on when DLSS is active
    bool  debugMotionVecs = false;  // show motion vectors instead of shaded color

    // DLSS Ray Reconstruction. Derived from `denoiser==3` by the UI.
    bool  dlssEnabled     = false;
    int   dlssQuality     = 4;      // 0 Perf, 1 Balanced, 2 Quality, 3 UltraPerf, 4 DLAA

    // DLSS runtime status (read-only; filled by the engine for the UI).
    bool     dlssAvailable = false; // GPU/driver supports DLSS
    bool     dlssActive    = false; // actually upscaling this frame
    uint32_t renderW = 0, renderH = 0;   // render resolution
    uint32_t displayW = 0, displayH = 0; // display resolution

    // Volumetric fog — froxel-based, RT-shadowed god rays.
    bool  fogEnabled      = true;
    int   fogQuality      = 2;     // froxel grid preset (0..3); see fogGridDim() / UI labels
    float fogDensity      = 0.09f; // extinction per world unit (higher = thicker)
    float fogScatter      = 0.7f;  // in-scatter intensity (scattering albedo)
    float fogAnisotropy   = 0.0f;  // Henyey-Greenstein g (forward scattering -> beams)
    float fogAmbient      = 0.0f;  // ambient in-scatter (sky fill in the medium)
    float fogTemporalAlpha= 0.02f; // froxel temporal EMA (lower = smoother, more lag)
    float fogMaxDistance  = 20.0f; // far extent of the froxel volume (world units)
    bool  fogJitter       = true;  // per-froxel sample jitter (off = deterministic, blocky)
    int   fogBlurRadius   = 1;     // deterministic volume blur radius (0=off,1=3³,2=5³,3=7³)
    // Local fog box: confine density to a world-space box (smooth edge) instead of
    // filling the whole scene. Removes "everything is fog" outside the room and the
    // straddle leak at walls. When fogBoxManual is off, the box auto-fits the scene
    // bounds each frame (and the UI shows them); on = artist places it by hand.
    bool  fogBoxEnabled = true;
    bool  fogBoxManual  = false;
    glm::vec3 fogBoxMin = glm::vec3(0.0f);
    glm::vec3 fogBoxMax = glm::vec3(0.0f);
    float fogBoxEdge    = 0.5f;     // smoothstep falloff width at the box faces (world units)
    bool  fogDebugBox   = false;    // draw the fog box as a wireframe

    // Realtime GI — DDGI. World-space probe grid + octahedral irradiance/depth
    // atlas, sampled in resolve.comp to replace the flat ambient term.
    bool  giEnabled    = true;
    float giIntensity  = 2.0f;  // indirect irradiance multiplier (0 = old flat ambient)
    float giHysteresis = 0.988f; // probe temporal blend (higher = stabler, slower to react)
    int   giRaysPerProbe = 64;  // rays traced per probe per frame (<= 128)
    float giNormalBias = 0.15f; // shading-point offset along N (self-occlusion fix, world units)
    float giSkyIntensity = 2.0f; // sky/env contribution on ray miss (procedural sky uncalibrated)
    float giMultiBounce  = 1.0f; // multi-bounce feedback gain (1 = physical, >1 exaggerates)
    int   giProbesX = 16, giProbesY = 8, giProbesZ = 16; // grid resolution (recreate)
    bool  giRelocation = true;   // slide probes out of geometry into open space (off = grid-fixed)
    bool  giDebugProbes = false; // visualize probes as shaded spheres
    // Probe grid bounds. When giGridManual is off, the engine auto-fits these to the
    // scene each frame (and the UI shows them); when on, the artist places the grid by
    // hand (fit it inside the room to kill exterior-probe leak).
    bool  giGridManual = false;
    glm::vec3 giGridMin = glm::vec3(0.0f);
    glm::vec3 giGridMax = glm::vec3(0.0f);

    // Camera flythrough path. Waypoints are captured from the live camera; Play
    // snaps to the first and moves smoothly through them (slerp rotation).
    CamWaypoint camPath[kMaxWaypoints];
    int   camPathCount = 0;
    float camPathSpeed = 1.0f;     // world units/sec along the path
    bool  camPathLoop  = true;    // restart from the first waypoint at the end

    bool  vsync           = true;  // FIFO when on; MAILBOX/IMMEDIATE when off

    // Frame-time graph.
    bool  showFrameGraph  = true;
    int   frameGraphHz    = 10;     // samples per second fed to the graph
    float frameGraphMaxMs = 30.0f;  // fixed Y-axis top (ms); values clamp to it
};

// Light direction (pointing toward the sun) from azimuth/elevation.
inline glm::vec3 sunDirection(const Settings& s) {
    float el = glm::radians(s.sunElevationDeg);
    float az = glm::radians(s.sunAzimuthDeg);
    return glm::normalize(glm::vec3(std::cos(el) * std::cos(az),
                                    std::sin(el),
                                    std::cos(el) * std::sin(az)));
}
