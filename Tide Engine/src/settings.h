#pragma once
#include "pch.h"

// Live-tweakable parameters (driven by the ImGui panel). Grows as features land.
struct Settings {
    float sunAzimuthDeg   = 193.0f;
    float sunElevationDeg = 28.0f;
    // Sun animation: when on, az/el sweep smoothly between the min/max below
    // (driven CPU-side each frame; reaches the GPU via the existing push constants).
    bool  sunAnimate      = false;
    float sunAzimuthMin   = 130.0f;
    float sunAzimuthMax   = 214.0f;
    float sunElevationMin = 18.0f;
    float sunElevationMax = 40.0f;
    float sunAnimSpeed    = 0.2f;   // radians/sec of the sweep phase (lower = slower)
    float ambient         = 0.33f;
    float sunIntensity    = 4.0f;  // directional sun radiance multiplier
    float exposure        = 1.0f;  // tonemap exposure

    // Shadows (ray traced).
    bool  shadowsEnabled  = true;
    float sunAngularSize  = 0.23f;   // sun cone angle (degrees; larger = softer penumbra)
    int   shadowSamples   = 1;      // rays per pixel (1 = hard shadow)
    // Denoiser / AA selector (single UI choice; maps to the fields below):
    //   0 Off, 1 Temporal, 2 SVGF, 3 DLSS Ray Reconstruction.
    int   denoiser = 2;
    // Derived from `denoiser` by the UI each frame (engine reads these):
    //   shadowDenoiseMode 0=Off 1=Temporal 2=SVGF (auto 0 while RR active).
    int   shadowDenoiseMode = 2;
    float shadowHistAlpha   = 0.02f; // temporal EMA blend (lower = smoother, more lag)
    int   svgfIterations    = 2;    // à-trous wavelet levels (step 1,2,4,...)
    float svgfPhiNormal     = 64.0f;// normal edge-stop sharpness (higher = more edge-preserving)
    float svgfPhiDepth      = 1.0f; // depth edge-stop tolerance (lower = more edge-preserving)

    // TAA / DLSS pipeline foundation (Faz 6.5 Aşama A).
    bool  taaJitter       = false;  // sub-pixel Halton jitter — auto-forced on when DLSS is active
    bool  debugMotionVecs = false;  // show motion vectors instead of shaded color

    // DLSS Ray Reconstruction (Faz 6.5). Derived from `denoiser==3` by the UI.
    bool  dlssEnabled     = false;
    int   dlssQuality     = 2;      // 0 Perf, 1 Balanced, 2 Quality, 3 UltraPerf, 4 DLAA

    // DLSS runtime status (read-only; filled by the engine for the UI).
    bool     dlssAvailable = false; // GPU/driver supports DLSS
    bool     dlssActive    = false; // actually upscaling this frame
    uint32_t renderW = 0, renderH = 0;   // render resolution
    uint32_t displayW = 0, displayH = 0; // display resolution

    // Volumetric fog (Faz 7) — froxel-based, RT-shadowed god rays.
    bool  fogEnabled      = false; // default off so density-0 path is a clean regression
    int   fogQuality      = 2;     // froxel grid preset (0..3); see fogGridDim() / UI labels
    float fogDensity      = 0.04f; // extinction per world unit (higher = thicker)
    float fogScatter      = 0.7f;  // in-scatter intensity (scattering albedo)
    float fogAnisotropy   = 0.76f; // Henyey-Greenstein g (forward scattering -> beams)
    float fogAmbient      = 0.06f; // ambient in-scatter (sky fill in the medium)
    float fogTemporalAlpha= 0.02f; // froxel temporal EMA (lower = smoother, more lag)
    float fogMaxDistance  = 40.0f; // far extent of the froxel volume (world units)
    bool  fogJitter       = true;  // per-froxel sample jitter (off = deterministic, blocky)
    bool  fogDepthCull    = true;  // kill froxels not fully in front of the surface (no leak)
    int   fogBlurRadius   = 1;     // deterministic volume blur radius (0=off,1=3³,2=5³,3=7³)

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
