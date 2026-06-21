#pragma once
#include "pch.h"

// Live-tweakable parameters (driven by the ImGui panel). Grows as features land.
struct Settings {
    float sunAzimuthDeg   = 193.0f;
    float sunElevationDeg = 28.0f;
    float ambient         = 0.66f;
    float sunIntensity    = 4.0f;  // directional sun radiance multiplier
    float exposure        = 1.0f;  // tonemap exposure

    // Shadows (PCSS).
    bool  shadowsEnabled  = true;
    float sunAngularSize  = 1.5f;   // PCSS softness (degrees; larger = softer penumbra)
    float shadowBias      = 0.0015f;// depth bias in light NDC (acne vs peter-panning)
    float shadowNormalBias= 0.05f;  // world-space offset along the normal before sampling

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
