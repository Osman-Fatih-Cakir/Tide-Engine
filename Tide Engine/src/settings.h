#pragma once
#include "pch.h"

// Live-tweakable parameters (driven by the ImGui panel). Grows as features land.
struct Settings {
    float sunAzimuthDeg   = 193.0f;
    float sunElevationDeg = 28.0f;
    float ambient         = 0.66f;
    float sunIntensity    = 4.0f;  // directional sun radiance multiplier
    float exposure        = 1.0f;  // tonemap exposure

    // Shadows (ray traced).
    bool  shadowsEnabled  = true;
    float sunAngularSize  = 0.5f;   // sun cone angle (degrees; larger = softer penumbra)
    int   shadowSamples   = 4;      // rays per pixel (1 = hard shadow)

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
