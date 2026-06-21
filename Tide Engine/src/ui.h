#pragma once
#include "pch.h"

class VulkanEngine;
struct Settings;

// Thin ImGui wrapper (GLFW + Vulkan backend, dynamic rendering).
class Ui {
public:
    void init(VulkanEngine& eng, VkFormat colorFormat,
              uint32_t minImageCount, uint32_t imageCount);
    void beginFrame();
    // dt = real wall-clock delta (timing/pacing). cpuMs = measured CPU work time
    // this frame (excludes VSync/fence idle) — what the graph/stats display.
    void buildPanel(Settings& s, float dt, float cpuMs);
    void render(VkCommandBuffer cmd);
    void destroy();

private:
    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED; // must outlive ImGui_ImplVulkan_Init

    // Frame-time stats, refreshed once per second. Two metrics:
    //  - real frame time (wall-clock dt) -> actual on-screen FPS (VSync-capped)
    //  - CPU work time (cpuMs)           -> engine cost / headroom (excl. idle)
    float m_acc         = 0.0f; // accumulated REAL time this second (pacing)
    int   m_frames      = 0;    // frames this second
    float m_realMax     = 0.0f; // slowest real frame this second (s)
    float m_cpuAcc      = 0.0f; // accumulated CPU ms this second
    float m_dispAvgMs   = 0.0f; // displayed: average REAL frame time (ms)
    float m_dispMaxMs   = 0.0f; // displayed: worst REAL frame time (ms)
    float m_dispCpuMs   = 0.0f; // displayed: average CPU work time (ms)

    // Rolling frame-time history (ms) for the plot. Sampled at most 100x/sec
    // (one point per ~10 ms window, value = avg frame time in that window).
    static constexpr int kHistory = 120;
    float m_msHistory[kHistory] = {};
    int   m_msHead = 0;            // next write slot (ring buffer)
    float m_plotAcc = 0.0f;        // REAL time accumulated toward the next sample
    int   m_plotFrames = 0;        // frames accumulated toward the next sample
};
