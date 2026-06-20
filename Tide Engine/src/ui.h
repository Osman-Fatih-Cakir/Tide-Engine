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
    void buildPanel(Settings& s, float dt);
    void render(VkCommandBuffer cmd);
    void destroy();

private:
    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED; // must outlive ImGui_ImplVulkan_Init

    // Frame-time stats, refreshed once per second.
    float m_acc        = 0.0f;  // accumulated time this second
    int   m_frames     = 0;     // frames this second
    float m_maxDt      = 0.0f;  // slowest frame (largest dt) this second
    float m_dispAvgMs  = 0.0f;  // displayed: average frame time
    float m_dispMaxMs  = 0.0f;  // displayed: worst (slowest) frame time

    // Rolling frame-time history (ms) for the plot. Sampled at most 100x/sec
    // (one point per ~10 ms window, value = avg frame time in that window).
    static constexpr int kHistory = 120;
    float m_msHistory[kHistory] = {};
    int   m_msHead = 0;            // next write slot (ring buffer)
    float m_plotAcc = 0.0f;        // time accumulated toward the next sample
    int   m_plotFrames = 0;        // frames accumulated toward the next sample
};
