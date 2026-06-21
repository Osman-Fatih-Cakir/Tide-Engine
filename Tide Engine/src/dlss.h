#pragma once
#include "pch.h"
#include <string>
#include <vector>

// Forward-declare NGX types so this header doesn't pull in the NGX SDK.
struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;

// DLSS Super Resolution via raw NVIDIA NGX (no Streamline abstraction).
// Renders the scene at a lower resolution and upscales to display resolution,
// using jitter + motion vectors + depth (Faz 6.5). Hardcoded, single feature.
class Dlss {
public:
    // Quality presets (map to NVSDK_NGX_PerfQuality_Value). DLAA = native res.
    enum class Quality { Performance, Balanced, Quality, UltraPerformance, DLAA };

    // NGX-required instance/device extensions — query BEFORE creating them and
    // add to the create-infos. Static (no init needed).
    static std::vector<std::string> requiredInstanceExtensions();
    static std::vector<std::string> requiredDeviceExtensions();

    // Init NGX after the device exists. Returns false if DLSS isn't available.
    bool init(VkInstance instance, VkPhysicalDevice phys, VkDevice device);
    void shutdown();
    bool available() const { return m_available; }

    // Optimal render resolution for a display size + quality. Falls back to
    // display size (DLAA) on failure.
    VkExtent2D optimalRenderExtent(VkExtent2D display, Quality q);

    // (Re)create the upscaling feature for the given sizes. Needs a one-time
    // command buffer (records nothing the caller must sync beyond submit).
    bool createFeature(VkCommandBuffer cmd, VkExtent2D renderExtent,
                       VkExtent2D displayExtent, Quality q);
    void releaseFeature();
    bool hasFeature() const { return m_feature != nullptr; }

    // Per-frame upscale. color/depth/motion are render-res inputs, out is
    // display-res. ALL must be in VK_IMAGE_LAYOUT_GENERAL. jitterPixels in
    // render-pixel space [-0.5,0.5]. reset=true drops temporal history.
    void evaluate(VkCommandBuffer cmd,
                  VkImage colorImg, VkImageView colorView,
                  VkImage depthImg, VkImageView depthView,
                  VkImage motionImg, VkImageView motionView,
                  VkImage outImg, VkImageView outView,
                  VkExtent2D renderExtent, glm::vec2 jitterPixels, bool reset);

private:
    bool                 m_available    = false;
    VkDevice             m_device       = VK_NULL_HANDLE;
    NVSDK_NGX_Parameter* m_params       = nullptr;
    NVSDK_NGX_Handle*    m_feature      = nullptr;
    VkExtent2D           m_displayExtent{}; // output resource dimensions
};
