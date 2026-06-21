#pragma once
#include "pch.h"
#include "gpu_image.h"
#include "settings.h"

class VulkanEngine;
class Scene;

// Visibility-buffer renderer (Faz 4). Owns the screen-sized vis (R32_UINT) and
// HDR (RGBA16F) targets and the three passes:
//   Visibility (raster) -> Resolve (compute, PBR -> HDR) -> Tonemap (HDR -> swapchain).
// Hardcoded on purpose; no abstraction layers.
class Renderer {
public:
    // One-time setup: pipelines + descriptor layouts. sceneSetLayout is the
    // bindless scene set (materials/textures/vertices/indices/draws).
    void init(VulkanEngine& eng, VkFormat swapchainFormat, VkFormat depthFormat,
              VkDescriptorSetLayout sceneSetLayout);

    // (Re)create the screen-sized targets; call on init and every resize.
    void createTargets(VulkanEngine& eng, VkExtent2D extent);
    void destroyTargets(VulkanEngine& eng);

    void destroy(VulkanEngine& eng);

    // Record the three passes. Leaves the swapchain image in
    // COLOR_ATTACHMENT_OPTIMAL (the ImGui pass loads it afterwards).
    void record(VkCommandBuffer cmd, const Scene& scene,
                const glm::mat4& viewProj, const glm::vec3& cameraPos,
                const Settings& settings, VkExtent2D extent,
                VkImage swapchainImage, VkImageView swapchainView,
                VkImage depthImage, VkImageView depthView,
                TracyVkCtx tracy);

private:
    VulkanEngine* m_eng = nullptr; // for cmd debug labels in record()
    uint32_t      m_frameIndex = 0; // dither/jitter seed for RT shadow sampling
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat     = VK_FORMAT_UNDEFINED;

    // Screen-sized targets.
    Image      m_vis{};                 // R32_UINT packed (drawID<<20 | primID)
    Image      m_hdr{};                 // RGBA16F linear radiance
    VkExtent2D m_extent = {};
    VkSampler  m_hdrSampler = VK_NULL_HANDLE;

    // Visibility pass (raster).
    VkPipeline       m_visPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_visLayout   = VK_NULL_HANDLE;

    // Transparent forward pass (blends into HDR after the opaque resolve).
    VkPipeline       m_transparentPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_transparentLayout   = VK_NULL_HANDLE;

    // Resolve pass (compute): set0 = scene, set1 = { vis storage, hdr storage }.
    VkPipeline            m_resolvePipeline = VK_NULL_HANDLE;
    VkPipelineLayout      m_resolveLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_resolveSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet       m_resolveSet       = VK_NULL_HANDLE;

    // Tonemap pass (fullscreen fragment): set0 = { hdr sampled }.
    VkPipeline            m_tonemapPipeline = VK_NULL_HANDLE;
    VkPipelineLayout      m_tonemapLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet       m_tonemapSet       = VK_NULL_HANDLE;

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
};
