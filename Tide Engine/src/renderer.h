#pragma once
#include "pch.h"
#include "gpu_image.h"
#include "settings.h"

class VulkanEngine;
class Scene;
class Dlss;

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

    // (Re)create the targets. renderExtent sizes vis/hdr/shadow/motion/depth;
    // displayExtent sizes the DLSS output. Call on init and every resize/mode change.
    void createTargets(VulkanEngine& eng, VkExtent2D renderExtent, VkExtent2D displayExtent);
    void destroyTargets(VulkanEngine& eng);

    void destroy(VulkanEngine& eng);

    // Render-resolution HDR image (DLSS input) so the engine can recreate the
    // feature against the exact resource. Valid after createTargets.
    VkExtent2D renderExtent() const { return m_extent; }

    // Record the passes: Visibility -> Resolve -> Transparent -> [DLSS upscale] ->
    // Tonemap. Passes A-C run at renderExtent; tonemap at displayExtent. When
    // dlssActive, upscales render-res HDR to display-res before tonemap.
    // Leaves the swapchain image in COLOR_ATTACHMENT_OPTIMAL.
    void record(VkCommandBuffer cmd, const Scene& scene,
                const glm::mat4& viewProj, const glm::vec3& cameraPos,
                const Settings& settings,
                VkExtent2D renderExtent, VkExtent2D displayExtent,
                VkImage swapchainImage, VkImageView swapchainView,
                VkImage depthImage, VkImageView depthView,
                Dlss* dlss, bool dlssActive,
                TracyVkCtx tracy);

private:
    VulkanEngine* m_eng = nullptr; // for cmd debug labels in record()
    uint32_t      m_frameIndex = 0; // dither/jitter seed for RT shadow sampling
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_depthFormat     = VK_FORMAT_UNDEFINED;

    // Screen-sized targets.
    Image      m_vis{};                 // R32_UINT packed (instanceID<<20 | primID)
    Image      m_hdr{};                 // RGBA16F linear radiance
    Image      m_shadowHist[2]{};       // RG16F temporal shadow: R=visibility, G=linear depth
    Image      m_motion{};              // RG16F screen-space motion vectors (prev - cur UV), for DLSS
    Image      m_dlssOutput{};          // RGBA16F display-res HDR (DLSS upscale target)
    VkExtent2D m_extent = {};           // render resolution
    VkExtent2D m_displayExtent = {};
    bool       m_dlssReset = true;      // drop DLSS temporal history (set on (re)create)
    VkSampler  m_hdrSampler = VK_NULL_HANDLE;
    VkSampler  m_histSampler = VK_NULL_HANDLE;

    // Temporal shadow state.
    uint32_t   m_histIndex = 0;         // which m_shadowHist is "current" (write)
    bool       m_haveHistory = false;
    glm::mat4  m_prevViewProj = glm::mat4(1.0f);
    float      m_prevSunAz = 1e9f, m_prevSunEl = 1e9f;

    // Visibility pass (raster).
    VkPipeline       m_visPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_visLayout   = VK_NULL_HANDLE;

    // Transparent forward pass (blends into HDR after the opaque resolve).
    VkPipeline       m_transparentPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_transparentLayout   = VK_NULL_HANDLE;

    // Resolve pass (compute): set0 = scene, set1 = { vis, hdr, shadowHist read/write }.
    // Two ping-pong sets so this frame reads last frame's history and writes the other.
    VkPipeline            m_resolvePipeline = VK_NULL_HANDLE;
    VkPipelineLayout      m_resolveLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_resolveSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet       m_resolveSet[2]    = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Tonemap pass (fullscreen fragment): set0 = { hdr sampled }.
    VkPipeline            m_tonemapPipeline = VK_NULL_HANDLE;
    VkPipelineLayout      m_tonemapLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet       m_tonemapSet       = VK_NULL_HANDLE; // samples m_hdr (no DLSS)
    VkDescriptorSet       m_tonemapSetDlss   = VK_NULL_HANDLE; // samples m_dlssOutput

    VkDescriptorPool m_pool = VK_NULL_HANDLE;
};
