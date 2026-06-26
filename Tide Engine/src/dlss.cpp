#include "dlss.h"

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssd.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

#include <glm/gtc/type_ptr.hpp>

// Arbitrary project id + custom engine — lets NGX init without a whitelisted
// NVIDIA application id (dev/research path). MUST be a valid UUID (hex only),
// else NGX returns FAIL_InvalidParameter.
static const char* kProjectId = "a1b2c3d4-e5f6-47a8-9b0c-d1e2f3a4b5c6";

static NVSDK_NGX_PerfQuality_Value toNgxQuality(Dlss::Quality q) {
    switch (q) {
        case Dlss::Quality::Performance:      return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case Dlss::Quality::Balanced:         return NVSDK_NGX_PerfQuality_Value_Balanced;
        case Dlss::Quality::Quality:          return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case Dlss::Quality::UltraPerformance: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
        case Dlss::Quality::DLAA:             return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    return NVSDK_NGX_PerfQuality_Value_MaxQuality;
}

// ---------------------------------------------------------------------------
// Required extensions (static; queried before instance/device creation).
// ---------------------------------------------------------------------------
std::vector<std::string> Dlss::requiredInstanceExtensions() {
    unsigned int iCount = 0, dCount = 0;
    const char** iExts = nullptr; const char** dExts = nullptr;
    std::vector<std::string> out;
    if (NVSDK_NGX_VULKAN_RequiredExtensions(&iCount, &iExts, &dCount, &dExts) == NVSDK_NGX_Result_Success)
        for (unsigned int i = 0; i < iCount; i++) out.emplace_back(iExts[i]);
    return out;
}
std::vector<std::string> Dlss::requiredDeviceExtensions() {
    unsigned int iCount = 0, dCount = 0;
    const char** iExts = nullptr; const char** dExts = nullptr;
    std::vector<std::string> out;
    if (NVSDK_NGX_VULKAN_RequiredExtensions(&iCount, &iExts, &dCount, &dExts) == NVSDK_NGX_Result_Success)
        for (unsigned int i = 0; i < dCount; i++) out.emplace_back(dExts[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------
bool Dlss::init(VkInstance instance, VkPhysicalDevice phys, VkDevice device) {
    m_device = device;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L".",
        instance, phys, device);
    if (NVSDK_NGX_FAILED(r)) {
        TE_WARN("DLSS: NGX init failed (0x%08x %ls)\n", r, GetNGXResultAsString(r));
        return false;
    }
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_params))) {
        TE_WARN("DLSS: GetCapabilityParameters failed\n");
        NVSDK_NGX_VULKAN_Shutdown1(device);
        return false;
    }
    // We use DLSS Ray Reconstruction (DLSS-D), so query the denoising capability,
    // not the plain Super Resolution one.
    int supported = 0;
    NVSDK_NGX_Parameter_GetI(m_params, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &supported);
    if (!supported) {
        int initResult = 0;
        NVSDK_NGX_Parameter_GetI(m_params, NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &initResult);
        TE_WARN("DLSS Ray Reconstruction: not available on this GPU/driver (initResult 0x%08x)\n", initResult);
        NVSDK_NGX_VULKAN_Shutdown1(device);
        return false;
    }
    m_available = true;
    TE_INFO("DLSS Ray Reconstruction: available\n");
    return true;
}

void Dlss::shutdown() {
    releaseFeature();
    if (m_device) NVSDK_NGX_VULKAN_Shutdown1(m_device);
    m_params = nullptr;
    m_available = false;
}

VkExtent2D Dlss::optimalRenderExtent(VkExtent2D display, Quality q) {
    VkExtent2D out = display;
    if (!m_available) return out;
    unsigned int ow = 0, oh = 0, maxw = 0, maxh = 0, minw = 0, minh = 0;
    float sharp = 0.0f;
    NVSDK_NGX_Result r = NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        m_params, display.width, display.height, toNgxQuality(q),
        &ow, &oh, &maxw, &maxh, &minw, &minh, &sharp);
    if (NVSDK_NGX_SUCCEED(r) && ow > 0 && oh > 0) { out.width = ow; out.height = oh; }
    return out;
}

bool Dlss::createFeature(VkCommandBuffer cmd, VkExtent2D renderExtent,
                         VkExtent2D displayExtent, Quality q) {
    releaseFeature();
    if (!m_available) return false;
    m_displayExtent = displayExtent;

    NVSDK_NGX_DLSSD_Create_Params p{};
    p.InDenoiseMode   = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    p.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed; // roughness in normals.w
    p.InUseHWDepth    = NVSDK_NGX_DLSS_Depth_Type_HW;         // raw D32 depth
    p.InWidth        = renderExtent.width;
    p.InHeight       = renderExtent.height;
    p.InTargetWidth  = displayExtent.width;
    p.InTargetHeight = displayExtent.height;
    p.InPerfQualityValue = toNgxQuality(q);
    // Linear HDR color, auto exposure (no exposure texture), motion at render res.
    p.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
        NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

    NVSDK_NGX_Result r =
        NGX_VULKAN_CREATE_DLSSD_EXT1(m_device, cmd, 1, 1, &m_feature, m_params, &p);
    if (NVSDK_NGX_FAILED(r)) {
        TE_WARN("DLSS: CreateFeature failed (0x%08x %ls)\n", r, GetNGXResultAsString(r));
        m_feature = nullptr;
        return false;
    }
    TE_INFO("DLSS: feature created %ux%u -> %ux%u\n",
            renderExtent.width, renderExtent.height, displayExtent.width, displayExtent.height);
    return true;
}

void Dlss::releaseFeature() {
    if (m_feature) { NVSDK_NGX_VULKAN_ReleaseFeature(m_feature); m_feature = nullptr; }
}

void Dlss::evaluate(VkCommandBuffer cmd,
                    VkImage colorImg, VkImageView colorView,
                    VkImage depthImg, VkImageView depthView,
                    VkImage motionImg, VkImageView motionView,
                    VkImage diffuseImg, VkImageView diffuseView,
                    VkImage specularImg, VkImageView specularView,
                    VkImage normalImg, VkImageView normalView,
                    VkImage outImg, VkImageView outView,
                    VkExtent2D renderExtent, glm::vec2 jitterPixels,
                    const glm::mat4& view, const glm::mat4& proj, bool reset) {
    if (!m_feature) return;

    VkImageSubresourceRange colorR{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageSubresourceRange depthR{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    const uint32_t rw = renderExtent.width, rh = renderExtent.height;

    NVSDK_NGX_Resource_VK color = NVSDK_NGX_Create_ImageView_Resource_VK(
        colorView, colorImg, colorR, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, false);
    NVSDK_NGX_Resource_VK depth = NVSDK_NGX_Create_ImageView_Resource_VK(
        depthView, depthImg, depthR, VK_FORMAT_D32_SFLOAT, rw, rh, false);
    NVSDK_NGX_Resource_VK motion = NVSDK_NGX_Create_ImageView_Resource_VK(
        motionView, motionImg, colorR, VK_FORMAT_R16G16_SFLOAT, rw, rh, false);
    // Ray Reconstruction guide buffers (all render-res).
    NVSDK_NGX_Resource_VK diffuse = NVSDK_NGX_Create_ImageView_Resource_VK(
        diffuseView, diffuseImg, colorR, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, false);
    NVSDK_NGX_Resource_VK specular = NVSDK_NGX_Create_ImageView_Resource_VK(
        specularView, specularImg, colorR, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, false);
    // World-space normal in xyz, roughness packed in w (Roughness_Mode_Packed).
    NVSDK_NGX_Resource_VK normal = NVSDK_NGX_Create_ImageView_Resource_VK(
        normalView, normalImg, colorR, VK_FORMAT_R16G16B16A16_SFLOAT, rw, rh, false);
    // Output is display-res (must match the feature's target dimensions).
    NVSDK_NGX_Resource_VK out = NVSDK_NGX_Create_ImageView_Resource_VK(
        outView, outImg, colorR, VK_FORMAT_R16G16B16A16_SFLOAT,
        m_displayExtent.width, m_displayExtent.height, true);

    // Camera matrices: glm is column-major; NGX takes float[16]. Copy to mutable
    // locals (the eval struct wants non-const float*).
    glm::mat4 worldToView = view;
    glm::mat4 viewToClip  = proj;

    NVSDK_NGX_VK_DLSSD_Eval_Params e{};
    e.pInColor             = &color;
    e.pInOutput            = &out;
    e.pInDepth             = &depth;
    e.pInMotionVectors     = &motion;
    e.pInDiffuseAlbedo     = &diffuse;
    e.pInSpecularAlbedo    = &specular;
    e.pInNormals           = &normal;
    e.pInRoughness         = &normal; // packed in normals.w; same resource
    e.pInWorldToViewMatrix = glm::value_ptr(worldToView);
    e.pInViewToClipMatrix  = glm::value_ptr(viewToClip);
    e.InJitterOffsetX      = jitterPixels.x;
    e.InJitterOffsetY      = jitterPixels.y;
    e.InRenderSubrectDimensions.Width  = rw;
    e.InRenderSubrectDimensions.Height = rh;
    e.InReset              = reset ? 1 : 0;
    // Motion is stored as UV delta; scale to render-pixel space.
    e.InMVScaleX           = (float)rw;
    e.InMVScaleY           = (float)rh;

    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, m_feature, m_params, &e);
    if (NVSDK_NGX_FAILED(r))
        TE_WARN("DLSS RR: evaluate failed (0x%08x %ls)\n", r, GetNGXResultAsString(r));
}
