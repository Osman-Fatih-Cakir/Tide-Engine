#pragma once
#include "pch.h"
#include "scene.h"
#include "camera.h"
#include "renderer.h"
#include "ui.h"
#include "settings.h"
#include "dlss.h"

// Number of frames the CPU may work ahead of the GPU.
inline constexpr uint32_t FRAMES_IN_FLIGHT = 2;

// Per-frame-in-flight CPU/GPU sync + recording objects.
struct FrameData {
    VkCommandPool   pool           = VK_NULL_HANDLE;
    VkCommandBuffer cmd            = VK_NULL_HANDLE;
    VkSemaphore     imageAvailable = VK_NULL_HANDLE;
    VkFence         inFlight       = VK_NULL_HANDLE;
};

// The whole engine: window, Vulkan context, swapchain, scene, and the (temporary)
// forward pass. Hardcoded on purpose; no abstraction layers.
class VulkanEngine {
public:
    void init();
    void run();
    void cleanup();

    // --- accessors / utilities used by uploaders (Scene, etc.) ---
    VkDevice         device() const { return m_device; }
    VmaAllocator     allocator() const { return m_allocator; }
    VkInstance       instance() const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    uint32_t         queueFamily() const { return m_queueFamily; }
    VkQueue          queue() const { return m_queue; }
    GLFWwindow*      window() const { return m_window; }
    // Run GPU work synchronously (staging copies, etc.).
    void immediateSubmit(const std::function<void(VkCommandBuffer)>& fn);
    void setDebugName(uint64_t handle, VkObjectType type, const char* name);
    // RenderDoc/Nsight command-buffer region labels (no-op without validation).
    void cmdBeginLabel(VkCommandBuffer cmd, const char* name);
    void cmdEndLabel(VkCommandBuffer cmd);

private:
    // --- setup steps ---
    void initWindow();
    void initInstance();
    void initSurface();
    void pickPhysicalDevice();
    void initDevice();
    void initAllocator();
    void createSwapchain();
    void cleanupSwapchain();
    void recreateSwapchain();
    void createDepthResources();
    void destroyDepthResources();
    VkExtent2D computeRenderExtent();  // render res from DLSS mode (display res if off)
    void recreateDlssFeature();        // (re)create the DLSS feature for current sizes
    void initFrames();
    void initProfiler();
    void initImmediate();
    void loadScene();

    // --- per-frame ---
    void drawFrame(float dt);
    void recordCommands(VkCommandBuffer cmd, uint32_t imageIndex);

    // --- window ---
    GLFWwindow* m_window           = nullptr;
    uint32_t    m_width            = 1600;
    uint32_t    m_height           = 900;
    bool        m_framebufferResized = false;
    bool        m_lastVsync          = true;  // tracks present-mode changes from UI
    float       m_lastCpuMs          = 0.0f;  // measured CPU work time (excl. VSync idle)

    // --- core ---
    VkInstance               m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             m_surface        = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    uint32_t                 m_queueFamily    = 0;
    VkQueue                  m_queue          = VK_NULL_HANDLE;
    VmaAllocator             m_allocator      = VK_NULL_HANDLE;

    // --- swapchain ---
    VkSwapchainKHR           m_swapchain      = VK_NULL_HANDLE;
    VkFormat                 m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D               m_swapchainExtent = {};  // display resolution
    VkExtent2D               m_renderExtent    = {};  // render resolution (<= display with DLSS)
    std::vector<VkImage>     m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    // Signalled when rendering to a given swapchain image is done (one per image).
    std::vector<VkSemaphore> m_renderFinished;

    // --- frames ---
    FrameData m_frames[FRAMES_IN_FLIGHT];
    uint32_t  m_currentFrame = 0;

    // --- immediate submit (staging uploads) ---
    VkCommandPool   m_immPool  = VK_NULL_HANDLE;
    VkCommandBuffer m_immCmd   = VK_NULL_HANDLE;
    VkFence         m_immFence = VK_NULL_HANDLE;

    // --- scene ---
    Scene m_scene;

    // --- depth ---
    VkImage       m_depthImage = VK_NULL_HANDLE;
    VmaAllocation m_depthAlloc = VK_NULL_HANDLE;
    VkImageView   m_depthView  = VK_NULL_HANDLE;
    VkFormat      m_depthFormat = VK_FORMAT_D32_SFLOAT;

    // --- visibility-buffer renderer + camera ---
    Renderer m_renderer;
    Camera   m_camera;

    // --- debug UI + tunables ---
    Ui       m_ui;
    Settings m_settings;

    // --- DLSS (super resolution) ---
    Dlss m_dlss;
    bool m_lastDlssEnabled = true;
    int  m_lastDlssQuality = -1;

    // --- profiling ---
    TracyVkCtx m_tracyCtx = nullptr;
};
