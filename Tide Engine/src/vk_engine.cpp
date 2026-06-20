#include "vk_engine.h"
#include "gltf_loader.h"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
// Validation is controlled by GPU_DEBUG (independent of Debug/Release).
#ifdef GPU_DEBUG
static constexpr bool kEnableValidation = true;
#else
static constexpr bool kEnableValidation = false;
#endif

// Device extensions we require up-front so later phases (bindless / RT GI)
// need no extra device setup.
static const char* kDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

// ---------------------------------------------------------------------------
// Debug messenger
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
            TE_ERROR("[VL] %s\n", data->pMessage);
        else
            TE_WARN("[VL] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

static VkDebugUtilsMessengerCreateInfoEXT makeMessengerInfo() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    return ci;
}

// ===========================================================================
// Lifecycle
// ===========================================================================
void VulkanEngine::init() {
    initWindow();
    initInstance();
    initSurface();
    pickPhysicalDevice();
    initDevice();
    initAllocator();
    createSwapchain();
    initFrames();
    initProfiler();
    initImmediate();
    loadScene();
    m_meshPipeline = createMeshPipeline(m_device, m_swapchainFormat, m_depthFormat);
}

void VulkanEngine::run() {
    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();
    double titleAccum = 0.0;

    while (!glfwWindowShouldClose(m_window)) {
        auto now = clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;

        glfwPollEvents();
        m_camera.update(m_window, (float)dt);
        drawFrame();

        titleAccum += dt;
        if (titleAccum > 0.25) {
            char title[128];
            std::snprintf(title, sizeof(title), "Tide Engine  |  %.2f ms  |  %.0f FPS",
                          dt * 1000.0, dt > 0.0 ? 1.0 / dt : 0.0);
            glfwSetWindowTitle(m_window, title);
            titleAccum = 0.0;
        }
        FrameMark; // Tracy CPU frame boundary
    }
    vkDeviceWaitIdle(m_device);
}

// ===========================================================================
// Window
// ===========================================================================
void VulkanEngine::initWindow() {
    if (!glfwInit()) {
        TE_ERROR("glfwInit failed\n");
        std::abort();
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // no OpenGL context
    m_window = glfwCreateWindow((int)m_width, (int)m_height, "Tide Engine", nullptr, nullptr);
    if (!m_window) {
        TE_ERROR("glfwCreateWindow failed\n");
        std::abort();
    }
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* w, int, int) {
        auto* e = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(w));
        e->m_framebufferResized = true;
    });
}

// ===========================================================================
// Instance
// ===========================================================================
void VulkanEngine::initInstance() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Tide Engine";
    app.apiVersion = VK_API_VERSION_1_3;

    // Required extensions: GLFW's + (debug) debug utils.
    uint32_t glfwCount = 0;
    const char** glfwExt = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> extensions(glfwExt, glfwExt + glfwCount);

    std::vector<const char*> layers;
    if (kEnableValidation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = (uint32_t)layers.size();
    ci.ppEnabledLayerNames = layers.data();

    VkDebugUtilsMessengerCreateInfoEXT dbg = makeMessengerInfo();
    if (kEnableValidation) ci.pNext = &dbg; // catch issues during create/destroy too

    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));

    if (kEnableValidation) {
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkCreateDebugUtilsMessengerEXT");
        if (fn) VK_CHECK(fn(m_instance, &dbg, nullptr, &m_debugMessenger));
    }
}

// ===========================================================================
// Surface
// ===========================================================================
void VulkanEngine::initSurface() {
    VK_CHECK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
}

// ===========================================================================
// Physical device
// ===========================================================================
static bool hasAllExtensions(VkPhysicalDevice dev) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> avail(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, avail.data());
    for (const char* req : kDeviceExtensions) {
        bool found = false;
        for (auto& e : avail)
            if (std::strcmp(e.extensionName, req) == 0) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

void VulkanEngine::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) { TE_ERROR("No Vulkan GPU found\n"); std::abort(); }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (auto dev : devices) {
        if (!hasAllExtensions(dev)) continue;

        // Need a queue family with graphics + present.
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops.data());

        std::optional<uint32_t> family;
        for (uint32_t i = 0; i < qcount; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &present);
            if ((qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                family = i; break;
            }
        }
        if (!family) continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        m_queueFamily = *family;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            m_physicalDevice = dev;
            TE_INFO("GPU: %s\n", props.deviceName);
            return;
        }
        if (fallback == VK_NULL_HANDLE) fallback = dev;
    }

    if (fallback != VK_NULL_HANDLE) {
        m_physicalDevice = fallback;
        return;
    }
    TE_ERROR("No GPU with required extensions (swapchain/accel/ray query)\n");
    std::abort();
}

// ===========================================================================
// Logical device
// ===========================================================================
void VulkanEngine::initDevice() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    // Feature chain — enabled now so later phases are shader-only.
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{};
    rayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQuery.rayQuery = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{};
    accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accel.accelerationStructure = VK_TRUE;
    accel.pNext = &rayQuery;

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.pNext = &accel;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.bufferDeviceAddress = VK_TRUE;
    f12.descriptorIndexing = VK_TRUE;
    f12.runtimeDescriptorArray = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    f12.descriptorBindingPartiallyBound = VK_TRUE;
    f12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    f12.scalarBlockLayout = VK_TRUE;
    f12.pNext = &f13;

    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f12;

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &f2;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = (uint32_t)(sizeof(kDeviceExtensions) / sizeof(char*));
    ci.ppEnabledExtensionNames = kDeviceExtensions;

    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device));
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

    setDebugName((uint64_t)m_device, VK_OBJECT_TYPE_DEVICE, "Main Device");
    setDebugName((uint64_t)m_queue, VK_OBJECT_TYPE_QUEUE, "Graphics Queue");
}

// ===========================================================================
// VMA
// ===========================================================================
void VulkanEngine::initAllocator() {
    VmaAllocatorCreateInfo ci{};
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    ci.physicalDevice = m_physicalDevice;
    ci.device = m_device;
    ci.instance = m_instance;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&ci, &m_allocator));
}

// ===========================================================================
// Swapchain
// ===========================================================================
void VulkanEngine::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &caps);

    // Format: prefer BGRA8 SRGB.
    uint32_t fcount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &fcount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fcount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &fcount, formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { chosen = f; break; }
    }
    m_swapchainFormat = chosen.format;

    // Present mode: prefer MAILBOX, fall back to FIFO (always available).
    uint32_t pcount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &pcount, nullptr);
    std::vector<VkPresentModeKHR> modes(pcount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &pcount, modes.data());
    VkPresentModeKHR present = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) { present = m; break; }

    // Extent.
    if (caps.currentExtent.width != UINT32_MAX) {
        m_swapchainExtent = caps.currentExtent;
    } else {
        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        m_swapchainExtent.width  = std::clamp((uint32_t)w, caps.minImageExtent.width,  caps.maxImageExtent.width);
        m_swapchainExtent.height = std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = m_surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = m_swapchainExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = present;
    ci.clipped = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapchain));

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());

    m_swapchainImageViews.resize(imageCount);
    m_renderFinished.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = m_swapchainImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = m_swapchainFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_swapchainImageViews[i]));

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_renderFinished[i]));
    }

    createDepthResources();
}

void VulkanEngine::createDepthResources() {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = m_depthFormat;
    ici.extent = {m_swapchainExtent.width, m_swapchainExtent.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VK_CHECK(vmaCreateImage(m_allocator, &ici, &ai, &m_depthImage, &m_depthAlloc, nullptr));

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = m_depthImage;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = m_depthFormat;
    vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(m_device, &vci, nullptr, &m_depthView));
}

void VulkanEngine::destroyDepthResources() {
    if (m_depthView) vkDestroyImageView(m_device, m_depthView, nullptr);
    if (m_depthImage) vmaDestroyImage(m_allocator, m_depthImage, m_depthAlloc);
    m_depthView = VK_NULL_HANDLE;
    m_depthImage = VK_NULL_HANDLE;
    m_depthAlloc = VK_NULL_HANDLE;
}

void VulkanEngine::cleanupSwapchain() {
    destroyDepthResources();
    for (auto v : m_swapchainImageViews) vkDestroyImageView(m_device, v, nullptr);
    for (auto s : m_renderFinished) vkDestroySemaphore(m_device, s, nullptr);
    m_swapchainImageViews.clear();
    m_renderFinished.clear();
    if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
}

void VulkanEngine::recreateSwapchain() {
    // Wait while minimized.
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(m_window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(m_device);
    cleanupSwapchain();
    createSwapchain();
}

// ===========================================================================
// Frames
// ===========================================================================
void VulkanEngine::initFrames() {
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = m_queueFamily;
        VK_CHECK(vkCreateCommandPool(m_device, &pci, nullptr, &m_frames[i].pool));

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = m_frames[i].pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &m_frames[i].cmd));

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_frames[i].imageAvailable));

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT; // so first wait passes
        VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &m_frames[i].inFlight));
    }
}

// ===========================================================================
// Profiler
// ===========================================================================
void VulkanEngine::initProfiler() {
#ifdef GPU_PROFILE
    // Tracy manages the command buffer fully (begin/submit/wait) during init.
    m_tracyCtx = TracyVkContext(m_physicalDevice, m_device, m_queue, m_frames[0].cmd);
#endif
}

// ===========================================================================
// Immediate submit (synchronous GPU work for staging uploads)
// ===========================================================================
void VulkanEngine::initImmediate() {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = m_queueFamily;
    VK_CHECK(vkCreateCommandPool(m_device, &pci, nullptr, &m_immPool));

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m_immPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &ai, &m_immCmd));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &m_immFence));
}

void VulkanEngine::immediateSubmit(const std::function<void(VkCommandBuffer)>& fn) {
    VK_CHECK(vkResetFences(m_device, 1, &m_immFence));
    VK_CHECK(vkResetCommandPool(m_device, m_immPool, 0));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_immCmd, &bi));
    fn(m_immCmd);
    VK_CHECK(vkEndCommandBuffer(m_immCmd));

    VkCommandBufferSubmitInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdInfo.commandBuffer = m_immCmd;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    VK_CHECK(vkQueueSubmit2(m_queue, 1, &submit, m_immFence));
    VK_CHECK(vkWaitForFences(m_device, 1, &m_immFence, VK_TRUE, UINT64_MAX));
}

// ===========================================================================
// Scene load
// ===========================================================================
void VulkanEngine::loadScene() {
    // Path is relative to the working directory. Try candidates so it runs whether
    // the cwd is the Bin output dir or the solution dir.
    static const char* kCandidates[] = {
        "../Resources/nowindows/Room_NoWindows.gltf",
        "Resources/nowindows/Room_NoWindows.gltf",
        "../../Resources/nowindows/Room_NoWindows.gltf",
    };

    MeshData data;
    bool loaded = false;
    for (const char* path : kCandidates) {
        if (loadGltf(path, data)) { TE_INFO("Loaded scene: %s\n", path); loaded = true; break; }
    }
    if (!loaded) {
        TE_ERROR("glTF load failed (all candidate paths); continuing with empty scene.\n");
        return;
    }
    m_scene.build(*this, data);

    // Frame the camera to the scene bounds.
    glm::vec3 center = 0.5f * (data.boundsMin + data.boundsMax);
    float radius = glm::length(data.boundsMax - data.boundsMin) * 0.5f;
    if (radius <= 0.0f) radius = 1.0f;
    m_camera.setLookAt(center + glm::vec3(0.0f, radius * 0.3f, radius * 1.5f), center);
    m_camera.farZ = radius * 10.0f + 10.0f;
    m_camera.speed = radius * 0.5f;
    TE_INFO("Scene bounds center=(%.2f,%.2f,%.2f) radius=%.2f\n",
            center.x, center.y, center.z, radius);
}

// ===========================================================================
// Drawing
// ===========================================================================
void VulkanEngine::recordCommands(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

    {
        TracyVkZone(m_tracyCtx, cmd, "Forward");

        // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL  (+ depth UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL)
        VkImageMemoryBarrier2 toAttach[2]{};
        toAttach[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttach[0].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        toAttach[0].srcAccessMask = 0;
        toAttach[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttach[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttach[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toAttach[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttach[0].image = m_swapchainImages[imageIndex];
        toAttach[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        toAttach[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttach[1].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        toAttach[1].srcAccessMask = 0;
        toAttach[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        toAttach[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        toAttach[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toAttach[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        toAttach[1].image = m_depthImage;
        toAttach[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers = toAttach;
        vkCmdPipelineBarrier2(cmd, &dep);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = m_swapchainImageViews[imageIndex];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.02f, 0.04f, 0.10f, 1.0f}};

        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = m_depthView;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = m_swapchainExtent;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        ri.pDepthAttachment = &depth;

        vkCmdBeginRendering(cmd, &ri);

        // Dynamic viewport/scissor.
        VkViewport vp{};
        vp.width = (float)m_swapchainExtent.width;
        vp.height = (float)m_swapchainExtent.height;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D scissor{};
        scissor.extent = m_swapchainExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Draw the scene (simple forward; Faz 4 replaces this with V-buffer).
        if (m_scene.indexCount > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_meshPipeline.pipeline);

            float aspect = (float)m_swapchainExtent.width / (float)m_swapchainExtent.height;
            glm::mat4 viewProj = m_camera.proj(aspect) * m_camera.view();

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &m_scene.vertexBuffer.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, m_scene.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            for (const MeshDraw& d : m_scene.draws) {
                MeshPush push{};
                push.viewProj = viewProj;
                push.model = d.transform;
                vkCmdPushConstants(cmd, m_meshPipeline.layout, VK_SHADER_STAGE_VERTEX_BIT,
                                   0, sizeof(MeshPush), &push);
                vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, d.vertexOffset, 0);
            }
        }

        vkCmdEndRendering(cmd);

        // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC
        VkImageMemoryBarrier2 toPresent{};
        toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        toPresent.dstAccessMask = 0;
        toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.image = m_swapchainImages[imageIndex];
        toPresent.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep2{};
        dep2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep2.imageMemoryBarrierCount = 1;
        dep2.pImageMemoryBarriers = &toPresent;
        vkCmdPipelineBarrier2(cmd, &dep2);
    }

#ifdef GPU_PROFILE
    TracyVkCollect(m_tracyCtx, cmd);
#endif

    VK_CHECK(vkEndCommandBuffer(cmd));
}

void VulkanEngine::drawFrame() {
    ZoneScoped; // Tracy CPU zone

    FrameData& frame = m_frames[m_currentFrame];

    VK_CHECK(vkWaitForFences(m_device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                         frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);

    VK_CHECK(vkResetFences(m_device, 1, &frame.inFlight));
    VK_CHECK(vkResetCommandPool(m_device, frame.pool, 0));
    recordCommands(frame.cmd, imageIndex);

    VkCommandBufferSubmitInfo cmdInfo{};
    cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmdInfo.commandBuffer = frame.cmd;

    VkSemaphoreSubmitInfo wait{};
    wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait.semaphore = frame.imageAvailable;
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signal{};
    signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal.semaphore = m_renderFinished[imageIndex];
    signal.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo2 submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &wait;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signal;
    VK_CHECK(vkQueueSubmit2(m_queue, 1, &submit, frame.inFlight));

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &m_renderFinished[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &m_swapchain;
    present.pImageIndices = &imageIndex;

    VkResult pres = vkQueuePresentKHR(m_queue, &present);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
        m_framebufferResized = false;
        recreateSwapchain();
    } else if (pres != VK_SUCCESS) {
        VK_CHECK(pres);
    }

    m_currentFrame = (m_currentFrame + 1) % FRAMES_IN_FLIGHT;
}

// ===========================================================================
// Helpers / cleanup
// ===========================================================================
void VulkanEngine::setDebugName(uint64_t handle, VkObjectType type, const char* name) {
    if (!kEnableValidation) return;
    auto fn = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
        m_device, "vkSetDebugUtilsObjectNameEXT");
    if (!fn) return;
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    fn(m_device, &info);
}

void VulkanEngine::cleanup() {
#ifdef GPU_PROFILE
    if (m_tracyCtx) TracyVkDestroy(m_tracyCtx);
#endif
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroyFence(m_device, m_frames[i].inFlight, nullptr);
        vkDestroySemaphore(m_device, m_frames[i].imageAvailable, nullptr);
        vkDestroyCommandPool(m_device, m_frames[i].pool, nullptr);
    }
    if (m_immFence) vkDestroyFence(m_device, m_immFence, nullptr);
    if (m_immPool) vkDestroyCommandPool(m_device, m_immPool, nullptr);
    destroyPipeline(m_device, m_meshPipeline);
    cleanupSwapchain();
    m_scene.destroy(m_allocator);
    if (m_allocator) vmaDestroyAllocator(m_allocator);
    if (m_device) vkDestroyDevice(m_device, nullptr);
    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}
