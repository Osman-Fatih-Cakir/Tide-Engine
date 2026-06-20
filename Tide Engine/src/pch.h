#pragma once

// --- Standard ---
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <set>
#include <optional>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <functional>

// --- Windows (for OutputDebugString logging; no console window) ---
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// --- Vulkan + GLFW (GLFW_INCLUDE_VULKAN is defined in project settings) ---
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

// --- VMA (declarations only here; implementation lives in vma_impl.cpp) ---
#include <vma/vk_mem_alloc.h>

// --- GLM (Vulkan depth range 0..1, radians) ---
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// --- Tracy (no-op automatically when TRACY_ENABLE is not defined) ---
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

// Log to the Visual Studio Output window (no console window needed).
// Every line is prefixed with [TIDE][LEVEL] so our output is easy to spot/filter
// among the DLL-load spam.
inline void TE_LogLevel(const char* level, const char* fmt, ...) {
    char body[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(body, sizeof(body), fmt, args);
    va_end(args);

    char line[1152];
    std::snprintf(line, sizeof(line), "[TIDE][%s] %s", level, body);
    OutputDebugStringA(line);
}

#define TE_INFO(...)  TE_LogLevel("INFO",  __VA_ARGS__)
#define TE_WARN(...)  TE_LogLevel("WARN",  __VA_ARGS__)
#define TE_ERROR(...) TE_LogLevel("ERROR", __VA_ARGS__)

// Abort-on-failure check. Keep it brutal and simple for a hardcoded demo.
#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult _err = (x);                                                   \
        if (_err != VK_SUCCESS) {                                              \
            TE_ERROR("VK_CHECK failed: VkResult=%d at %s:%d\n",                \
                     (int)_err, __FILE__, __LINE__);                           \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
