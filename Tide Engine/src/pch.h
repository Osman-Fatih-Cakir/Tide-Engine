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

// --- Windows (for OutputDebugString logging; no console window) ---
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// --- Vulkan + GLFW (GLFW_INCLUDE_VULKAN is defined in project settings) ---
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

// --- VMA (declarations only here; implementation lives in vma_impl.cpp) ---
#include <vma/vk_mem_alloc.h>

// --- Tracy (no-op automatically when TRACY_ENABLE is not defined) ---
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

// Log to the Visual Studio Output window (no console window needed).
inline void TE_Log(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    OutputDebugStringA(buf);
}

// Abort-on-failure check. Keep it brutal and simple for a hardcoded demo.
#define VK_CHECK(x)                                                            \
    do {                                                                       \
        VkResult _err = (x);                                                   \
        if (_err != VK_SUCCESS) {                                              \
            TE_Log("[VK_CHECK] VkResult=%d at %s:%d\n",                        \
                   (int)_err, __FILE__, __LINE__);                             \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
