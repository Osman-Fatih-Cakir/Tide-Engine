#pragma once
#include "pch.h"
#include "gpu_buffer.h"

class VulkanEngine;
class Scene;

// One acceleration structure + the buffer that backs it.
struct AccelStructure {
    VkAccelerationStructureKHR handle  = VK_NULL_HANDLE;
    Buffer                     buffer{};
    VkDeviceAddress            address = 0;
};

// BLAS-per-draw + a single TLAS over the scene. Static scene -> built once.
// Used by ray-query shadow rays now and RTGI rays later (same structure).
struct SceneAccel {
    std::vector<AccelStructure> blas;
    AccelStructure              tlas;
    Buffer                      instanceBuffer{}; // host-visible TLAS instances

    void destroy(VkDevice device, VmaAllocator alloc);
};

// Build BLAS for every draw and a TLAS instancing them with per-draw transforms.
// Instance mask: BLEND (glass) draws get 0x02 (shadow rays cull them); others
// 0x01. instanceCustomIndex = drawID so hits can look up material/UV for the
// alpha-mask test. Records into one immediate submit.
void buildSceneAccel(VulkanEngine& eng, Scene& scene);
