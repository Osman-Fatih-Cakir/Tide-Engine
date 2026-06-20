#pragma once
#include "mesh.h"
#include "gpu_buffer.h"

class VulkanEngine;

// GPU-resident scene: one big vertex buffer, one big index buffer, one material
// SSBO. Buffers are RT-ready (device address + accel build input). Bindless
// textures + drawing come in later steps.
class Scene {
public:
    void build(VulkanEngine& eng, const MeshData& data);
    void destroy(VmaAllocator alloc);

    Buffer vertexBuffer{};
    Buffer indexBuffer{};
    Buffer materialBuffer{};

    std::vector<MeshDraw> draws;
    uint32_t vertexCount   = 0;
    uint32_t indexCount    = 0;
    uint32_t materialCount = 0;
};
