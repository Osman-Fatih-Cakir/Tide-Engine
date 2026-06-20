#pragma once
#include "mesh.h"
#include "gpu_buffer.h"
#include "gpu_image.h"

class VulkanEngine;

// GPU-resident scene: one big vertex buffer, one big index buffer, one material
// SSBO. Buffers are RT-ready (device address + accel build input). Bindless
// textures + drawing come in later steps.
class Scene {
public:
    void build(VulkanEngine& eng, const MeshData& data);
    void destroy(VkDevice device, VmaAllocator alloc);

    Buffer vertexBuffer{};
    Buffer indexBuffer{};
    Buffer materialBuffer{};
    Buffer drawBuffer{};   // GpuDraw[] — read by the resolve compute

    std::vector<MeshDraw> draws;
    uint32_t vertexCount   = 0;
    uint32_t indexCount    = 0;
    uint32_t materialCount = 0;
    uint32_t drawCount     = 0;

    // --- bindless textures + material SSBO ---
    std::vector<Image>    textures;
    VkSampler             sampler        = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSet  = VK_NULL_HANDLE;
    uint32_t              textureCount   = 0;

private:
    void buildTexturesAndDescriptors(VulkanEngine& eng, const MeshData& data);
};
