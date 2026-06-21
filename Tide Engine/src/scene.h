#pragma once
#include "mesh.h"
#include "gpu_buffer.h"
#include "gpu_image.h"
#include "accel.h"

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

    SceneAccel accel{};    // BLAS-per-draw + TLAS (b5); ray-query shadows + RTGI

    std::vector<MeshDraw> draws;
    // Indices into `draws`, split by material alpha mode. Same drawID space as
    // the GpuDraw SSBO (so the V-buffer can push the global index).
    std::vector<uint32_t> opaqueIndices;
    std::vector<uint32_t> transparentIndices;
    // World-space bounds (copied from MeshData; used for the shadow ortho frustum).
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);

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
