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
    Buffer instanceBuffer{}; // GpuInstance[] (b4) — transform + geometryID
    Buffer geometryBuffer{}; // GpuGeometry[] (b6) — firstIndex/vertexOffset/material

    SceneAccel accel{};      // BLAS-per-geometry + TLAS (b5); ray-query shadows + RTGI

    // CPU-side geometry + instance tables (used by the vis pass draws and accel).
    std::vector<Geometry>     geometries;
    std::vector<MeshInstance> instances;
    // Indices into `instances`, split by the geometry's material alpha mode.
    // Same instanceID space the V-buffer packs into the visibility ID.
    std::vector<uint32_t> opaqueInstances;
    std::vector<uint32_t> transparentInstances;
    // World-space bounds (copied from MeshData; used for camera framing).
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);

    uint32_t vertexCount   = 0;
    uint32_t indexCount    = 0;
    uint32_t materialCount = 0;
    uint32_t geometryCount = 0;
    uint32_t instanceCount = 0;

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
