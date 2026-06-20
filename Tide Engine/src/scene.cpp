#include "scene.h"
#include "vk_engine.h"

// Create a device-local buffer and fill it via a staging copy.
static Buffer makeDeviceBuffer(VulkanEngine& eng, const void* src,
                               VkDeviceSize size, VkBufferUsageFlags usage,
                               const char* name) {
    VmaAllocator alloc = eng.allocator();

    Buffer dst = createBuffer(alloc, size,
                              usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO);

    Buffer staging = createBuffer(alloc, size,
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                  VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(staging.mapped, src, size);

    eng.immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &copy);
    });

    destroyBuffer(alloc, staging);
    eng.setDebugName((uint64_t)dst.buffer, VK_OBJECT_TYPE_BUFFER, name);
    return dst;
}

void Scene::build(VulkanEngine& eng, const MeshData& data) {
    vertexCount   = (uint32_t)data.vertices.size();
    indexCount    = (uint32_t)data.indices.size();
    materialCount = (uint32_t)data.materials.size();
    draws         = data.draws;

    // Vertex/index buffers are RT-ready: device address + accel build input.
    const VkBufferUsageFlags rtFlags =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    if (vertexCount)
        vertexBuffer = makeDeviceBuffer(eng, data.vertices.data(),
                                        vertexCount * sizeof(Vertex),
                                        rtFlags | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                        "Scene Vertex Buffer");
    if (indexCount)
        indexBuffer = makeDeviceBuffer(eng, data.indices.data(),
                                       indexCount * sizeof(uint32_t),
                                       rtFlags | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                       "Scene Index Buffer");
    if (materialCount)
        materialBuffer = makeDeviceBuffer(eng, data.materials.data(),
                                          materialCount * sizeof(GpuMaterial),
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          "Scene Material Buffer");

    TE_INFO("Scene: draws=%u  vertices=%u  indices=%u  materials=%u  images=%u\n",
            (uint32_t)draws.size(), vertexCount, indexCount, materialCount, data.imageCount);
}

void Scene::destroy(VmaAllocator alloc) {
    destroyBuffer(alloc, vertexBuffer);
    destroyBuffer(alloc, indexBuffer);
    destroyBuffer(alloc, materialBuffer);
}
