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

    buildTexturesAndDescriptors(eng, data);

    TE_INFO("Scene: draws=%u  vertices=%u  indices=%u  materials=%u  textures=%u\n",
            (uint32_t)draws.size(), vertexCount, indexCount, materialCount, textureCount);
}

void Scene::buildTexturesAndDescriptors(VulkanEngine& eng, const MeshData& data) {
    VkDevice device = eng.device();

    // Upload all textures.
    textureCount = (uint32_t)data.textures.size();
    textures.reserve(textureCount);
    for (const auto& td : data.textures) {
        if (td.width <= 0 || td.height <= 0) { textures.push_back(Image{}); continue; }
        textures.push_back(createTextureImage(eng, td.rgba.data(), td.width, td.height));
    }

    // One sampler for everything.
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(device, &sci, nullptr, &sampler));

    const uint32_t arrayCount = textureCount > 0 ? textureCount : 1;

    // Descriptor set layout: binding0 = material SSBO, binding1 = bindless texture array.
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = arrayCount;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorBindingFlags flags[2] = {
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = 2;
    flagsInfo.pBindingFlags = flags;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.pNext = &flagsInfo;
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    lci.bindingCount = 2;
    lci.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &setLayout));

    // Pool.
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = arrayCount;

    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = descriptorPool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &setLayout;
    VK_CHECK(vkAllocateDescriptorSets(device, &dai, &descriptorSet));

    std::vector<VkWriteDescriptorSet> writes;

    VkDescriptorBufferInfo matInfo{};
    if (materialBuffer.buffer) {
        matInfo.buffer = materialBuffer.buffer;
        matInfo.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptorSet;
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = &matInfo;
        writes.push_back(w);
    }

    // Write the whole array so material image indices stay aligned. Empty slots
    // (failed decodes) get a fallback view to keep them valid.
    VkImageView fallback = VK_NULL_HANDLE;
    for (auto& t : textures) if (t.view) { fallback = t.view; break; }

    std::vector<VkDescriptorImageInfo> imageInfos;
    if (fallback) {
        imageInfos.resize(textureCount);
        for (uint32_t i = 0; i < textureCount; i++) {
            imageInfos[i].sampler = sampler;
            imageInfos[i].imageView = textures[i].view ? textures[i].view : fallback;
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptorSet;
        w.dstBinding = 1;
        w.dstArrayElement = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = textureCount;
        w.pImageInfo = imageInfos.data();
        writes.push_back(w);
    }

    if (!writes.empty())
        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void Scene::destroy(VkDevice device, VmaAllocator alloc) {
    for (auto& img : textures) destroyImage(alloc, device, img);
    if (sampler) vkDestroySampler(device, sampler, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    destroyBuffer(alloc, vertexBuffer);
    destroyBuffer(alloc, indexBuffer);
    destroyBuffer(alloc, materialBuffer);
}
