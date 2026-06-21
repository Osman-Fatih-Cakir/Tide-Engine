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

    // Flatten draws into the GPU draw SSBO (resolve compute reconstructs triangles).
    drawCount = (uint32_t)draws.size();
    if (drawCount) {
        std::vector<GpuDraw> gpuDraws(drawCount);
        for (uint32_t i = 0; i < drawCount; i++) {
            gpuDraws[i].transform     = draws[i].transform;
            gpuDraws[i].firstIndex    = draws[i].firstIndex;
            gpuDraws[i].vertexOffset  = (uint32_t)draws[i].vertexOffset;
            gpuDraws[i].materialIndex = draws[i].materialIndex;
        }
        drawBuffer = makeDeviceBuffer(eng, gpuDraws.data(),
                                      drawCount * sizeof(GpuDraw),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      "Scene Draw Buffer");
    }

    // Classify draws by material alpha mode: BLEND -> transparent, else opaque.
    for (uint32_t i = 0; i < (uint32_t)draws.size(); i++) {
        uint32_t mi = draws[i].materialIndex;
        bool blend = mi < data.materials.size() &&
                     data.materials[mi].alphaMode == ALPHA_BLEND;
        (blend ? transparentIndices : opaqueIndices).push_back(i);
    }

    boundsMin = data.boundsMin;
    boundsMax = data.boundsMax;

    buildTexturesAndDescriptors(eng, data);

    TE_INFO("Scene: draws=%u (opaque=%u transparent=%u)  vertices=%u  indices=%u  materials=%u  textures=%u\n",
            (uint32_t)draws.size(), (uint32_t)opaqueIndices.size(),
            (uint32_t)transparentIndices.size(),
            vertexCount, indexCount, materialCount, textureCount);
}

void Scene::buildTexturesAndDescriptors(VulkanEngine& eng, const MeshData& data) {
    VkDevice device = eng.device();

    // Upload all textures. Color data (baseColor) is sRGB; everything else
    // (normal / ORM / metalRough) is linear. Decide per image from material usage.
    textureCount = (uint32_t)data.textures.size();
    std::vector<bool> srgb(textureCount, false);
    for (const auto& m : data.materials)
        if (m.baseColorTexture >= 0 && m.baseColorTexture < (int)textureCount)
            srgb[m.baseColorTexture] = true;

    textures.reserve(textureCount);
    for (uint32_t i = 0; i < textureCount; i++) {
        const auto& td = data.textures[i];
        if (td.width <= 0 || td.height <= 0) { textures.push_back(Image{}); continue; }
        Image img = createTextureImage(eng, td.rgba.data(), td.width, td.height, srgb[i]);
        char name[48];
        snprintf(name, sizeof(name), "Texture %u (%s)", i, srgb[i] ? "sRGB" : "linear");
        eng.setDebugName((uint64_t)img.image, VK_OBJECT_TYPE_IMAGE, name);
        textures.push_back(img);
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

    // Scene descriptor set (read by the resolve compute):
    //   b0 material SSBO, b1 bindless texture array,
    //   b2 vertex SSBO, b3 index SSBO, b4 draw SSBO.
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (uint32_t i = 0; i < 5; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = arrayCount;
    // Materials (b0) + textures (b1) are also read by the transparent forward frag.
    bindings[0].stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorBindingFlags flags[5] = {
        0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        0, 0, 0,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flagsInfo.bindingCount = 5;
    flagsInfo.pBindingFlags = flags;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.pNext = &flagsInfo;
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    lci.bindingCount = 5;
    lci.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &setLayout));

    // Pool.
    VkDescriptorPoolSize sizes[2]{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 4; // material + vertex + index + draw
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

    // b0 material, b2 vertex, b3 index, b4 draw — all storage buffers.
    struct { uint32_t binding; VkBuffer buffer; } ssbos[] = {
        {0, materialBuffer.buffer},
        {2, vertexBuffer.buffer},
        {3, indexBuffer.buffer},
        {4, drawBuffer.buffer},
    };
    VkDescriptorBufferInfo bufInfos[4]{};
    for (uint32_t i = 0; i < 4; i++) {
        if (!ssbos[i].buffer) continue;
        bufInfos[i].buffer = ssbos[i].buffer;
        bufInfos[i].range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descriptorSet;
        w.dstBinding = ssbos[i].binding;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo = &bufInfos[i];
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
    destroyBuffer(alloc, drawBuffer);
}
