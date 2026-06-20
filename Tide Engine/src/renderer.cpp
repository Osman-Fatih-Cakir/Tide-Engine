#include "renderer.h"
#include "vk_engine.h"
#include "scene.h"
#include "shader.h"
#include "mesh.h"
#include <cstddef> // offsetof

// ---------------------------------------------------------------------------
// Push constants (must match the GLSL layouts).
// ---------------------------------------------------------------------------
struct VisPush {
    glm::mat4 viewProj;
    glm::mat4 model;
    uint32_t  drawID;
};
struct ResolvePush {
    glm::mat4  viewProj;
    glm::vec4  cameraPos;  // xyz
    glm::vec4  sunDir;     // xyz = dir to sun, w = ambient
    glm::vec4  sunColor;   // rgb = radiance (intensity baked in)
    glm::uvec2 screenSize;
};
struct TonemapPush {
    float exposure;
};

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------
static Image makeImage(VulkanEngine& eng, VkFormat format, VkImageUsageFlags usage,
                       VkExtent2D extent, VkImageAspectFlags aspect, const char* name) {
    Image img{};
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {extent.width, extent.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VK_CHECK(vmaCreateImage(eng.allocator(), &ici, &ai, &img.image, &img.alloc, nullptr));

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(eng.device(), &vci, nullptr, &img.view));

    eng.setDebugName((uint64_t)img.image, VK_OBJECT_TYPE_IMAGE, name);
    return img;
}

static void imgBarrier(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspect,
                       VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                       VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                       VkImageLayout oldL, VkImageLayout newL) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldL;
    b.newLayout = newL;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Pipeline builders.
// ---------------------------------------------------------------------------
void Renderer::init(VulkanEngine& eng, VkFormat swapchainFormat, VkFormat depthFormat,
                    VkDescriptorSetLayout sceneSetLayout) {
    VkDevice device = eng.device();
    m_swapchainFormat = swapchainFormat;
    m_depthFormat = depthFormat;

    // HDR sampler (linear, clamp) used by the tonemap pass.
    {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device, &sci, nullptr, &m_hdrSampler));
    }

    // ---- Descriptor set layouts (resolve set1, tonemap set0) ----
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0; // vis storage image (read)
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; // hdr storage image (write)
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2;
        lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_resolveSetLayout));
    }
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; // hdr sampled
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1;
        lci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_tonemapSetLayout));
    }

    // ---- Descriptor pool + sets (views written in createTargets) ----
    {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[0].descriptorCount = 2;
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 2;
        pci.poolSizeCount = 2;
        pci.pPoolSizes = sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &m_pool));

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = m_pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m_resolveSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_resolveSet));
        dai.pSetLayouts = &m_tonemapSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_tonemapSet));
    }

    // ---- Visibility pipeline (raster, position-only vertex input) ----
    {
        VkShaderModule vert = loadShaderModule(device, "shaders/vis.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkShaderModule frag = loadShaderModule(device, "shaders/vis.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag; stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions = &attr;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT; // R32_UINT, single channel
        cba.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dynStates;

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = sizeof(VisPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_visLayout));

        VkFormat visFormat = VK_FORMAT_R32_UINT;
        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &visFormat;
        rendering.depthAttachmentFormat = depthFormat;

        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &rendering;
        gpi.stageCount = 2; gpi.pStages = stages;
        gpi.pVertexInputState = &vi;
        gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vp;
        gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms;
        gpi.pDepthStencilState = &ds;
        gpi.pColorBlendState = &cb;
        gpi.pDynamicState = &dyn;
        gpi.layout = m_visLayout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &m_visPipeline));
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }

    // ---- Resolve pipeline (compute) ----
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/resolve.comp", VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(ResolvePush);
        VkDescriptorSetLayout sets[2] = {sceneSetLayout, m_resolveSetLayout};
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 2;
        lci.pSetLayouts = sets;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_resolveLayout));

        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp;
        cpi.stage.pName = "main";
        cpi.layout = m_resolveLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_resolvePipeline));
        vkDestroyShaderModule(device, comp, nullptr);
    }

    // ---- Tonemap pipeline (fullscreen fragment, no vertex input/depth) ----
    {
        VkShaderModule vert = loadShaderModule(device, "shaders/fullscreen.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkShaderModule frag = loadShaderModule(device, "shaders/tonemap.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag; stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cba;
        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynStates;

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = sizeof(TonemapPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &m_tonemapSetLayout;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_tonemapLayout));

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &m_swapchainFormat;

        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &rendering;
        gpi.stageCount = 2; gpi.pStages = stages;
        gpi.pVertexInputState = &vi;
        gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vp;
        gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms;
        gpi.pColorBlendState = &cb;
        gpi.pDynamicState = &dyn;
        gpi.layout = m_tonemapLayout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &m_tonemapPipeline));
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }
}

void Renderer::createTargets(VulkanEngine& eng, VkExtent2D extent) {
    destroyTargets(eng);
    m_extent = extent;

    m_vis = makeImage(eng, VK_FORMAT_R32_UINT,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                      extent, VK_IMAGE_ASPECT_COLOR_BIT, "Vis Image");
    m_hdr = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                      extent, VK_IMAGE_ASPECT_COLOR_BIT, "HDR Image");

    // Point resolve set (vis read, hdr write) and tonemap set (hdr sampled) at
    // the fresh views.
    VkDescriptorImageInfo visInfo{};
    visInfo.imageView = m_vis.view;
    visInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo hdrStore{};
    hdrStore.imageView = m_hdr.view;
    hdrStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo hdrSamp{};
    hdrSamp.sampler = m_hdrSampler;
    hdrSamp.imageView = m_hdr.view;
    hdrSamp.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w[3]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = m_resolveSet; w[0].dstBinding = 0;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].descriptorCount = 1; w[0].pImageInfo = &visInfo;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = m_resolveSet; w[1].dstBinding = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[1].descriptorCount = 1; w[1].pImageInfo = &hdrStore;
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[2].dstSet = m_tonemapSet; w[2].dstBinding = 0;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[2].descriptorCount = 1; w[2].pImageInfo = &hdrSamp;
    vkUpdateDescriptorSets(eng.device(), 3, w, 0, nullptr);
}

void Renderer::destroyTargets(VulkanEngine& eng) {
    destroyImage(eng.allocator(), eng.device(), m_vis);
    destroyImage(eng.allocator(), eng.device(), m_hdr);
}

void Renderer::destroy(VulkanEngine& eng) {
    VkDevice device = eng.device();
    destroyTargets(eng);
    if (m_hdrSampler) vkDestroySampler(device, m_hdrSampler, nullptr);
    if (m_visPipeline) vkDestroyPipeline(device, m_visPipeline, nullptr);
    if (m_visLayout) vkDestroyPipelineLayout(device, m_visLayout, nullptr);
    if (m_resolvePipeline) vkDestroyPipeline(device, m_resolvePipeline, nullptr);
    if (m_resolveLayout) vkDestroyPipelineLayout(device, m_resolveLayout, nullptr);
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_tonemapLayout) vkDestroyPipelineLayout(device, m_tonemapLayout, nullptr);
    if (m_resolveSetLayout) vkDestroyDescriptorSetLayout(device, m_resolveSetLayout, nullptr);
    if (m_tonemapSetLayout) vkDestroyDescriptorSetLayout(device, m_tonemapSetLayout, nullptr);
    if (m_pool) vkDestroyDescriptorPool(device, m_pool, nullptr);
}

// ---------------------------------------------------------------------------
// Per-frame recording.
// ---------------------------------------------------------------------------
void Renderer::record(VkCommandBuffer cmd, const Scene& scene,
                      const glm::mat4& viewProj, const glm::vec3& cameraPos,
                      const Settings& settings, VkExtent2D extent,
                      VkImage swapchainImage, VkImageView swapchainView,
                      VkImage depthImage, VkImageView depthView,
                      TracyVkCtx tracy) {
    VkViewport vp{};
    vp.width = (float)extent.width;
    vp.height = (float)extent.height;
    vp.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = extent;

    // ===================== Pass A: Visibility (raster) =====================
    {
        TracyVkZone(tracy, cmd, "Visibility");
        imgBarrier(cmd, m_vis.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        imgBarrier(cmd, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = m_vis.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color.uint32[0] = 0xFFFFFFFFu; // empty sentinel

        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = depthView;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = extent;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);

        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_visPipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene.vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, scene.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        for (uint32_t i = 0; i < (uint32_t)scene.draws.size(); i++) {
            const MeshDraw& d = scene.draws[i];
            VisPush push{};
            push.viewProj = viewProj;
            push.model = d.transform;
            push.drawID = i;
            vkCmdPushConstants(cmd, m_visLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(VisPush), &push);
            vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, d.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);
    }

    // ===================== Pass B: Resolve (compute -> HDR) =====================
    {
        TracyVkZone(tracy, cmd, "Resolve");
        imgBarrier(cmd, m_vis.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
        VkDescriptorSet sets[2] = {scene.descriptorSet, m_resolveSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveLayout,
                                0, 2, sets, 0, nullptr);

        glm::vec3 sun = sunDirection(settings);
        ResolvePush push{};
        push.viewProj = viewProj;
        push.cameraPos = glm::vec4(cameraPos, 1.0f);
        push.sunDir = glm::vec4(sun, settings.ambient);
        push.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), 0.0f);
        push.screenSize = glm::uvec2(extent.width, extent.height);
        vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ResolvePush), &push);

        vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    }

    // ===================== Pass C: Tonemap (HDR -> swapchain) =====================
    {
        TracyVkZone(tracy, cmd, "Tonemap");
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        imgBarrier(cmd, swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = swapchainView;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = extent;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &ri);

        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapLayout,
                                0, 1, &m_tonemapSet, 0, nullptr);
        TonemapPush tp{settings.exposure};
        vkCmdPushConstants(cmd, m_tonemapLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(TonemapPush), &tp);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }
}
