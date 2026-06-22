#include "renderer.h"
#include "vk_engine.h"
#include "scene.h"
#include "shader.h"
#include "mesh.h"
#include "dlss.h"
#include <cstddef>   // offsetof
#include <algorithm> // sort
#include <vector>

// ---------------------------------------------------------------------------
// Push constants (must match the GLSL layouts).
// shadowCfg = (x=sunAngularSizeRad, y=sampleCount, z=shadowsEnabled, w=frameIndex)
// ---------------------------------------------------------------------------
struct VisPush {
    glm::mat4 viewProj;
    glm::mat4 model;
    uint32_t  drawID;
};
struct ResolvePush {
    glm::mat4  viewProj;
    glm::mat4  prevViewProj; // for temporal reprojection of the shadow history
    glm::vec4  cameraPos;  // xyz
    glm::vec4  sunDir;     // xyz = dir to sun, w = ambient
    glm::vec4  sunColor;   // rgb = radiance
    glm::vec4  shadowCfg;  // ray-traced shadow config
    glm::vec4  temporal;   // x=reset y=histAlpha z=denoiseOn
    glm::vec4  jitter;     // xy = current jitter in NDC, z = debugMotion flag
    glm::uvec2 screenSize;
};
struct CompositePush {
    glm::uvec2 screenSize;
};
struct TonemapPush {
    float exposure;
};
struct TransparentPush {
    glm::mat4 viewProj;
    glm::mat4 model;
    glm::vec4 cameraPos;    // w = materialIndex (float)
    glm::vec4 sunDir;
    glm::vec4 sunColor;
    glm::vec4 shadowCfg;
};

// Halton low-discrepancy sequence (radical inverse), for sub-pixel TAA/DLSS jitter.
static float halton(uint32_t i, uint32_t base) {
    float f = 1.0f, r = 0.0f;
    while (i > 0) { f /= (float)base; r += f * (float)(i % base); i /= base; }
    return r;
}

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
    m_eng = &eng;
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
    // Shadow-history sampler (linear, clamp) for temporal reprojection lookups.
    {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device, &sci, nullptr, &m_histSampler));
    }

    // ---- Descriptor set layouts (resolve set1, tonemap set0) ----
    {
        VkDescriptorSetLayoutBinding b[10]{};
        b[0].binding = 0; // vis storage image (read)
        b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1;
        b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; // hdr storage image (write)
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1;
        b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[2].binding = 2; // shadow history (read, sampled)
        b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[2].descriptorCount = 1;
        b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[3].binding = 3; // shadow history (write, storage)
        b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[3].descriptorCount = 1;
        b[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[4].binding = 4; // motion vectors (write, storage)
        b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[4].descriptorCount = 1;
        b[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // b5..b7 RR guides (diffuse, specular, normal+rough); b8 directLight; b9 shadowOut.
        for (uint32_t i = 5; i <= 9; i++) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 10;
        lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_resolveSetLayout));
    }
    // Composite set: directLight(read) + shadow(read) + hdr(read/write), all storage.
    {
        VkDescriptorSetLayoutBinding b[3]{};
        for (uint32_t i = 0; i < 3; i++) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3;
        lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_compositeSetLayout));
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
        // 2 resolve sets * 9 storage (vis,hdr,histWrite,motion,diff,spec,normal,directLight,
        // shadowOut) + 3 composite storage = 21.
        sizes[0].descriptorCount = 21;
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = 4;  // 2 tonemap (hdr/dlss) + 2 resolve sets (histRead)
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 5;               // 2 resolve + 1 composite + 2 tonemap
        pci.poolSizeCount = 2;
        pci.pPoolSizes = sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &m_pool));

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = m_pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m_resolveSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_resolveSet[0]));
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_resolveSet[1]));
        dai.pSetLayouts = &m_compositeSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_compositeSet));
        dai.pSetLayouts = &m_tonemapSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_tonemapSet));
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_tonemapSetDlss));
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

    // ---- Transparent forward pipeline (blends into HDR, tests opaque depth) ----
    {
        VkShaderModule vert = loadShaderModule(device, "shaders/transparent.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkShaderModule frag = loadShaderModule(device, "shaders/transparent.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
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
        VkVertexInputAttributeDescription attrs[4]{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex, position)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(Vertex, normal)};
        attrs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex, uv)};
        attrs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 4;
        vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE; // glass is two-sided
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE; // transparency doesn't occlude itself
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
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
        pc.size = sizeof(TransparentPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &sceneSetLayout; // scene set carries the TLAS (b5)
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_transparentLayout));

        VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &hdrFormat;
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
        gpi.layout = m_transparentLayout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &m_transparentPipeline));
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

    // ---- Composite pipeline (compute): hdr = ambient + directLight * shadow ----
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/composite.comp", VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(CompositePush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &m_compositeSetLayout;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_compositeLayout));

        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp;
        cpi.stage.pName = "main";
        cpi.layout = m_compositeLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_compositePipeline));
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

void Renderer::createTargets(VulkanEngine& eng, VkExtent2D extent, VkExtent2D displayExtent) {
    destroyTargets(eng);
    m_extent = extent;
    m_displayExtent = displayExtent;
    m_dlssReset = true; // history is invalid after (re)creating resources

    m_vis = makeImage(eng, VK_FORMAT_R32_UINT,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                      extent, VK_IMAGE_ASPECT_COLOR_BIT, "Vis Image");
    m_hdr = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                      extent, VK_IMAGE_ASPECT_COLOR_BIT, "HDR Image");
    for (int i = 0; i < 2; i++)
        m_shadowHist[i] = makeImage(eng, VK_FORMAT_R16G16_SFLOAT,
                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                    extent, VK_IMAGE_ASPECT_COLOR_BIT,
                                    i == 0 ? "Shadow Hist 0" : "Shadow Hist 1");
    m_motion = makeImage(eng, VK_FORMAT_R16G16_SFLOAT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         extent, VK_IMAGE_ASPECT_COLOR_BIT, "Motion Vectors");
    // Ray Reconstruction guide buffers (render-res). NGX reads them (sampled).
    m_gbufDiffuse = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              extent, VK_IMAGE_ASPECT_COLOR_BIT, "GBuffer Diffuse");
    m_gbufSpecular = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               extent, VK_IMAGE_ASPECT_COLOR_BIT, "GBuffer Specular");
    m_gbufNormal = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             extent, VK_IMAGE_ASPECT_COLOR_BIT, "GBuffer Normal+Rough");
    // Deferred shadow split: unshadowed direct radiance + shadow visibility (à-trous ping-pong).
    m_directLight = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT,
                              extent, VK_IMAGE_ASPECT_COLOR_BIT, "Direct Light");
    m_shadowOut = makeImage(eng, VK_FORMAT_R16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            extent, VK_IMAGE_ASPECT_COLOR_BIT, "Shadow Out A");
    m_shadowOut2 = makeImage(eng, VK_FORMAT_R16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             extent, VK_IMAGE_ASPECT_COLOR_BIT, "Shadow Out B");
    // DLSS upscale target (display resolution). STORAGE so NGX can write it;
    // TRANSFER_DST because NGX clears it internally.
    m_dlssOutput = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                             displayExtent, VK_IMAGE_ASPECT_COLOR_BIT, "DLSS Output");

    // Initialize both history images to GENERAL so the per-frame "read = GENERAL ->
    // SHADER_READ" barrier is always valid (first frame forces a history reset).
    eng.immediateSubmit([&](VkCommandBuffer cmd) {
        for (int i = 0; i < 2; i++)
            imgBarrier(cmd, m_shadowHist[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    });
    m_haveHistory = false;
    m_histIndex = 0;

    // Resolve sets ping-pong: set[k] writes hist[k] (b3) and reads hist[1-k] (b2).
    VkDescriptorImageInfo visInfo{};
    visInfo.imageView = m_vis.view;
    visInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo hdrStore{};
    hdrStore.imageView = m_hdr.view;
    hdrStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo histSamp[2]{}, histStore[2]{};
    for (int i = 0; i < 2; i++) {
        histSamp[i].sampler = m_histSampler;
        histSamp[i].imageView = m_shadowHist[i].view;
        histSamp[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        histStore[i].imageView = m_shadowHist[i].view;
        histStore[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorImageInfo motionStore{};
    motionStore.imageView = m_motion.view;
    motionStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo diffStore{}, specStore{}, normStore{};
    diffStore.imageView = m_gbufDiffuse.view;  diffStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    specStore.imageView = m_gbufSpecular.view; specStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    normStore.imageView = m_gbufNormal.view;   normStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo directStore{}, shadowOutStore{};
    directStore.imageView    = m_directLight.view; directStore.imageLayout    = VK_IMAGE_LAYOUT_GENERAL;
    shadowOutStore.imageView = m_shadowOut.view;   shadowOutStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo hdrSamp{};
    hdrSamp.sampler = m_hdrSampler;
    hdrSamp.imageView = m_hdr.view;
    hdrSamp.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo dlssSamp{};
    dlssSamp.sampler = m_hdrSampler;
    dlssSamp.imageView = m_dlssOutput.view;
    dlssSamp.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::vector<VkWriteDescriptorSet> w;
    auto add = [&](VkDescriptorSet set, uint32_t bind, VkDescriptorType t,
                   const VkDescriptorImageInfo* info) {
        VkWriteDescriptorSet x{};
        x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x.dstSet = set; x.dstBinding = bind; x.descriptorType = t;
        x.descriptorCount = 1; x.pImageInfo = info;
        w.push_back(x);
    };
    for (int k = 0; k < 2; k++) {
        add(m_resolveSet[k], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &visInfo);
        add(m_resolveSet[k], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrStore);
        add(m_resolveSet[k], 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &histSamp[1 - k]);
        add(m_resolveSet[k], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &histStore[k]);
        add(m_resolveSet[k], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &motionStore);
        add(m_resolveSet[k], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffStore);
        add(m_resolveSet[k], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specStore);
        add(m_resolveSet[k], 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normStore);
        add(m_resolveSet[k], 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);
        add(m_resolveSet[k], 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOutStore);
    }
    // Composite: directLight(read) + shadowOut(read) + hdr(read/write).
    add(m_compositeSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);
    add(m_compositeSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOutStore);
    add(m_compositeSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrStore);
    add(m_tonemapSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdrSamp);
    add(m_tonemapSetDlss, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dlssSamp);
    vkUpdateDescriptorSets(eng.device(), (uint32_t)w.size(), w.data(), 0, nullptr);
}

void Renderer::destroyTargets(VulkanEngine& eng) {
    destroyImage(eng.allocator(), eng.device(), m_vis);
    destroyImage(eng.allocator(), eng.device(), m_hdr);
    destroyImage(eng.allocator(), eng.device(), m_shadowHist[0]);
    destroyImage(eng.allocator(), eng.device(), m_shadowHist[1]);
    destroyImage(eng.allocator(), eng.device(), m_motion);
    destroyImage(eng.allocator(), eng.device(), m_gbufDiffuse);
    destroyImage(eng.allocator(), eng.device(), m_gbufSpecular);
    destroyImage(eng.allocator(), eng.device(), m_gbufNormal);
    destroyImage(eng.allocator(), eng.device(), m_directLight);
    destroyImage(eng.allocator(), eng.device(), m_shadowOut);
    destroyImage(eng.allocator(), eng.device(), m_shadowOut2);
    destroyImage(eng.allocator(), eng.device(), m_dlssOutput);
}

void Renderer::destroy(VulkanEngine& eng) {
    VkDevice device = eng.device();
    destroyTargets(eng);
    if (m_hdrSampler) vkDestroySampler(device, m_hdrSampler, nullptr);
    if (m_histSampler) vkDestroySampler(device, m_histSampler, nullptr);
    if (m_visPipeline) vkDestroyPipeline(device, m_visPipeline, nullptr);
    if (m_visLayout) vkDestroyPipelineLayout(device, m_visLayout, nullptr);
    if (m_transparentPipeline) vkDestroyPipeline(device, m_transparentPipeline, nullptr);
    if (m_transparentLayout) vkDestroyPipelineLayout(device, m_transparentLayout, nullptr);
    if (m_resolvePipeline) vkDestroyPipeline(device, m_resolvePipeline, nullptr);
    if (m_resolveLayout) vkDestroyPipelineLayout(device, m_resolveLayout, nullptr);
    if (m_compositePipeline) vkDestroyPipeline(device, m_compositePipeline, nullptr);
    if (m_compositeLayout) vkDestroyPipelineLayout(device, m_compositeLayout, nullptr);
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_tonemapLayout) vkDestroyPipelineLayout(device, m_tonemapLayout, nullptr);
    if (m_resolveSetLayout) vkDestroyDescriptorSetLayout(device, m_resolveSetLayout, nullptr);
    if (m_compositeSetLayout) vkDestroyDescriptorSetLayout(device, m_compositeSetLayout, nullptr);
    if (m_tonemapSetLayout) vkDestroyDescriptorSetLayout(device, m_tonemapSetLayout, nullptr);
    if (m_pool) vkDestroyDescriptorPool(device, m_pool, nullptr);
}

// ---------------------------------------------------------------------------
// Per-frame recording.
// ---------------------------------------------------------------------------
void Renderer::record(VkCommandBuffer cmd, const Scene& scene,
                      const glm::mat4& viewProj, const glm::mat4& view, const glm::mat4& proj,
                      const glm::vec3& cameraPos,
                      const Settings& settings,
                      VkExtent2D renderExtent, VkExtent2D displayExtent,
                      VkImage swapchainImage, VkImageView swapchainView,
                      VkImage depthImage, VkImageView depthView,
                      Dlss* dlss, bool dlssActive,
                      TracyVkCtx tracy) {
    VkExtent2D extent = renderExtent; // passes A-C run at render resolution
    VkViewport vp{};
    vp.width = (float)extent.width;
    vp.height = (float)extent.height;
    vp.maxDepth = 1.0f;
    VkRect2D scissor{};
    scissor.extent = extent;

    uint32_t frame = m_frameIndex++;

    // Sub-pixel TAA/DLSS jitter: shift the projection by a Halton offset in [-0.5,0.5]
    // pixels, expressed in NDC (clip range is 2). The raster + resolve reconstruction
    // use the jittered matrix; motion vectors are computed unjittered (see resolve.comp,
    // which removes jitter.xy from the current UV). prevViewProj stays unjittered.
    glm::vec2 jitterNDC(0.0f);
    glm::vec2 jitterPixels(0.0f); // render-pixel space, for DLSS
    if (settings.taaJitter || dlssActive) {
        float jx = halton((frame & 1023u) + 1u, 2u) - 0.5f;
        float jy = halton((frame & 1023u) + 1u, 3u) - 0.5f;
        jitterPixels = glm::vec2(jx, jy);
        jitterNDC = glm::vec2(2.0f * jx / (float)extent.width,
                              2.0f * jy / (float)extent.height);
    }
    // Add jx*clip.w to clip.x (and jy to .y) so the shift is constant in NDC/pixels:
    // row0 += jx * row3, row1 += jy * row3 (glm column-major: element [col][row]).
    glm::mat4 jitteredVP = viewProj;
    for (int c = 0; c < 4; c++) {
        jitteredVP[c][0] += jitterNDC.x * jitteredVP[c][3];
        jitteredVP[c][1] += jitterNDC.y * jitteredVP[c][3];
    }

    // Ray-traced shadow config (shared by resolve + transparent passes).
    glm::vec3 sun = sunDirection(settings);
    glm::vec4 shadowCfg(
        glm::radians(settings.sunAngularSize),               // sun cone half-angle (rad)
        settings.shadowsEnabled ? (float)settings.shadowSamples : 0.0f,
        settings.shadowsEnabled ? 1.0f : 0.0f,
        (float)(frame & 0xFFFF));                             // frame index for dither

    // Temporal shadow denoise state. Reset history when it's the first frame, the
    // sun moved (shadows changed), or denoise is off. When DLSS Ray Reconstruction
    // is active it owns denoising — feed it the RAW noisy shadow (our accumulation off).
    bool denoise = settings.shadowDenoise && !dlssActive;
    bool sunMoved = settings.sunAzimuthDeg != m_prevSunAz ||
                    settings.sunElevationDeg != m_prevSunEl;
    bool reset = !m_haveHistory || sunMoved || !denoise;
    glm::vec4 temporal(reset ? 1.0f : 0.0f,
                       settings.shadowHistAlpha,
                       denoise ? 1.0f : 0.0f, 0.0f);
    uint32_t cur = m_histIndex;

    // ===================== Pass A: Visibility (raster) =====================
    {
        TracyVkZone(tracy, cmd, "Visibility");
        m_eng->cmdBeginLabel(cmd, "Visibility Pass");
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
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // transparent pass tests against it
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

        for (uint32_t instID : scene.opaqueInstances) {
            const MeshInstance& inst = scene.instances[instID];
            const Geometry& g = scene.geometries[inst.geometryID];
            VisPush push{};
            push.viewProj = jitteredVP;
            push.model = inst.transform;
            push.drawID = instID; // packed into the visibility ID
            vkCmdPushConstants(cmd, m_visLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(VisPush), &push);
            vkCmdDrawIndexed(cmd, g.indexCount, 1, g.firstIndex, g.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass B: Resolve (compute -> HDR) =====================
    {
        TracyVkZone(tracy, cmd, "Resolve");
        m_eng->cmdBeginLabel(cmd, "Resolve Pass (PBR+RT shadow)");
        imgBarrier(cmd, m_vis.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // Shadow history: read prev (hist[1-cur]) GENERAL->SHADER_READ; write hist[cur]->GENERAL.
        imgBarrier(cmd, m_shadowHist[1 - cur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        imgBarrier(cmd, m_shadowHist[cur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // Motion vectors + RR guide buffers: fully overwritten each frame.
        imgBarrier(cmd, m_motion.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        for (VkImage g : {m_gbufDiffuse.image, m_gbufSpecular.image, m_gbufNormal.image,
                          m_directLight.image, m_shadowOut.image})
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
        VkDescriptorSet sets[2] = {scene.descriptorSet, m_resolveSet[cur]};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveLayout,
                                0, 2, sets, 0, nullptr);

        ResolvePush push{};
        push.viewProj = jitteredVP;          // reconstruction must match the jittered raster
        push.prevViewProj = m_prevViewProj;  // unjittered previous frame
        push.cameraPos = glm::vec4(cameraPos, 1.0f);
        push.sunDir = glm::vec4(sun, settings.ambient);
        push.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), 0.0f);
        push.shadowCfg = shadowCfg;
        push.temporal = temporal;
        push.jitter = glm::vec4(jitterNDC, settings.debugMotionVecs ? 1.0f : 0.0f, 0.0f);
        push.screenSize = glm::uvec2(extent.width, extent.height);
        vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ResolvePush), &push);

        vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        m_eng->cmdEndLabel(cmd);
    }

    // Advance temporal state for next frame.
    m_prevViewProj = viewProj;
    m_prevSunAz = settings.sunAzimuthDeg;
    m_prevSunEl = settings.sunElevationDeg;
    m_haveHistory = true;
    m_histIndex = 1 - cur;

    // ===================== Pass B2: Composite (deferred shadow recombine) =====================
    // hdr (ambient) += directLight * shadow. Skipped in motion-vector debug (resolve already
    // wrote the debug visualization straight into hdr). (À-trous filtering lands in D2.)
    if (!settings.debugMotionVecs) {
        TracyVkZone(tracy, cmd, "Composite");
        m_eng->cmdBeginLabel(cmd, "Composite Pass (shadow recombine)");
        // directLight + shadowOut: resolve wrote them (GENERAL) -> composite reads them.
        for (VkImage g : {m_directLight.image, m_shadowOut.image})
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        // hdr: resolve wrote ambient -> composite reads + writes final (in place).
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compositeLayout,
                                0, 1, &m_compositeSet, 0, nullptr);
        CompositePush cp{glm::uvec2(extent.width, extent.height)};
        vkCmdPushConstants(cmd, m_compositeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(CompositePush), &cp);
        vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass C: Transparent forward (blend -> HDR) =====================
    // After this, HDR is in GENERAL (no transparency) or COLOR_ATTACHMENT (with).
    bool hasTransparent = !scene.transparentInstances.empty();
    if (hasTransparent) {
        TracyVkZone(tracy, cmd, "Transparent");
        m_eng->cmdBeginLabel(cmd, "Transparent Pass");
        // HDR: compute write (GENERAL) -> color attachment blend.
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        // Depth: vis wrote it -> transparent tests it (read-only).
        imgBarrier(cmd, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = m_hdr.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // keep the opaque resolve
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = depthView;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_transparentPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_transparentLayout,
                                0, 1, &scene.descriptorSet, 0, nullptr);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene.vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, scene.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Back-to-front sort (centroid ~ transform origin), so blending is correct.
        std::vector<uint32_t> sorted(scene.transparentInstances.begin(),
                                     scene.transparentInstances.end());
        std::sort(sorted.begin(), sorted.end(), [&](uint32_t a, uint32_t b) {
            glm::vec3 ca = glm::vec3(scene.instances[a].transform[3]);
            glm::vec3 cb = glm::vec3(scene.instances[b].transform[3]);
            return glm::dot(ca - cameraPos, ca - cameraPos) >
                   glm::dot(cb - cameraPos, cb - cameraPos);
        });

        for (uint32_t instID : sorted) {
            const MeshInstance& inst = scene.instances[instID];
            const Geometry& g = scene.geometries[inst.geometryID];
            TransparentPush push{};
            push.viewProj = jitteredVP; // match the jittered opaque depth
            push.model = inst.transform;
            push.cameraPos = glm::vec4(cameraPos, (float)g.materialIndex);
            push.sunDir = glm::vec4(sun, settings.ambient);
            push.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), 0.0f);
            push.shadowCfg = shadowCfg;
            vkCmdPushConstants(cmd, m_transparentLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(TransparentPush), &push);
            vkCmdDrawIndexed(cmd, g.indexCount, 1, g.firstIndex, g.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
    }

    // HDR source state after the (optional) transparent pass.
    VkPipelineStageFlags2 hdrStage = hasTransparent
        ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    VkAccessFlags2 hdrAccess = hasTransparent
        ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkImageLayout hdrOld = hasTransparent
        ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;

    // ===================== Pass D0: DLSS upscale (render-res HDR -> display-res) =====================
    // The tonemap then reads m_dlssOutput. NGX expects all resources in GENERAL.
    if (dlssActive && dlss) {
        TracyVkZone(tracy, cmd, "DLSS");
        m_eng->cmdBeginLabel(cmd, "DLSS Upscale");
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   hdrStage, hdrAccess,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                   hdrOld, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, m_motion.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        // RR guide buffers: written by resolve (GENERAL) -> read by NGX.
        for (VkImage g : {m_gbufDiffuse.image, m_gbufSpecular.image, m_gbufNormal.image})
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, m_dlssOutput.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        dlss->evaluate(cmd, m_hdr.image, m_hdr.view, depthImage, depthView,
                       m_motion.image, m_motion.view,
                       m_gbufDiffuse.image, m_gbufDiffuse.view,
                       m_gbufSpecular.image, m_gbufSpecular.view,
                       m_gbufNormal.image, m_gbufNormal.view,
                       m_dlssOutput.image, m_dlssOutput.view,
                       renderExtent, jitterPixels, view, proj, m_dlssReset);
        m_dlssReset = false;
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass D: Tonemap (HDR -> swapchain, display res) =====================
    {
        TracyVkZone(tracy, cmd, "Tonemap");
        m_eng->cmdBeginLabel(cmd, "Tonemap Pass");

        VkDescriptorSet tonemapSet = m_tonemapSet;
        if (dlssActive && dlss) {
            // DLSS output (NGX wrote it, GENERAL) -> sampled by tonemap.
            imgBarrier(cmd, m_dlssOutput.image, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            tonemapSet = m_tonemapSetDlss;
        } else {
            imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                       hdrStage, hdrAccess,
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                       hdrOld, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
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
        ri.renderArea.extent = displayExtent;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport vpD{};
        vpD.width = (float)displayExtent.width;
        vpD.height = (float)displayExtent.height;
        vpD.maxDepth = 1.0f;
        VkRect2D scissorD{};
        scissorD.extent = displayExtent;
        vkCmdSetViewport(cmd, 0, 1, &vpD);
        vkCmdSetScissor(cmd, 0, 1, &scissorD);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapLayout,
                                0, 1, &tonemapSet, 0, nullptr);
        TonemapPush tp{settings.exposure};
        vkCmdPushConstants(cmd, m_tonemapLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(TonemapPush), &tp);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
    }
}
