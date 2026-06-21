#include "renderer.h"
#include "vk_engine.h"
#include "scene.h"
#include "shader.h"
#include "mesh.h"
#include <cstddef>   // offsetof
#include <algorithm> // sort
#include <vector>

// ---------------------------------------------------------------------------
// Shadow map resolution. Pick from {1024, 2048, 4096}; everything derives from
// this single constant (target size + 1/dim in the PCSS params).
// ---------------------------------------------------------------------------
static constexpr uint32_t kShadowDim = 2048;

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
    glm::vec4  sunColor;   // rgb = radiance, w = shadows enabled (0/1)
    glm::uvec2 screenSize;
    glm::uvec2 _pad;       // std430 aligns the following mat4 to a 16-byte offset
    glm::mat4  lightViewProj;
    glm::vec4  shadowParams; // x=lightSizeUV y=normalBias z=depthBias w=invShadowDim
};
struct TonemapPush {
    float exposure;
};
// Kept at exactly 256 bytes (RTX 3070 maxPushConstantsSize) by packing the
// material index into cameraPos.w instead of a separate uint + padding.
struct TransparentPush {
    glm::mat4 viewProj;
    glm::mat4 model;
    glm::vec4 cameraPos;    // w = materialIndex (float)
    glm::vec4 sunDir;
    glm::vec4 sunColor;
    glm::mat4 lightViewProj;
    glm::vec4 shadowParams;
};
struct ShadowPush {
    glm::mat4 lightViewProj;
    glm::mat4 model;
    uint32_t  materialIndex;
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

    // Shadow-map sampler: linear, clamp to a white (= far, lit) border so samples
    // outside the light frustum read as unoccluded. PCSS compares depth manually.
    {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK(vkCreateSampler(device, &sci, nullptr, &m_shadowSampler));
    }

    // Shadow map image (fixed size, independent of the swapchain).
    m_shadowMap = makeImage(eng, m_depthFormat,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            {kShadowDim, kShadowDim}, VK_IMAGE_ASPECT_DEPTH_BIT, "Shadow Map");

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
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; // shadow map sampled (read by resolve compute + transparent frag)
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1;
        lci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_shadowSetLayout));
    }

    // ---- Descriptor pool + sets (views written in createTargets) ----
    {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[0].descriptorCount = 2;
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = 2; // tonemap (hdr) + shadow set (shadow map)
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 3;
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
        dai.pSetLayouts = &m_shadowSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_shadowSet));

        // Shadow set points at the (fixed-size) shadow map; written once here.
        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.sampler = m_shadowSampler;
        shadowInfo.imageView = m_shadowMap.view;
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet sw{};
        sw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw.dstSet = m_shadowSet; sw.dstBinding = 0;
        sw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw.descriptorCount = 1; sw.pImageInfo = &shadowInfo;
        vkUpdateDescriptorSets(device, 1, &sw, 0, nullptr);
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

    // ---- Shadow pipeline (depth-only, light's POV; alpha-mask discard) ----
    {
        VkShaderModule vert = loadShaderModule(device, "shaders/shadow.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkShaderModule frag = loadShaderModule(device, "shaders/shadow.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
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
        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 2;
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
        rs.cullMode = VK_CULL_MODE_NONE; // two-sided (thin blinds); bias fights acne
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        rs.depthBiasEnable = VK_TRUE; // slope-scaled, set dynamically

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendStateCreateInfo cb{}; // no color attachments
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_DEPTH_BIAS};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 3;
        dyn.pDynamicStates = dynStates;

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = sizeof(ShadowPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &sceneSetLayout; // bindless materials/textures for alpha mask
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_shadowLayout));

        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 0;
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
        gpi.layout = m_shadowLayout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &m_shadowPipeline));
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
        VkDescriptorSetLayout tsets[2] = {sceneSetLayout, m_shadowSetLayout};
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 2;
        lci.pSetLayouts = tsets;
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
        VkDescriptorSetLayout sets[3] = {sceneSetLayout, m_resolveSetLayout, m_shadowSetLayout};
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 3;
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
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
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
    destroyImage(eng.allocator(), device, m_shadowMap);
    if (m_shadowSampler) vkDestroySampler(device, m_shadowSampler, nullptr);
    if (m_shadowPipeline) vkDestroyPipeline(device, m_shadowPipeline, nullptr);
    if (m_shadowLayout) vkDestroyPipelineLayout(device, m_shadowLayout, nullptr);
    if (m_shadowSetLayout) vkDestroyDescriptorSetLayout(device, m_shadowSetLayout, nullptr);
    if (m_hdrSampler) vkDestroySampler(device, m_hdrSampler, nullptr);
    if (m_visPipeline) vkDestroyPipeline(device, m_visPipeline, nullptr);
    if (m_visLayout) vkDestroyPipelineLayout(device, m_visLayout, nullptr);
    if (m_transparentPipeline) vkDestroyPipeline(device, m_transparentPipeline, nullptr);
    if (m_transparentLayout) vkDestroyPipelineLayout(device, m_transparentLayout, nullptr);
    if (m_resolvePipeline) vkDestroyPipeline(device, m_resolvePipeline, nullptr);
    if (m_resolveLayout) vkDestroyPipelineLayout(device, m_resolveLayout, nullptr);
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_tonemapLayout) vkDestroyPipelineLayout(device, m_tonemapLayout, nullptr);
    if (m_resolveSetLayout) vkDestroyDescriptorSetLayout(device, m_resolveSetLayout, nullptr);
    if (m_tonemapSetLayout) vkDestroyDescriptorSetLayout(device, m_tonemapSetLayout, nullptr);
    if (m_pool) vkDestroyDescriptorPool(device, m_pool, nullptr);
}

// Directional light ortho frustum tight around the scene bounds. No Y flip:
// the same matrix rasterizes and samples the map, so the convention is internal
// (uv = ndc.xy*0.5+0.5). GLM_FORCE_DEPTH_ZERO_TO_ONE gives Vulkan's [0,1] z.
static glm::mat4 computeLightViewProj(const glm::vec3& sunDir,
                                      const glm::vec3& center, float radius) {
    glm::vec3 L = glm::normalize(sunDir);
    glm::vec3 eye = center + L * radius;
    glm::vec3 up = (std::fabs(L.y) > 0.99f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    glm::mat4 view = glm::lookAt(eye, center, up);
    glm::mat4 proj = glm::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * radius);
    return proj * view;
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

    // Light matrix + PCSS params (shared by shadow/resolve/transparent passes).
    glm::vec3 sceneCenter = 0.5f * (scene.boundsMin + scene.boundsMax);
    float sceneRadius = glm::length(scene.boundsMax - scene.boundsMin) * 0.5f;
    if (sceneRadius <= 0.0f) sceneRadius = 1.0f;
    glm::vec3 sun = sunDirection(settings);
    glm::mat4 lightVP = computeLightViewProj(sun, sceneCenter, sceneRadius);
    glm::vec4 shadowParams(
        std::tan(glm::radians(settings.sunAngularSize * 0.5f)), // light size in UV (penumbra scale)
        settings.shadowNormalBias,
        settings.shadowBias,
        1.0f / (float)kShadowDim);
    float shadowsOn = settings.shadowsEnabled ? 1.0f : 0.0f;

    // ===================== Pass S: Shadow map (depth-only, light POV) =====================
    // Always rendered (cheap) so the map stays in a valid SHADER_READ_ONLY layout
    // for the descriptor; the lighting passes branch on shadowsOn before sampling.
    {
        TracyVkZone(tracy, cmd, "Shadow");
        m_eng->cmdBeginLabel(cmd, "Shadow Pass");
        imgBarrier(cmd, m_shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = m_shadowMap.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = {kShadowDim, kShadowDim};
        ri.layerCount = 1;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport svp{};
        svp.width = (float)kShadowDim; svp.height = (float)kShadowDim; svp.maxDepth = 1.0f;
        VkRect2D sscissor{}; sscissor.extent = {kShadowDim, kShadowDim};
        vkCmdSetViewport(cmd, 0, 1, &svp);
        vkCmdSetScissor(cmd, 0, 1, &sscissor);
        vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 2.0f); // slope-scaled, fights acne
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowLayout,
                                0, 1, &scene.descriptorSet, 0, nullptr);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene.vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, scene.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Casters = opaque + alpha-masked (BLEND/glass casts no shadow).
        for (uint32_t drawID : scene.opaqueIndices) {
            const MeshDraw& d = scene.draws[drawID];
            ShadowPush push{};
            push.lightViewProj = lightVP;
            push.model = d.transform;
            push.materialIndex = d.materialIndex;
            vkCmdPushConstants(cmd, m_shadowLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ShadowPush), &push);
            vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, d.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);

        // Depth attachment -> sampled (read by resolve compute + transparent frag).
        imgBarrier(cmd, m_shadowMap.image, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        m_eng->cmdEndLabel(cmd);
    }

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

        for (uint32_t drawID : scene.opaqueIndices) {
            const MeshDraw& d = scene.draws[drawID];
            VisPush push{};
            push.viewProj = viewProj;
            push.model = d.transform;
            push.drawID = drawID;
            vkCmdPushConstants(cmd, m_visLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(VisPush), &push);
            vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, d.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass B: Resolve (compute -> HDR) =====================
    {
        TracyVkZone(tracy, cmd, "Resolve");
        m_eng->cmdBeginLabel(cmd, "Resolve Pass (PBR+PCSS)");
        imgBarrier(cmd, m_vis.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
        VkDescriptorSet sets[3] = {scene.descriptorSet, m_resolveSet, m_shadowSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveLayout,
                                0, 3, sets, 0, nullptr);

        ResolvePush push{};
        push.viewProj = viewProj;
        push.cameraPos = glm::vec4(cameraPos, 1.0f);
        push.sunDir = glm::vec4(sun, settings.ambient);
        push.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), shadowsOn);
        push.screenSize = glm::uvec2(extent.width, extent.height);
        push.lightViewProj = lightVP;
        push.shadowParams = shadowParams;
        vkCmdPushConstants(cmd, m_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(ResolvePush), &push);

        vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass C: Transparent forward (blend -> HDR) =====================
    // After this, HDR is in GENERAL (no transparency) or COLOR_ATTACHMENT (with).
    bool hasTransparent = !scene.transparentIndices.empty();
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
        VkDescriptorSet tsets[2] = {scene.descriptorSet, m_shadowSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_transparentLayout,
                                0, 2, tsets, 0, nullptr);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &scene.vertexBuffer.buffer, &offset);
        vkCmdBindIndexBuffer(cmd, scene.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

        // Back-to-front sort (centroid ~ transform origin), so blending is correct.
        std::vector<uint32_t> sorted(scene.transparentIndices.begin(),
                                     scene.transparentIndices.end());
        std::sort(sorted.begin(), sorted.end(), [&](uint32_t a, uint32_t b) {
            glm::vec3 ca = glm::vec3(scene.draws[a].transform[3]);
            glm::vec3 cb = glm::vec3(scene.draws[b].transform[3]);
            return glm::dot(ca - cameraPos, ca - cameraPos) >
                   glm::dot(cb - cameraPos, cb - cameraPos);
        });

        for (uint32_t drawID : sorted) {
            const MeshDraw& d = scene.draws[drawID];
            TransparentPush push{};
            push.viewProj = viewProj;
            push.model = d.transform;
            push.cameraPos = glm::vec4(cameraPos, (float)d.materialIndex);
            push.sunDir = glm::vec4(sun, settings.ambient);
            push.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), shadowsOn);
            push.lightViewProj = lightVP;
            push.shadowParams = shadowParams;
            vkCmdPushConstants(cmd, m_transparentLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(TransparentPush), &push);
            vkCmdDrawIndexed(cmd, d.indexCount, 1, d.firstIndex, d.vertexOffset, 0);
        }
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass D: Tonemap (HDR -> swapchain) =====================
    {
        TracyVkZone(tracy, cmd, "Tonemap");
        m_eng->cmdBeginLabel(cmd, "Tonemap Pass");
        // HDR source layout/stage depends on whether the transparent pass ran.
        VkPipelineStageFlags2 hdrStage = hasTransparent
            ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        VkAccessFlags2 hdrAccess = hasTransparent
            ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkImageLayout hdrOld = hasTransparent
            ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   hdrStage, hdrAccess,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   hdrOld, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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
        m_eng->cmdEndLabel(cmd);
    }
}
