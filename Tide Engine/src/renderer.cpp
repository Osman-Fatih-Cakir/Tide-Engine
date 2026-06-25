#include "renderer.h"
#include "vk_engine.h"
#include "scene.h"
#include "shader.h"
#include "mesh.h"
#include "dlss.h"
#include <cstddef>   // offsetof
#include <cstring>   // memcpy
#include <algorithm> // sort
#include <cmath>     // fabs
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
struct AtrousPush {
    glm::uvec2 screenSize;
    int        step;
    float      phiNormal;
    float      phiDepth;
};
struct TonemapPush {
    float exposure;
};
struct FogScatterPush {
    glm::mat4  invViewProj;
    glm::mat4  prevViewProj;
    glm::vec4  camPos;
    glm::vec4  sunDir;
    glm::vec4  sunColor;
    glm::vec4  fog;      // x=density y=scatter z=anisotropy w=ambient
    glm::uvec4 grid;     // xyz dims, w frame
    glm::vec4  zRange;   // x=zn y=zf z=temporalAlpha w=reset
    glm::vec4  misc;     // x=jitterScale
    glm::vec4  boxMin;   // xyz local fog box min, w = enable
    glm::vec4  boxMax;   // xyz local fog box max, w = edge softness
};
struct FogIntegratePush {
    glm::uvec4 grid;
    glm::vec4  zRange;   // x=zn y=zf
};
struct FogApplyPush {
    glm::uvec4 grid;
    glm::uvec2 screenSize;
    glm::vec2  zRange;   // x=zn y=zf
};

// Froxel near extent (world units). Slightly past the camera near plane.
static constexpr float FOG_ZNEAR = 0.1f;

// Froxel grid dimensions per quality preset. Higher XY resolves thin light shafts
// (small blind gaps); higher Z sharpens shaft depth. Must match the UI labels.
static VkExtent3D fogGridDim(int quality) {
    switch (quality) {
        case 0:  return {160, 90, 64};
        case 1:  return {240, 135, 96};
        case 2:  return {320, 180, 128};
        case 3:  return {480, 270, 160}; // Ultra
        default: return {240, 135, 96};
    }
}
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

// DDGI (Faz 8). Octahedral tile resolutions + per-probe ray budget (SSBO stride).
// Must match shaders/ddgi.glsl.
static constexpr uint32_t DDGI_MAX_RAYS  = 128;
static constexpr uint32_t DDGI_IRR_RES   = 8;
static constexpr uint32_t DDGI_DEPTH_RES = 16;
struct DdgiParams {                 // matches the DdgiParams UBO in ddgi.glsl (std140)
    glm::vec4  gridOrigin;
    glm::vec4  gridSpacing;
    glm::ivec4 gridCounts;          // xyz = Nx,Ny,Nz ; w = raysPerProbe
    glm::vec4  sunDir;              // xyz dir, w = sky/ambient intensity
    glm::vec4  sunColor;
    glm::vec4  params;              // x=hysteresis y=intensity z=normalBias w=frame
    glm::vec4  shadowCfg;           // x=coneRad y=samples z=shadowsOn w=maxRayDist
    glm::vec4  misc;               // x = use GI in resolve (0/1)
};
struct DdgiUpdatePush { int mode; }; // 0 = irradiance atlas, 1 = depth atlas
struct ProbeDebugPush { glm::mat4 viewProj; float radius; };
struct BoxDebugPush { glm::mat4 viewProj; glm::vec4 boxMin; glm::vec4 boxMax; };

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

static Image makeImage3D(VulkanEngine& eng, VkFormat format, VkImageUsageFlags usage,
                         VkExtent3D extent, const char* name) {
    Image img{};
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_3D;
    ici.format = format;
    ici.extent = extent;
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
    vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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
    // À-trous set: shadowIn(read) + shadowOut(write) + normal(read) + directLight(read).
    {
        VkDescriptorSetLayoutBinding b[4]{};
        for (uint32_t i = 0; i < 4; i++) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4;
        lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_atrousSetLayout));
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

    // ---- DDGI set layouts (created early; resolve set2 references the sample one) ----
    {
        // Trace set: b0 UBO, b1 ray SSBO (write), b2 irradiance + b3 depth (prev-frame
        // sampled, for multi-bounce feedback).
        VkDescriptorSetLayoutBinding b[4]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        for (int i = 0; i < 4; i++) { b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4; lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_ddgiTraceSetLayout));
    }
    {
        // Update set: b0 UBO, b1 ray SSBO (read), b2 irradiance (storage), b3 depth (storage).
        VkDescriptorSetLayoutBinding b[4]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        for (int i = 0; i < 4; i++) { b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4; lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_ddgiUpdateSetLayout));
    }
    {
        // Sample set (resolve set2): b0 UBO, b1 irradiance (sampler), b2 depth (sampler).
        // Also used by the probe debug pipeline (vertex reads b0, fragment b0+b1).
        VkDescriptorSetLayoutBinding b[3]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        for (int i = 0; i < 3; i++) { b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                              VK_SHADER_STAGE_FRAGMENT_BIT; }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3; lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_ddgiSampleSetLayout));
    }

    // ---- Descriptor pool + sets (views written in createTargets) ----
    {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        // 2 resolve * 9 storage + 2 composite * 3 + 2 à-trous * 4 = 18 + 6 + 8 = 32.
        sizes[0].descriptorCount = 32;
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = 4;  // 2 tonemap (hdr/dlss) + 2 resolve sets (histRead)
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 8;               // 2 resolve + 2 composite + 2 à-trous + 2 tonemap
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
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_compositeSetB));
        dai.pSetLayouts = &m_atrousSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_atrousSet[0]));
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_atrousSet[1]));
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
        // set2 = DDGI sampling (UBO + irradiance + depth). Built just below; the
        // layout must exist before this pipeline layout — see the DDGI block.
        VkDescriptorSetLayout sets[3] = {sceneSetLayout, m_resolveSetLayout, m_ddgiSampleSetLayout};
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

    // ---- À-trous pipeline (compute, SVGF spatial shadow filter) ----
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/atrous.comp", VK_SHADER_STAGE_COMPUTE_BIT);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(AtrousPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &m_atrousSetLayout;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_atrousLayout));

        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp;
        cpi.stage.pName = "main";
        cpi.layout = m_atrousLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_atrousPipeline));
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

    // ======================= Volumetric fog (Faz 7) =======================
    // Linear 3D sampler (clamp) for froxel reprojection + the apply lookup.
    {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device, &sci, nullptr, &m_froxelSampler));
    }
    // Fog descriptor set layouts.
    {
        // Scatter set: b0 scatterOut (storage), b1 scatterPrev (sampler).
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2; lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_fogScatterSetLayout));
    }
    {
        // Integrate set: b0 scatterIn (storage), b1 integratedOut (storage).
        VkDescriptorSetLayoutBinding b[2]{};
        for (uint32_t i = 0; i < 2; i++) {
            b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2; lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_fogIntegrateSetLayout));
    }
    {
        // Apply set: b0 integrated (sampler), b1 hdr (storage rw), b2 directLight (storage read).
        VkDescriptorSetLayoutBinding b[3]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3; lci.pBindings = b;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &lci, nullptr, &m_fogApplySetLayout));
    }
    // Fog descriptor pool + sets (views written in createTargets).
    {
        VkDescriptorPoolSize sizes[2]{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[0].descriptorCount = 10; // scatter 2*2 + integrate 2*2 + apply 2
        sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = 3; // scatter 2 (prev) + apply 1 (integrated)
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 5; // 2 scatter + 2 integrate + 1 apply
        pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &m_fogPool));

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = m_fogPool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m_fogScatterSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_fogScatterSet[0]));
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_fogScatterSet[1]));
        dai.pSetLayouts = &m_fogIntegrateSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_fogIntegrateSet[0]));
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_fogIntegrateSet[1]));
        dai.pSetLayouts = &m_fogApplySetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_fogApplySet));
    }
    // Scatter pipeline: set0 scene (TLAS), set1 fog scatter.
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/froxel_scatter.comp", VK_SHADER_STAGE_COMPUTE_BIT);
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(FogScatterPush);
        VkDescriptorSetLayout sets[2] = {sceneSetLayout, m_fogScatterSetLayout};
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 2; lci.pSetLayouts = sets;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_fogScatterLayout));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp; cpi.stage.pName = "main";
        cpi.layout = m_fogScatterLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_fogScatterPipeline));
        vkDestroyShaderModule(device, comp, nullptr);
    }
    // Integrate pipeline.
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/froxel_integrate.comp", VK_SHADER_STAGE_COMPUTE_BIT);
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(FogIntegratePush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1; lci.pSetLayouts = &m_fogIntegrateSetLayout;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_fogIntegrateLayout));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp; cpi.stage.pName = "main";
        cpi.layout = m_fogIntegrateLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_fogIntegratePipeline));
        vkDestroyShaderModule(device, comp, nullptr);
    }
    // Apply pipeline.
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/fog_apply.comp", VK_SHADER_STAGE_COMPUTE_BIT);
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(FogApplyPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1; lci.pSetLayouts = &m_fogApplySetLayout;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_fogApplyLayout));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp; cpi.stage.pName = "main";
        cpi.layout = m_fogApplyLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_fogApplyPipeline));
        vkDestroyShaderModule(device, comp, nullptr);
    }

    // ======================= DDGI (Faz 8) =======================
    // Shared params UBO (host-visible, persistently mapped; updated each frame).
    m_ddgiUbo = createBuffer(eng.allocator(), sizeof(DdgiParams),
                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT);
    // Pool + one set each (atlas/buffer views written in createTargets).
    {
        VkDescriptorPoolSize sizes[4]{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         sizes[0].descriptorCount = 3;
        sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;        sizes[1].descriptorCount = 2;
        sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;         sizes[2].descriptorCount = 2;
        sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sizes[3].descriptorCount = 4; // trace 2 + sample 2
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 3; pci.poolSizeCount = 4; pci.pPoolSizes = sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &m_ddgiPool));

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = m_ddgiPool; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m_ddgiTraceSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_ddgiTraceSet));
        dai.pSetLayouts = &m_ddgiUpdateSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_ddgiUpdateSet));
        dai.pSetLayouts = &m_ddgiSampleSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &dai, &m_ddgiSampleSet));
    }
    // Trace pipeline: set0 scene (TLAS + bindless), set1 trace.
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/ddgi_trace.comp", VK_SHADER_STAGE_COMPUTE_BIT);
        VkDescriptorSetLayout sets[2] = {sceneSetLayout, m_ddgiTraceSetLayout};
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 2; lci.pSetLayouts = sets;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_ddgiTraceLayout));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp; cpi.stage.pName = "main";
        cpi.layout = m_ddgiTraceLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_ddgiTracePipeline));
        vkDestroyShaderModule(device, comp, nullptr);
    }
    // Update pipeline: set0 update, push = mode.
    {
        VkShaderModule comp = loadShaderModule(device, "shaders/ddgi_update.comp", VK_SHADER_STAGE_COMPUTE_BIT);
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc.size = sizeof(DdgiUpdatePush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1; lci.pSetLayouts = &m_ddgiUpdateSetLayout;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_ddgiUpdateLayout));
        VkComputePipelineCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = comp; cpi.stage.pName = "main";
        cpi.layout = m_ddgiUpdateLayout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &m_ddgiUpdatePipeline));
        vkDestroyShaderModule(device, comp, nullptr);
    }
    // Probe debug pipeline: instanced icosahedron spheres into HDR, depth-tested.
    {
        VkShaderModule vert = loadShaderModule(device, "shaders/probe_debug.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkShaderModule frag = loadShaderModule(device, "shaders/probe_debug.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag; stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{}; // no vertex buffers (generated in VS)
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
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
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.size = sizeof(ProbeDebugPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.setLayoutCount = 1; lci.pSetLayouts = &m_ddgiSampleSetLayout;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_probeDebugLayout));

        VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = &hdrFormat;
        rendering.depthAttachmentFormat = depthFormat;
        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &rendering;
        gpi.stageCount = 2; gpi.pStages = stages;
        gpi.pVertexInputState = &vi; gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vp; gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &ds;
        gpi.pColorBlendState = &cb; gpi.pDynamicState = &dyn;
        gpi.layout = m_probeDebugLayout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &m_probeDebugPipeline));
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }
    // Fog box debug pipeline: line-list wireframe into HDR, depth-tested, no sets.
    {
        VkShaderModule vert = loadShaderModule(device, "shaders/box_debug.vert", VK_SHADER_STAGE_VERTEX_BIT);
        VkShaderModule frag = loadShaderModule(device, "shaders/box_debug.frag", VK_SHADER_STAGE_FRAGMENT_BIT);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag; stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        // No depth test: the box is a placement overlay, drawn on top so it stays
        // visible even when its edges sit behind the room walls (the common case for
        // an auto-fit box hugging the scene bounds).
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
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
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.size = sizeof(BoxDebugPush);
        VkPipelineLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
        VK_CHECK(vkCreatePipelineLayout(device, &lci, nullptr, &m_boxDebugLayout));

        VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo rendering{};
        rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        rendering.colorAttachmentCount = 1; rendering.pColorAttachmentFormats = &hdrFormat;
        rendering.depthAttachmentFormat = depthFormat;
        VkGraphicsPipelineCreateInfo gpi{};
        gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpi.pNext = &rendering;
        gpi.stageCount = 2; gpi.pStages = stages;
        gpi.pVertexInputState = &vi; gpi.pInputAssemblyState = &ia;
        gpi.pViewportState = &vp; gpi.pRasterizationState = &rs;
        gpi.pMultisampleState = &ms; gpi.pDepthStencilState = &ds;
        gpi.pColorBlendState = &cb; gpi.pDynamicState = &dyn;
        gpi.layout = m_boxDebugLayout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &m_boxDebugPipeline));
        vkDestroyShaderModule(device, vert, nullptr);
        vkDestroyShaderModule(device, frag, nullptr);
    }
}

void Renderer::createTargets(VulkanEngine& eng, VkExtent2D extent, VkExtent2D displayExtent,
                             const Settings& settings) {
    destroyTargets(eng);
    m_extent = extent;
    m_displayExtent = displayExtent;
    m_dlssReset = true; // history is invalid after (re)creating resources
    m_froxelDim = fogGridDim(settings.fogQuality);
    m_haveFogHistory = false;
    m_fogHistIndex = 0;

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

    // Froxel volumes (3D). Kept permanently in GENERAL: storage writes + sampler reads.
    VkImageUsageFlags froxelUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    for (int i = 0; i < 2; i++)
        m_froxelScatter[i] = makeImage3D(eng, VK_FORMAT_R16G16B16A16_SFLOAT, froxelUsage,
                                         m_froxelDim, i == 0 ? "Froxel Scatter 0" : "Froxel Scatter 1");
    m_froxelIntegrated = makeImage3D(eng, VK_FORMAT_R16G16B16A16_SFLOAT, froxelUsage,
                                     m_froxelDim, "Froxel Integrated");

    // ----- DDGI atlases + ray SSBO (sized to the active probe grid) -----
    m_ddgiCounts = glm::ivec3(std::max(1, settings.giProbesX),
                              std::max(1, settings.giProbesY),
                              std::max(1, settings.giProbesZ));
    m_ddgiHaveHistory = false;
    int Nx = m_ddgiCounts.x, Ny = m_ddgiCounts.y, Nz = m_ddgiCounts.z;
    VkExtent2D irrExt   = {(uint32_t)(Nx * DDGI_IRR_RES),   (uint32_t)(Ny * Nz * DDGI_IRR_RES)};
    VkExtent2D depthExt = {(uint32_t)(Nx * DDGI_DEPTH_RES), (uint32_t)(Ny * Nz * DDGI_DEPTH_RES)};
    VkImageUsageFlags atlasUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT; // TRANSFER_DST: cleared below
    m_ddgiIrradiance = makeImage(eng, VK_FORMAT_R16G16B16A16_SFLOAT, atlasUsage,
                                 irrExt, VK_IMAGE_ASPECT_COLOR_BIT, "DDGI Irradiance");
    m_ddgiDepth = makeImage(eng, VK_FORMAT_R16G16_SFLOAT, atlasUsage,
                            depthExt, VK_IMAGE_ASPECT_COLOR_BIT, "DDGI Depth");
    VkDeviceSize rayBytes = (VkDeviceSize)Nx * Ny * Nz * DDGI_MAX_RAYS * sizeof(glm::vec4);
    m_ddgiRays = createBuffer(eng.allocator(), rayBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                              VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    // Initialize history images + shadowOut2 + froxel volumes to GENERAL so the per-frame
    // GENERAL->GENERAL barriers are always valid. (shadowHist read needs it; shadowOut2 is
    // only ever touched by à-trous, which assumes GENERAL — resolve never transitions it.)
    eng.immediateSubmit([&](VkCommandBuffer cmd) {
        for (VkImage g : {m_shadowHist[0].image, m_shadowHist[1].image, m_shadowOut2.image,
                          m_froxelScatter[0].image, m_froxelScatter[1].image, m_froxelIntegrated.image})
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        // DDGI atlases: clear to 0 (so first-frame temporal mix reads no NaNs), then GENERAL.
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (VkImage g : {m_ddgiIrradiance.image, m_ddgiDepth.image}) {
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdClearColorImage(cmd, g, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
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
    VkDescriptorImageInfo directStore{}, shadowOutStore{}, shadowOut2Store{};
    directStore.imageView     = m_directLight.view; directStore.imageLayout     = VK_IMAGE_LAYOUT_GENERAL;
    shadowOutStore.imageView  = m_shadowOut.view;   shadowOutStore.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
    shadowOut2Store.imageView = m_shadowOut2.view;  shadowOut2Store.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
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
    // Composite: directLight(read) + shadow(read) + hdr(read/write). Two variants:
    // A reads shadowOut (Off/Temporal/RR), B reads shadowOut2 (after odd à-trous count).
    add(m_compositeSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);
    add(m_compositeSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOutStore);
    add(m_compositeSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrStore);
    add(m_compositeSetB, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);
    add(m_compositeSetB, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOut2Store);
    add(m_compositeSetB, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrStore);
    // À-trous ping-pong: set[0] shadowOut->shadowOut2, set[1] shadowOut2->shadowOut.
    add(m_atrousSet[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOutStore);
    add(m_atrousSet[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOut2Store);
    add(m_atrousSet[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normStore);
    add(m_atrousSet[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);
    add(m_atrousSet[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOut2Store);
    add(m_atrousSet[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowOutStore);
    add(m_atrousSet[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normStore);
    add(m_atrousSet[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);
    add(m_tonemapSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdrSamp);
    add(m_tonemapSetDlss, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dlssSamp);

    // ----- Fog descriptor writes (all froxel volumes live in GENERAL) -----
    VkDescriptorImageInfo froxelStore[2]{}, froxelSamp[2]{}, integStore{}, integSamp{};
    for (int i = 0; i < 2; i++) {
        froxelStore[i].imageView = m_froxelScatter[i].view;
        froxelStore[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        froxelSamp[i].sampler = m_froxelSampler;
        froxelSamp[i].imageView = m_froxelScatter[i].view;
        froxelSamp[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    integStore.imageView = m_froxelIntegrated.view; integStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    integSamp.sampler = m_froxelSampler;
    integSamp.imageView = m_froxelIntegrated.view;  integSamp.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // Scatter set[k]: writes scatter[k] (b0), reads scatter[1-k] (b1).
    for (int k = 0; k < 2; k++) {
        add(m_fogScatterSet[k], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &froxelStore[k]);
        add(m_fogScatterSet[k], 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &froxelSamp[1 - k]);
    }
    // Integrate set[k]: reads scatter[k] (b0), writes integrated (b1).
    for (int k = 0; k < 2; k++) {
        add(m_fogIntegrateSet[k], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &froxelStore[k]);
        add(m_fogIntegrateSet[k], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &integStore);
    }
    // Apply set: integrated (b0, sampled), hdr (b1, rw), directLight (b2, read .a depth).
    add(m_fogApplySet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &integSamp);
    add(m_fogApplySet, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hdrStore);
    add(m_fogApplySet, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &directStore);

    // ----- DDGI descriptor writes -----
    VkDescriptorBufferInfo uboInfo{m_ddgiUbo.buffer, 0, sizeof(DdgiParams)};
    VkDescriptorBufferInfo rayInfo{m_ddgiRays.buffer, 0, VK_WHOLE_SIZE};
    auto addBuf = [&](VkDescriptorSet set, uint32_t bind, VkDescriptorType t,
                      const VkDescriptorBufferInfo* info) {
        VkWriteDescriptorSet x{};
        x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x.dstSet = set; x.dstBinding = bind; x.descriptorType = t;
        x.descriptorCount = 1; x.pBufferInfo = info;
        w.push_back(x);
    };
    VkDescriptorImageInfo ddgiIrrStore{}, ddgiDepthStore{}, ddgiIrrSamp{}, ddgiDepthSamp{};
    ddgiIrrStore.imageView   = m_ddgiIrradiance.view; ddgiIrrStore.imageLayout   = VK_IMAGE_LAYOUT_GENERAL;
    ddgiDepthStore.imageView = m_ddgiDepth.view;      ddgiDepthStore.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ddgiIrrSamp.sampler = m_hdrSampler;   ddgiIrrSamp.imageView = m_ddgiIrradiance.view; ddgiIrrSamp.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ddgiDepthSamp.sampler = m_hdrSampler; ddgiDepthSamp.imageView = m_ddgiDepth.view;    ddgiDepthSamp.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // Trace: UBO + ray SSBO (write) + prev-frame irradiance/depth (sampled, multi-bounce).
    addBuf(m_ddgiTraceSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
    addBuf(m_ddgiTraceSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rayInfo);
    add(m_ddgiTraceSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ddgiIrrSamp);
    add(m_ddgiTraceSet, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ddgiDepthSamp);
    // Update: UBO + ray SSBO (read) + irradiance/depth (storage).
    addBuf(m_ddgiUpdateSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
    addBuf(m_ddgiUpdateSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rayInfo);
    add(m_ddgiUpdateSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ddgiIrrStore);
    add(m_ddgiUpdateSet, 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ddgiDepthStore);
    // Sample (resolve set2): UBO + irradiance/depth (sampled).
    addBuf(m_ddgiSampleSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboInfo);
    add(m_ddgiSampleSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ddgiIrrSamp);
    add(m_ddgiSampleSet, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ddgiDepthSamp);

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
    destroyImage(eng.allocator(), eng.device(), m_froxelScatter[0]);
    destroyImage(eng.allocator(), eng.device(), m_froxelScatter[1]);
    destroyImage(eng.allocator(), eng.device(), m_froxelIntegrated);
    destroyImage(eng.allocator(), eng.device(), m_ddgiIrradiance);
    destroyImage(eng.allocator(), eng.device(), m_ddgiDepth);
    destroyBuffer(eng.allocator(), m_ddgiRays);
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
    if (m_atrousPipeline) vkDestroyPipeline(device, m_atrousPipeline, nullptr);
    if (m_atrousLayout) vkDestroyPipelineLayout(device, m_atrousLayout, nullptr);
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_tonemapLayout) vkDestroyPipelineLayout(device, m_tonemapLayout, nullptr);
    if (m_resolveSetLayout) vkDestroyDescriptorSetLayout(device, m_resolveSetLayout, nullptr);
    if (m_compositeSetLayout) vkDestroyDescriptorSetLayout(device, m_compositeSetLayout, nullptr);
    if (m_atrousSetLayout) vkDestroyDescriptorSetLayout(device, m_atrousSetLayout, nullptr);
    if (m_tonemapSetLayout) vkDestroyDescriptorSetLayout(device, m_tonemapSetLayout, nullptr);
    if (m_pool) vkDestroyDescriptorPool(device, m_pool, nullptr);
    // Fog.
    if (m_froxelSampler) vkDestroySampler(device, m_froxelSampler, nullptr);
    if (m_fogScatterPipeline) vkDestroyPipeline(device, m_fogScatterPipeline, nullptr);
    if (m_fogScatterLayout) vkDestroyPipelineLayout(device, m_fogScatterLayout, nullptr);
    if (m_fogIntegratePipeline) vkDestroyPipeline(device, m_fogIntegratePipeline, nullptr);
    if (m_fogIntegrateLayout) vkDestroyPipelineLayout(device, m_fogIntegrateLayout, nullptr);
    if (m_fogApplyPipeline) vkDestroyPipeline(device, m_fogApplyPipeline, nullptr);
    if (m_fogApplyLayout) vkDestroyPipelineLayout(device, m_fogApplyLayout, nullptr);
    if (m_fogScatterSetLayout) vkDestroyDescriptorSetLayout(device, m_fogScatterSetLayout, nullptr);
    if (m_fogIntegrateSetLayout) vkDestroyDescriptorSetLayout(device, m_fogIntegrateSetLayout, nullptr);
    if (m_fogApplySetLayout) vkDestroyDescriptorSetLayout(device, m_fogApplySetLayout, nullptr);
    if (m_fogPool) vkDestroyDescriptorPool(device, m_fogPool, nullptr);
    // DDGI.
    destroyBuffer(eng.allocator(), m_ddgiUbo);
    if (m_ddgiTracePipeline) vkDestroyPipeline(device, m_ddgiTracePipeline, nullptr);
    if (m_ddgiTraceLayout) vkDestroyPipelineLayout(device, m_ddgiTraceLayout, nullptr);
    if (m_ddgiUpdatePipeline) vkDestroyPipeline(device, m_ddgiUpdatePipeline, nullptr);
    if (m_ddgiUpdateLayout) vkDestroyPipelineLayout(device, m_ddgiUpdateLayout, nullptr);
    if (m_ddgiTraceSetLayout) vkDestroyDescriptorSetLayout(device, m_ddgiTraceSetLayout, nullptr);
    if (m_ddgiUpdateSetLayout) vkDestroyDescriptorSetLayout(device, m_ddgiUpdateSetLayout, nullptr);
    if (m_ddgiSampleSetLayout) vkDestroyDescriptorSetLayout(device, m_ddgiSampleSetLayout, nullptr);
    if (m_ddgiPool) vkDestroyDescriptorPool(device, m_ddgiPool, nullptr);
    if (m_probeDebugPipeline) vkDestroyPipeline(device, m_probeDebugPipeline, nullptr);
    if (m_probeDebugLayout) vkDestroyPipelineLayout(device, m_probeDebugLayout, nullptr);
    if (m_boxDebugPipeline) vkDestroyPipeline(device, m_boxDebugPipeline, nullptr);
    if (m_boxDebugLayout) vkDestroyPipelineLayout(device, m_boxDebugLayout, nullptr);
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

    // Denoise mode: 0 Off, 1 Temporal, 2 SVGF (= temporal + à-trous). DLSS Ray
    // Reconstruction owns denoising when active -> feed it RAW shadow (our pipeline off).
    int  mode    = dlssActive ? 0 : settings.shadowDenoiseMode;
    bool temporalOn = mode >= 1;     // temporal accumulation (modes 1,2)
    bool svgfOn     = mode == 2;     // à-trous spatial filter
    // Reset temporal history only on a LARGE sun jump (e.g. dragging the slider).
    // A slow animation moves the sun a tiny amount per frame, which the temporal
    // accumulation should follow (not reset), or shadows/fog stay noisy.
    bool sunMoved = std::fabs(settings.sunAzimuthDeg - m_prevSunAz) > 1.0f ||
                    std::fabs(settings.sunElevationDeg - m_prevSunEl) > 1.0f;
    bool reset = !m_haveHistory || sunMoved || !temporalOn;
    glm::vec4 temporal(reset ? 1.0f : 0.0f,
                       settings.shadowHistAlpha,
                       temporalOn ? 1.0f : 0.0f, 0.0f);
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

    // ===================== Pass A2: DDGI (probe trace + update) =====================
    // World-space probe grid -> octahedral irradiance/depth atlases, read by resolve
    // to replace flat ambient. The UBO is filled every frame (misc.x gates the
    // resolve sampling) so the GI-off path stays a clean regression.
    bool giOn = settings.giEnabled && !settings.debugMotionVecs;
    {
        // Grid bounds come from Settings (auto-fit to scene by the engine, or placed by
        // hand when giGridManual). origin = min, spacing spans min..max over the probes.
        glm::vec3 omin = settings.giGridMin, omax = settings.giGridMax;
        glm::vec3 ext = omax - omin;
        glm::vec3 spacing(1.0f);
        for (int i = 0; i < 3; i++)
            spacing[i] = (m_ddgiCounts[i] > 1) ? (omax[i] - omin[i]) / float(m_ddgiCounts[i] - 1)
                                               : (omax[i] - omin[i]);
        int rays = std::max(1, std::min(settings.giRaysPerProbe, (int)DDGI_MAX_RAYS));
        float maxDist = glm::length(ext) * 1.5f + 1.0f;

        DdgiParams dp{};
        dp.gridOrigin  = glm::vec4(omin, 0.0f);
        dp.gridSpacing = glm::vec4(spacing, 0.0f);
        dp.gridCounts  = glm::ivec4(m_ddgiCounts, rays);
        dp.sunDir      = glm::vec4(sun, settings.ambient); // w = sky intensity on miss
        dp.sunColor    = glm::vec4(glm::vec3(settings.sunIntensity), 0.0f);
        dp.params      = glm::vec4(m_ddgiHaveHistory ? settings.giHysteresis : 0.0f,
                                   settings.giIntensity, settings.giNormalBias,
                                   (float)(frame & 0xFFFF));
        dp.shadowCfg   = glm::vec4(0.0f, 0.0f, settings.shadowsEnabled ? 1.0f : 0.0f, maxDist);
        // misc: x = use GI in resolve, y = sky GI strength, z = multi-bounce gain.
        dp.misc        = glm::vec4(giOn ? 1.0f : 0.0f, settings.giSkyIntensity,
                                   settings.giMultiBounce, 0.0f);
        memcpy(m_ddgiUbo.mapped, &dp, sizeof(dp));

        if (giOn) {
            TracyVkZone(tracy, cmd, "DDGI");
            m_eng->cmdBeginLabel(cmd, "DDGI Pass (trace + update)");
            uint32_t probeCount = (uint32_t)(m_ddgiCounts.x * m_ddgiCounts.y * m_ddgiCounts.z);

            auto memBarrier = [&](VkAccessFlags2 src, VkAccessFlags2 dst) {
                VkMemoryBarrier2 mb{};
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mb.srcAccessMask = src;
                mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mb.dstAccessMask = dst;
                VkDependencyInfo dep{}; dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
                vkCmdPipelineBarrier2(cmd, &dep);
            };

            // Prev frame's update read the ray buffer; this frame's trace overwrites it.
            memBarrier(VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            // Prev frame's atlases (storage-written) -> sampled by trace (multi-bounce feedback).
            for (VkImage g : {m_ddgiIrradiance.image, m_ddgiDepth.image})
                imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

            // ---- Trace: one workgroup per probe (128 rays/group) ----
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ddgiTracePipeline);
            VkDescriptorSet tset[2] = {scene.descriptorSet, m_ddgiTraceSet};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ddgiTraceLayout,
                                    0, 2, tset, 0, nullptr);
            vkCmdDispatch(cmd, probeCount, 1, 1);

            // Trace writes -> update reads the ray buffer.
            memBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            // Trace sampled the atlases -> update overwrites them (WAR).
            for (VkImage g : {m_ddgiIrradiance.image, m_ddgiDepth.image})
                imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

            // ---- Update: gather into irradiance (mode 0) then depth (mode 1) atlases ----
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ddgiUpdatePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ddgiUpdateLayout,
                                    0, 1, &m_ddgiUpdateSet, 0, nullptr);
            uint32_t irrW = (uint32_t)(m_ddgiCounts.x * DDGI_IRR_RES);
            uint32_t irrH = (uint32_t)(m_ddgiCounts.y * m_ddgiCounts.z * DDGI_IRR_RES);
            uint32_t depW = (uint32_t)(m_ddgiCounts.x * DDGI_DEPTH_RES);
            uint32_t depH = (uint32_t)(m_ddgiCounts.y * m_ddgiCounts.z * DDGI_DEPTH_RES);
            DdgiUpdatePush up{0};
            vkCmdPushConstants(cmd, m_ddgiUpdateLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(up), &up);
            vkCmdDispatch(cmd, (irrW + 7) / 8, (irrH + 7) / 8, 1);
            up.mode = 1;
            vkCmdPushConstants(cmd, m_ddgiUpdateLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(up), &up);
            vkCmdDispatch(cmd, (depW + 7) / 8, (depH + 7) / 8, 1);

            // Atlases written (storage) -> sampled by resolve.
            for (VkImage g : {m_ddgiIrradiance.image, m_ddgiDepth.image})
                imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            m_eng->cmdEndLabel(cmd);
            m_ddgiHaveHistory = true;
        }
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
        VkDescriptorSet sets[3] = {scene.descriptorSet, m_resolveSet[cur], m_ddgiSampleSet};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolveLayout,
                                0, 3, sets, 0, nullptr);

        ResolvePush push{};
        push.viewProj = jitteredVP;          // reconstruction must match the jittered raster
        push.prevViewProj = m_prevViewProj;  // unjittered previous frame
        push.cameraPos = glm::vec4(cameraPos, 1.0f);
        push.sunDir = glm::vec4(sun, settings.ambient);
        push.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), 0.0f);
        push.shadowCfg = shadowCfg;
        push.temporal = temporal;
        // DLSS mip LOD bias: log2(renderRes/displayRes) (negative) keeps textures sharp
        // for the upscaler; 0 when rendering at native res.
        float mipBias = dlssActive
            ? std::log2((float)renderExtent.width / (float)displayExtent.width) : 0.0f;
        push.jitter = glm::vec4(jitterNDC, settings.debugMotionVecs ? 1.0f : 0.0f, mipBias);
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

    // ===================== Pass B1: À-trous (SVGF spatial shadow filter) =====================
    // Edge-aware wavelet, ping-pong shadowOut <-> shadowOut2 with doubling step.
    // finalInB tracks which buffer holds the result (composite reads that one).
    bool finalInB = false;
    bool doAtrous = svgfOn && settings.shadowsEnabled && !settings.debugMotionVecs;
    if (doAtrous) {
        TracyVkZone(tracy, cmd, "A-trous");
        m_eng->cmdBeginLabel(cmd, "A-trous Pass (SVGF)");
        // normal + directLight (depth in .a): resolve wrote them -> à-trous reads them.
        for (VkImage g : {m_gbufNormal.image, m_directLight.image})
            imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

        int iters = std::max(1, std::min(settings.svgfIterations, 5));
        for (int i = 0; i < iters; i++) {
            // Hazard sync between passes: both shadow buffers read+write either way.
            for (VkImage g : {m_shadowOut.image, m_shadowOut2.image})
                imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousPipeline);
            // set[0]: shadowOut->shadowOut2, set[1]: shadowOut2->shadowOut.
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousLayout,
                                    0, 1, &m_atrousSet[i & 1], 0, nullptr);
            AtrousPush ap{glm::uvec2(extent.width, extent.height), 1 << i,
                          settings.svgfPhiNormal, settings.svgfPhiDepth};
            vkCmdPushConstants(cmd, m_atrousLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(AtrousPush), &ap);
            vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
            finalInB = ((i & 1) == 0); // even pass writes shadowOut2 (B)
        }
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Pass B2: Composite (deferred shadow recombine) =====================
    // hdr (ambient) += directLight * shadow. Skipped in motion-vector debug (resolve already
    // wrote the debug visualization straight into hdr).
    if (!settings.debugMotionVecs) {
        TracyVkZone(tracy, cmd, "Composite");
        m_eng->cmdBeginLabel(cmd, "Composite Pass (shadow recombine)");
        VkImage shadowFinalImg = finalInB ? m_shadowOut2.image : m_shadowOut.image;
        VkDescriptorSet compSet = finalInB ? m_compositeSetB : m_compositeSet;
        // directLight + final shadow: written (resolve or à-trous) -> composite reads them.
        for (VkImage g : {m_directLight.image, shadowFinalImg})
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
                                0, 1, &compSet, 0, nullptr);
        CompositePush cp{glm::uvec2(extent.width, extent.height)};
        vkCmdPushConstants(cmd, m_compositeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(CompositePush), &cp);
        vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        m_eng->cmdEndLabel(cmd);
    }

    // ===================== Volumetric Fog (Faz 7) =====================
    // Build the froxel volume now (scatter -> integrate). The APPLY onto HDR is
    // deferred until AFTER the transparent pass so glass is fogged too (it draws
    // later and writes no depth). directLight.a carries linear depth for the lookup.
    bool fogOn = settings.fogEnabled && !settings.debugMotionVecs;
    glm::uvec4 fogGrid(m_froxelDim.width, m_froxelDim.height, m_froxelDim.depth, frame & 0xFFFFu);
    float fogZn = FOG_ZNEAR, fogZf = settings.fogMaxDistance;
    {
        if (fogOn) {
            TracyVkZone(tracy, cmd, "Volumetric");
            m_eng->cmdBeginLabel(cmd, "Volumetric Build");
            uint32_t fcur = m_fogHistIndex;
            glm::uvec4 grid = fogGrid;
            float zn = fogZn, zf = fogZf;
            bool fogReset = !m_haveFogHistory || sunMoved;

            // ---- Scatter (per froxel) ----
            for (VkImage g : {m_froxelScatter[0].image, m_froxelScatter[1].image})
                imgBarrier(cmd, g, VK_IMAGE_ASPECT_COLOR_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                           VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fogScatterPipeline);
            VkDescriptorSet sSets[2] = {scene.descriptorSet, m_fogScatterSet[fcur]};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fogScatterLayout,
                                    0, 2, sSets, 0, nullptr);
            FogScatterPush sp{};
            sp.invViewProj = glm::inverse(viewProj);
            sp.prevViewProj = m_prevFogViewProj;
            sp.camPos = glm::vec4(cameraPos, 1.0f);
            sp.sunDir = glm::vec4(sun, 0.0f);
            sp.sunColor = glm::vec4(glm::vec3(settings.sunIntensity), 0.0f);
            sp.fog = glm::vec4(settings.fogDensity, settings.fogScatter,
                               settings.fogAnisotropy, settings.fogAmbient);
            sp.grid = grid;
            sp.zRange = glm::vec4(zn, zf, settings.fogTemporalAlpha, fogReset ? 1.0f : 0.0f);
            sp.misc = glm::vec4(settings.fogJitter ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
            sp.boxMin = glm::vec4(settings.fogBoxMin, settings.fogBoxEnabled ? 1.0f : 0.0f);
            sp.boxMax = glm::vec4(settings.fogBoxMax, settings.fogBoxEdge);
            vkCmdPushConstants(cmd, m_fogScatterLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(FogScatterPush), &sp);
            vkCmdDispatch(cmd, (m_froxelDim.width + 3) / 4, (m_froxelDim.height + 3) / 4,
                          (m_froxelDim.depth + 3) / 4);

            // ---- Integrate (front-to-back along Z) ----
            imgBarrier(cmd, m_froxelScatter[fcur].image, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            imgBarrier(cmd, m_froxelIntegrated.image, VK_IMAGE_ASPECT_COLOR_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fogIntegratePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fogIntegrateLayout,
                                    0, 1, &m_fogIntegrateSet[fcur], 0, nullptr);
            FogIntegratePush ip{grid, glm::vec4(zn, zf, (float)settings.fogBlurRadius, 0.0f)};
            vkCmdPushConstants(cmd, m_fogIntegrateLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(FogIntegratePush), &ip);
            vkCmdDispatch(cmd, (m_froxelDim.width + 7) / 8, (m_froxelDim.height + 7) / 8, 1);

            m_eng->cmdEndLabel(cmd);
            m_haveFogHistory = true;
            m_fogHistIndex = 1 - fcur;
        } else {
            m_haveFogHistory = false; // volume goes stale while fog is off
        }
        m_prevFogViewProj = viewProj;
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

    // ===================== Volumetric Fog APPLY (after transparency) =====================
    // Composites the prebuilt fog volume onto HDR now, so glass (drawn above) is fogged
    // too. Uses the opaque depth (directLight.a) for the slice -> glass is fogged by the
    // distance to the surface behind it (cheap; transparency writes no depth anyway).
    if (fogOn) {
        TracyVkZone(tracy, cmd, "Volumetric Apply");
        m_eng->cmdBeginLabel(cmd, "Volumetric Fog Apply");
        imgBarrier(cmd, m_froxelIntegrated.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        imgBarrier(cmd, m_directLight.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        // hdr: from its post-transparent state -> GENERAL for compute read+write.
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   hdrStage, hdrAccess,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   hdrOld, VK_IMAGE_LAYOUT_GENERAL);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fogApplyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fogApplyLayout,
                                0, 1, &m_fogApplySet, 0, nullptr);
        FogApplyPush ap{fogGrid, glm::uvec2(extent.width, extent.height), glm::vec2(fogZn, fogZf)};
        vkCmdPushConstants(cmd, m_fogApplyLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(FogApplyPush), &ap);
        vkCmdDispatch(cmd, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
        m_eng->cmdEndLabel(cmd);
        // HDR now lives in GENERAL, last written by the fog apply compute.
        hdrStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        hdrAccess = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        hdrOld = VK_IMAGE_LAYOUT_GENERAL;
    }

    // ===================== Probe debug viz (instanced spheres into HDR) =====================
    if (settings.giDebugProbes) {
        m_eng->cmdBeginLabel(cmd, "DDGI Probe Debug");
        // HDR -> color attachment (from whatever its current state is).
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   hdrStage, hdrAccess,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                   hdrOld, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        imgBarrier(cmd, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = m_hdr.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = depthView;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = extent; ri.layerCount = 1;
        ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_probeDebugPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_probeDebugLayout,
                                0, 1, &m_ddgiSampleSet, 0, nullptr);
        glm::vec3 gext = settings.giGridMax - settings.giGridMin;
        float minSp = 1e9f;
        for (int i = 0; i < 3; i++)
            if (m_ddgiCounts[i] > 1) minSp = std::min(minSp, gext[i] / float(m_ddgiCounts[i] - 1));
        ProbeDebugPush dpush{jitteredVP, (minSp < 1e9f ? minSp : 1.0f) * 0.15f};
        vkCmdPushConstants(cmd, m_probeDebugLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(dpush), &dpush);
        uint32_t probeCount = (uint32_t)(m_ddgiCounts.x * m_ddgiCounts.y * m_ddgiCounts.z);
        vkCmdDraw(cmd, 60, probeCount, 0, 0); // 60 = icosahedron vertices
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
        hdrStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        hdrAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        hdrOld = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    // ===================== Fog box debug viz (wireframe into HDR) =====================
    if (settings.fogDebugBox) {
        m_eng->cmdBeginLabel(cmd, "Fog Box Debug");
        imgBarrier(cmd, m_hdr.image, VK_IMAGE_ASPECT_COLOR_BIT,
                   hdrStage, hdrAccess,
                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
                   hdrOld, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        imgBarrier(cmd, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = m_hdr.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = depthView;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = extent; ri.layerCount = 1;
        ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
        ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_boxDebugPipeline);
        BoxDebugPush bpush{jitteredVP, glm::vec4(settings.fogBoxMin, 0.0f),
                           glm::vec4(settings.fogBoxMax, 0.0f)};
        vkCmdPushConstants(cmd, m_boxDebugLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(bpush), &bpush);
        vkCmdDraw(cmd, 24, 1, 0, 0); // 12 edges x 2 vertices
        vkCmdEndRendering(cmd);
        m_eng->cmdEndLabel(cmd);
        hdrStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        hdrAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        hdrOld = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

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
