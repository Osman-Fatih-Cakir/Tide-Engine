#pragma once
#include "pch.h"

struct GraphicsPipeline {
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
};

// Simple forward mesh pipeline (dynamic rendering). Vertex layout = Vertex,
// push constant = MeshPush. Faz 4 will replace this with the V-buffer path.
GraphicsPipeline createMeshPipeline(VkDevice device, VkFormat colorFormat,
                                    VkFormat depthFormat);

void destroyPipeline(VkDevice device, GraphicsPipeline& p);
