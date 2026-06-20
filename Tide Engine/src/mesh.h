#pragma once
#include "pch.h"

// Interleaved vertex. Tangent comes later (Faz 6/7 normal mapping).
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

// Material as it will live in the bindless SSBO. Texture fields are indices into
// the bindless texture array (-1 = none). std430-friendly layout.
struct GpuMaterial {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    int   baseColorTexture = -1;
    int   normalTexture    = -1;
    int   metalRoughTexture = -1;
    float metallicFactor   = 1.0f;
    float roughnessFactor  = 1.0f;
    int   _pad0 = 0;
    int   _pad1 = 0;
    int   _pad2 = 0;
};

// One drawable primitive: a slice of the shared index buffer + its material.
struct MeshDraw {
    uint32_t  firstIndex    = 0;
    uint32_t  indexCount    = 0;
    int32_t   vertexOffset  = 0;
    uint32_t  materialIndex = 0;
    glm::mat4 transform     = glm::mat4(1.0f); // node world transform
};

// CPU-side result of loading a glTF. Knows nothing about Vulkan.
struct MeshData {
    std::vector<Vertex>      vertices;
    std::vector<uint32_t>    indices;
    std::vector<GpuMaterial> materials;
    std::vector<MeshDraw>    draws;
    uint32_t                 imageCount = 0; // textures referenced (decoded in Faz 2B)
};
