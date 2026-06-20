#pragma once
#include "pch.h"
#include <limits>

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
    int   alphaMode   = 0;     // 0 = OPAQUE, 1 = MASK, 2 = BLEND
    float alphaCutoff = 0.5f;
    int   _pad = 0;
};

// glTF alpha modes (matches GpuMaterial::alphaMode).
enum AlphaMode { ALPHA_OPAQUE = 0, ALPHA_MASK = 1, ALPHA_BLEND = 2 };

// Per-draw record as it lives in the draw SSBO (read by the resolve compute to
// reconstruct a triangle from a packed visibility ID). scalar layout in GLSL.
struct GpuDraw {
    glm::mat4 transform;        // node world transform
    uint32_t  firstIndex   = 0; // offset into the shared index buffer
    uint32_t  vertexOffset = 0; // added to each fetched index
    uint32_t  materialIndex = 0;
    uint32_t  _pad = 0;
};

// Decoded texture (RGBA8) ready for GPU upload.
struct TextureData {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
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
    std::vector<TextureData> textures;   // indexed by glTF image source index

    // World-space AABB (for auto camera placement).
    glm::vec3 boundsMin = glm::vec3( std::numeric_limits<float>::max());
    glm::vec3 boundsMax = glm::vec3(-std::numeric_limits<float>::max());
};
