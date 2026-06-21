#pragma once
#include "pch.h"
#include <limits>

// Interleaved vertex. Tangent comes from the glTF (Blender export); .w = handedness.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent = glm::vec4(1, 0, 0, 1);
};

// Material as it will live in the bindless SSBO. Texture fields are indices into
// the bindless texture array (-1 = none). 16-byte aligned (scalar/std430 in GLSL).
struct GpuMaterial {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    int   baseColorTexture  = -1;
    int   normalTexture     = -1;
    int   metalRoughTexture = -1;
    int   occlusionTexture  = -1;   // glTF occlusion (R channel = AO)
    float metallicFactor    = 1.0f;
    float roughnessFactor   = 1.0f;
    float occlusionStrength = 1.0f;
    float alphaCutoff       = 0.5f;
    int   alphaMode = 0;            // 0 = OPAQUE, 1 = MASK, 2 = BLEND
    int   _pad0 = 0;
    int   _pad1 = 0;
    int   _pad2 = 0;
};

// glTF alpha modes (matches GpuMaterial::alphaMode).
enum AlphaMode { ALPHA_OPAQUE = 0, ALPHA_MASK = 1, ALPHA_BLEND = 2 };

// Unique mesh-primitive geometry (deduplicated across glTF node references).
// Lives once in the shared vertex/index buffers; many instances reference it.
// GPU mirror (geometry SSBO, std430, 16 bytes) — indexCount isn't needed by the
// resolve shader so it's CPU-only.
struct GpuGeometry {
    uint32_t firstIndex    = 0; // offset into the shared index buffer
    uint32_t vertexOffset  = 0; // added to each fetched index
    uint32_t materialIndex = 0;
    uint32_t _pad          = 0;
};
struct Geometry {               // CPU-side: GpuGeometry + the draw-call index count
    uint32_t firstIndex    = 0;
    uint32_t indexCount    = 0;
    int32_t  vertexOffset  = 0;
    uint32_t materialIndex = 0;
};

// One placement of a geometry (a glTF node referencing a mesh). GPU mirror
// (instance SSBO, scalar layout) reconstructs world position + finds the geometry.
struct GpuInstance {
    glm::mat4 transform;        // node world transform
    uint32_t  geometryID = 0;
    uint32_t  _pad0 = 0, _pad1 = 0, _pad2 = 0;
};
struct MeshInstance {           // CPU-side
    glm::mat4 transform = glm::mat4(1.0f);
    uint32_t  geometryID = 0;
};

// Decoded texture (RGBA8) ready for GPU upload.
struct TextureData {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;
};

// CPU-side result of loading a glTF. Knows nothing about Vulkan. Geometry is
// deduplicated: each unique (mesh,primitive) appears once in `geometries` (and
// once in the vertex/index buffers); `instances` place them with transforms.
struct MeshData {
    std::vector<Vertex>       vertices;
    std::vector<uint32_t>     indices;
    std::vector<GpuMaterial>  materials;
    std::vector<Geometry>     geometries;
    std::vector<MeshInstance> instances;
    std::vector<TextureData>  textures;   // indexed by glTF image source index

    // World-space AABB (for auto camera placement).
    glm::vec3 boundsMin = glm::vec3( std::numeric_limits<float>::max());
    glm::vec3 boundsMax = glm::vec3(-std::numeric_limits<float>::max());
};
