#include "gltf_loader.h"

// tinygltf declarations only (implementation is in gltf_impl.cpp).
#include "tiny_gltf.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>
#include <limits>

// --- node local transform ---
static glm::mat4 nodeLocal(const tinygltf::Node& n) {
    if (n.matrix.size() == 16) {
        glm::mat4 m(1.0f);
        for (int i = 0; i < 16; i++) glm::value_ptr(m)[i] = (float)n.matrix[i];
        return m;
    }
    glm::mat4 t(1.0f), r(1.0f), s(1.0f);
    if (n.translation.size() == 3)
        t = glm::translate(glm::mat4(1.0f),
                           glm::vec3(n.translation[0], n.translation[1], n.translation[2]));
    if (n.rotation.size() == 4) {
        glm::quat q((float)n.rotation[3], (float)n.rotation[0],
                    (float)n.rotation[1], (float)n.rotation[2]);
        r = glm::mat4_cast(q);
    }
    if (n.scale.size() == 3)
        s = glm::scale(glm::mat4(1.0f),
                       glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
    return t * r * s;
}

// --- raw accessor access ---
static const unsigned char* accessorPtr(const tinygltf::Model& m,
                                        const tinygltf::Accessor& acc,
                                        size_t& stride) {
    const tinygltf::BufferView& view = m.bufferViews[acc.bufferView];
    const tinygltf::Buffer& buf = m.buffers[view.buffer];
    stride = acc.ByteStride(view);
    return buf.data.data() + view.byteOffset + acc.byteOffset;
}

// Per-geometry local-space AABB, used to expand world bounds per instance
// without re-touching the (shared) vertex data.
struct GeoCache {
    std::map<uint64_t, uint32_t> ids; // (meshIdx<<32 | primIdx) -> geometryID
    std::vector<glm::vec3> localMin, localMax;
};

// Extract a unique (mesh,primitive) geometry once; later references reuse it.
// Returns geometryID, or -1 if the primitive has no indices/positions.
static int getOrAddGeometry(const tinygltf::Model& model, const tinygltf::Primitive& prim,
                            int meshIdx, int primIdx, MeshData& out, GeoCache& cache) {
    uint64_t key = ((uint64_t)(uint32_t)meshIdx << 32) | (uint32_t)primIdx;
    auto it = cache.ids.find(key);
    if (it != cache.ids.end()) return (int)it->second;

    if (prim.indices < 0) return -1;
    auto itPos = prim.attributes.find("POSITION");
    if (itPos == prim.attributes.end()) return -1;

    const tinygltf::Accessor& posAcc = model.accessors[itPos->second];
    size_t posStride = 0;
    const unsigned char* posBase = accessorPtr(model, posAcc, posStride);

    const unsigned char* nrmBase = nullptr; size_t nrmStride = 0;
    auto itN = prim.attributes.find("NORMAL");
    if (itN != prim.attributes.end())
        nrmBase = accessorPtr(model, model.accessors[itN->second], nrmStride);

    const unsigned char* uvBase = nullptr; size_t uvStride = 0;
    auto itUV = prim.attributes.find("TEXCOORD_0");
    if (itUV != prim.attributes.end())
        uvBase = accessorPtr(model, model.accessors[itUV->second], uvStride);

    const unsigned char* tanBase = nullptr; size_t tanStride = 0;
    auto itT = prim.attributes.find("TANGENT");
    if (itT != prim.attributes.end())
        tanBase = accessorPtr(model, model.accessors[itT->second], tanStride);

    uint32_t baseVertex = (uint32_t)out.vertices.size();
    glm::vec3 lmin( std::numeric_limits<float>::max());
    glm::vec3 lmax(-std::numeric_limits<float>::max());
    out.vertices.reserve(out.vertices.size() + posAcc.count);
    for (size_t i = 0; i < posAcc.count; i++) {
        Vertex v{};
        const float* p = (const float*)(posBase + i * posStride);
        v.position = glm::vec3(p[0], p[1], p[2]);
        lmin = glm::min(lmin, v.position);
        lmax = glm::max(lmax, v.position);
        if (nrmBase) {
            const float* n = (const float*)(nrmBase + i * nrmStride);
            v.normal = glm::vec3(n[0], n[1], n[2]);
        }
        if (uvBase) {
            const float* uv = (const float*)(uvBase + i * uvStride);
            v.uv = glm::vec2(uv[0], uv[1]);
        }
        if (tanBase) {
            const float* t = (const float*)(tanBase + i * tanStride);
            v.tangent = glm::vec4(t[0], t[1], t[2], t[3]); // glTF: xyz dir, w handedness
        }
        out.vertices.push_back(v);
    }

    // indices (component type may be u8/u16/u32; stored local to this primitive)
    const tinygltf::Accessor& idxAcc = model.accessors[prim.indices];
    size_t idxStride = 0;
    const unsigned char* idxBase = accessorPtr(model, idxAcc, idxStride);

    uint32_t firstIndex = (uint32_t)out.indices.size();
    out.indices.reserve(out.indices.size() + idxAcc.count);
    for (size_t i = 0; i < idxAcc.count; i++) {
        const unsigned char* e = idxBase + i * idxStride;
        uint32_t index = 0;
        switch (idxAcc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  index = *(const uint8_t*)e;  break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: index = *(const uint16_t*)e; break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   index = *(const uint32_t*)e; break;
            default: break;
        }
        out.indices.push_back(index);
    }

    Geometry geo{};
    geo.firstIndex = firstIndex;
    geo.indexCount = (uint32_t)idxAcc.count;
    geo.vertexOffset = (int32_t)baseVertex;
    geo.materialIndex = prim.material >= 0 ? (uint32_t)prim.material : 0;
    uint32_t id = (uint32_t)out.geometries.size();
    out.geometries.push_back(geo);
    cache.ids[key] = id;
    cache.localMin.push_back(lmin);
    cache.localMax.push_back(lmax);
    return (int)id;
}

static void processNode(const tinygltf::Model& model, int nodeIdx,
                        const glm::mat4& parent, MeshData& out, GeoCache& cache) {
    const tinygltf::Node& node = model.nodes[nodeIdx];
    glm::mat4 world = parent * nodeLocal(node);
    if (node.mesh >= 0) {
        const auto& prims = model.meshes[node.mesh].primitives;
        for (int p = 0; p < (int)prims.size(); p++) {
            int gid = getOrAddGeometry(model, prims[p], node.mesh, p, out, cache);
            if (gid < 0) continue;
            MeshInstance inst{};
            inst.transform = world;
            inst.geometryID = (uint32_t)gid;
            out.instances.push_back(inst);
            // Expand world bounds from the geometry's 8 local AABB corners.
            glm::vec3 mn = cache.localMin[gid], mx = cache.localMax[gid];
            for (int c = 0; c < 8; c++) {
                glm::vec3 corner((c & 1) ? mx.x : mn.x,
                                 (c & 2) ? mx.y : mn.y,
                                 (c & 4) ? mx.z : mn.z);
                glm::vec3 wp = glm::vec3(world * glm::vec4(corner, 1.0f));
                out.boundsMin = glm::min(out.boundsMin, wp);
                out.boundsMax = glm::max(out.boundsMax, wp);
            }
        }
    }
    for (int child : node.children)
        processNode(model, child, world, out, cache);
}

bool loadGltf(const char* path, MeshData& out) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!warn.empty()) TE_WARN("glTF: %s\n", warn.c_str());
    if (!err.empty())  TE_ERROR("glTF: %s\n", err.c_str());
    if (!ok) { TE_ERROR("glTF: failed to load %s\n", path); return false; }

    // Materials (default one if the file has none).
    if (model.materials.empty()) {
        out.materials.push_back(GpuMaterial{});
    } else {
        for (const auto& m : model.materials) {
            GpuMaterial gm{};
            const auto& pbr = m.pbrMetallicRoughness;
            if (pbr.baseColorFactor.size() == 4)
                gm.baseColorFactor = glm::vec4(pbr.baseColorFactor[0], pbr.baseColorFactor[1],
                                               pbr.baseColorFactor[2], pbr.baseColorFactor[3]);
            gm.metallicFactor  = (float)pbr.metallicFactor;
            gm.roughnessFactor = (float)pbr.roughnessFactor;
            // Store the source IMAGE index; matches the bindless texture array slot.
            auto srcImage = [&](int texIdx) -> int {
                if (texIdx < 0) return -1;
                return model.textures[texIdx].source;
            };
            gm.baseColorTexture  = srcImage(pbr.baseColorTexture.index);
            gm.metalRoughTexture = srcImage(pbr.metallicRoughnessTexture.index);
            gm.normalTexture     = srcImage(m.normalTexture.index);
            gm.occlusionTexture  = srcImage(m.occlusionTexture.index);
            gm.occlusionStrength = (float)m.occlusionTexture.strength;
            // Alpha mode: OPAQUE (default) / MASK / BLEND.
            if (m.alphaMode == "BLEND")     gm.alphaMode = ALPHA_BLEND;
            else if (m.alphaMode == "MASK") gm.alphaMode = ALPHA_MASK;
            else                            gm.alphaMode = ALPHA_OPAQUE;
            gm.alphaCutoff = (float)m.alphaCutoff;
            // Emissive (KHR core): factor (linear) and optional sRGB map.
            if (m.emissiveFactor.size() == 3)
                gm.emissiveFactor = glm::vec4(m.emissiveFactor[0], m.emissiveFactor[1],
                                              m.emissiveFactor[2], 0.0f);
            gm.emissiveTexture = srcImage(m.emissiveTexture.index);
            out.materials.push_back(gm);
        }
    }

    // Walk the default scene's node hierarchy.
    GeoCache cache;
    int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (!model.scenes.empty()) {
        for (int nodeIdx : model.scenes[sceneIdx].nodes)
            processNode(model, nodeIdx, glm::mat4(1.0f), out, cache);
    } else {
        // No scene graph: just take every node.
        for (int i = 0; i < (int)model.nodes.size(); i++)
            processNode(model, i, glm::mat4(1.0f), out, cache);
    }
    TE_INFO("glTF: %zu unique geometries, %zu instances\n",
            out.geometries.size(), out.instances.size());

    // Textures: tinygltf already decoded images into model.images. Convert to RGBA8.
    out.textures.reserve(model.images.size());
    for (const auto& img : model.images) {
        TextureData td;
        td.width = img.width;
        td.height = img.height;
        td.rgba.resize((size_t)img.width * img.height * 4, 255);
        const int comp = img.component;
        for (size_t px = 0; px < (size_t)img.width * img.height; px++) {
            for (int c = 0; c < 4; c++) {
                if (c < comp) td.rgba[px * 4 + c] = img.image[px * comp + c];
                else if (c == 3) td.rgba[px * 4 + c] = 255; // opaque alpha
                else td.rgba[px * 4 + c] = (comp == 1) ? img.image[px * comp] : 0;
            }
        }
        out.textures.push_back(std::move(td));
    }

    return true;
}
