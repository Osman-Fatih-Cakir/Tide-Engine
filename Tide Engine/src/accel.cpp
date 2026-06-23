#include "accel.h"
#include "vk_engine.h"
#include "scene.h"
#include "mesh.h"
#include <vector>
#include <cstring>

// KHR accel functions aren't exported by the loader; fetch them once.
namespace {
PFN_vkGetAccelerationStructureBuildSizesKHR    pGetSizes   = nullptr;
PFN_vkCreateAccelerationStructureKHR           pCreate     = nullptr;
PFN_vkCmdBuildAccelerationStructuresKHR        pCmdBuild   = nullptr;
PFN_vkGetAccelerationStructureDeviceAddressKHR pGetAddr    = nullptr;
PFN_vkDestroyAccelerationStructureKHR          pDestroy    = nullptr;

void loadFns(VkDevice device) {
    if (pCreate) return;
    pGetSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    pCreate   = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    pCmdBuild = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
    pGetAddr  = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    pDestroy  = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
}

Buffer makeAccelBuffer(VmaAllocator alloc, VkDeviceSize size, VkBufferUsageFlags usage) {
    return createBuffer(alloc, size,
                        usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
}
} // namespace

void SceneAccel::destroy(VkDevice device, VmaAllocator alloc) {
    if (pDestroy) {
        if (tlas.handle) pDestroy(device, tlas.handle, nullptr);
        for (auto& b : blas) if (b.handle) pDestroy(device, b.handle, nullptr);
    }
    destroyBuffer(alloc, tlas.buffer);
    for (auto& b : blas) destroyBuffer(alloc, b.buffer);
    destroyBuffer(alloc, instanceBuffer);
    blas.clear();
    tlas = {};
}

void buildSceneAccel(VulkanEngine& eng, Scene& scene) {
    VkDevice device = eng.device();
    VmaAllocator alloc = eng.allocator();
    loadFns(device);

    // One BLAS per unique geometry (shared across instances). TLAS instances are
    // built only for opaque instances — transparent (glass) casts no shadow, so
    // it's excluded; no instance masks, no alpha test, all occluders opaque.
    const uint32_t geoCount = scene.geometryCount;
    if (geoCount == 0 || !scene.vertexBuffer.buffer || !scene.indexBuffer.buffer) return;

    VkDeviceAddress vtxAddr = bufferAddress(device, scene.vertexBuffer.buffer);
    VkDeviceAddress idxAddr = bufferAddress(device, scene.indexBuffer.buffer);

    SceneAccel& out = scene.accel;
    out.blas.resize(geoCount);

    // ----- BLAS: one per geometry, slicing the shared vertex/index buffers -----
    std::vector<Buffer> scratch; // kept alive until the submit completes
    scratch.reserve(geoCount + 1);

    eng.immediateSubmit([&](VkCommandBuffer cmd) {
        for (uint32_t k = 0; k < geoCount; k++) {
            const Geometry& d = scene.geometries[k];

            VkAccelerationStructureGeometryKHR geo{};
            geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR; // all opaque -> ray query auto-commits
            auto& tri = geo.geometry.triangles;
            tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            tri.vertexData.deviceAddress = vtxAddr;
            tri.vertexStride = sizeof(Vertex);
            // Safe in-bounds upper bound: vertexOffset + maxVertex < vertexCount.
            tri.maxVertex = (scene.vertexCount > (uint32_t)d.vertexOffset + 1)
                          ? scene.vertexCount - 1 - (uint32_t)d.vertexOffset : 0;
            tri.indexType = VK_INDEX_TYPE_UINT32;
            tri.indexData.deviceAddress = idxAddr;
            tri.transformData.deviceAddress = 0;

            VkAccelerationStructureBuildGeometryInfoKHR bi{};
            bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            bi.geometryCount = 1;
            bi.pGeometries = &geo;

            uint32_t primCount = d.indexCount / 3u;
            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            pGetSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                      &bi, &primCount, &sizes);

            out.blas[k].buffer = makeAccelBuffer(alloc, sizes.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
            VkAccelerationStructureCreateInfoKHR ci{};
            ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            ci.buffer = out.blas[k].buffer.buffer;
            ci.size = sizes.accelerationStructureSize;
            ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            VK_CHECK(pCreate(device, &ci, nullptr, &out.blas[k].handle));

            Buffer s = makeAccelBuffer(alloc, sizes.buildScratchSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            bi.dstAccelerationStructure = out.blas[k].handle;
            bi.scratchData.deviceAddress = bufferAddress(device, s.buffer);
            scratch.push_back(s);

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount  = primCount;
            range.primitiveOffset = d.firstIndex * sizeof(uint32_t); // byte offset into index buf
            range.firstVertex     = (uint32_t)d.vertexOffset;        // added to each index
            range.transformOffset = 0;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            pCmdBuild(cmd, 1, &bi, &pRange);

            VkAccelerationStructureDeviceAddressInfoKHR ai{};
            ai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            ai.accelerationStructure = out.blas[k].handle;
            out.blas[k].address = pGetAddr(device, &ai);
        }

        // BLAS writes -> TLAS reads.
        VkMemoryBarrier2 mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);

        // ----- TLAS instances (one per opaque scene instance, sharing BLAS) -----
        const std::vector<uint32_t>& opaque = scene.opaqueInstances;
        const uint32_t instCount = (uint32_t)opaque.size();
        std::vector<VkAccelerationStructureInstanceKHR> instances(instCount);
        for (uint32_t k = 0; k < instCount; k++) {
            const MeshInstance& si = scene.instances[opaque[k]];
            const glm::mat4& m = si.transform; // column-major
            VkAccelerationStructureInstanceKHR inst{};
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    inst.transform.matrix[r][c] = m[c][r]; // -> row-major 3x4
            inst.instanceCustomIndex = opaque[k];  // scene instance index (DDGI hit lookup)
            inst.mask = 0xFFu;
            inst.instanceShaderBindingTableRecordOffset = 0;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = out.blas[si.geometryID].address; // shared BLAS
            instances[k] = inst;
        }
        VkDeviceSize instBytes = sizeof(VkAccelerationStructureInstanceKHR) * instCount;
        out.instanceBuffer = createBuffer(alloc, instBytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
        memcpy(out.instanceBuffer.mapped, instances.data(), instBytes);

        // Instance data must be visible to the TLAS build (host write -> AS build read).
        VkMemoryBarrier2 hb{};
        hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        hb.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        hb.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
        hb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        hb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo hdep{};
        hdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        hdep.memoryBarrierCount = 1;
        hdep.pMemoryBarriers = &hb;
        vkCmdPipelineBarrier2(cmd, &hdep);

        VkAccelerationStructureGeometryKHR geo{};
        geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geo.geometry.instances.arrayOfPointers = VK_FALSE;
        geo.geometry.instances.data.deviceAddress = bufferAddress(device, out.instanceBuffer.buffer);

        VkAccelerationStructureBuildGeometryInfoKHR bi{};
        bi.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &geo;

        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        pGetSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                  &bi, &instCount, &sizes);

        out.tlas.buffer = makeAccelBuffer(alloc, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
        VkAccelerationStructureCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        ci.buffer = out.tlas.buffer.buffer;
        ci.size = sizes.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(pCreate(device, &ci, nullptr, &out.tlas.handle));

        Buffer s = makeAccelBuffer(alloc, sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        bi.dstAccelerationStructure = out.tlas.handle;
        bi.scratchData.deviceAddress = bufferAddress(device, s.buffer);
        scratch.push_back(s);

        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = instCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
        pCmdBuild(cmd, 1, &bi, &pRange);
    });

    for (auto& s : scratch) destroyBuffer(alloc, s);

    TE_INFO("Accel: %u BLAS (per geometry) + 1 TLAS (%u opaque instances)\n",
            geoCount, (uint32_t)scene.opaqueInstances.size());
}
