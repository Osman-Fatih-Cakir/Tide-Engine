#pragma once
#include "pch.h"

// A GPU buffer + its VMA allocation. Plain data holder; no ownership magic.
struct Buffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void*         mapped     = nullptr; // non-null only for host-visible mapped buffers
    VkDeviceSize  size       = 0;
};

// Create a buffer. Pass allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
// | VMA_ALLOCATION_CREATE_MAPPED_BIT for a CPU-writable staging buffer.
inline Buffer createBuffer(VmaAllocator alloc, VkDeviceSize size,
                           VkBufferUsageFlags usage,
                           VmaMemoryUsage memUsage,
                           VmaAllocationCreateFlags allocFlags = 0) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;

    VmaAllocationCreateInfo ai{};
    ai.usage = memUsage;
    ai.flags = allocFlags;

    Buffer out{};
    out.size = size;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(alloc, &bi, &ai, &out.buffer, &out.allocation, &info));
    if (allocFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT) out.mapped = info.pMappedData;
    return out;
}

inline void destroyBuffer(VmaAllocator alloc, Buffer& b) {
    if (b.buffer) vmaDestroyBuffer(alloc, b.buffer, b.allocation);
    b = {};
}

inline VkDeviceAddress bufferAddress(VkDevice device, VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}
