#pragma once
#include "pch.h"

class VulkanEngine;

struct Image {
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView   view  = VK_NULL_HANDLE;
};

// Create an SRGB RGBA8 sampled texture from CPU pixels (1 mip, no filtering setup).
Image createTextureImage(VulkanEngine& eng, const unsigned char* rgba, int w, int h);

void destroyImage(VmaAllocator alloc, VkDevice device, Image& img);
