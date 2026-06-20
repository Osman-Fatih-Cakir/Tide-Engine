#pragma once
#include "pch.h"

class VulkanEngine;

struct Image {
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView   view  = VK_NULL_HANDLE;
};

// Create an RGBA8 sampled texture from CPU pixels (1 mip, no filtering setup).
// srgb=true for color (baseColor); false for linear data (normal/ORM/metalRough).
Image createTextureImage(VulkanEngine& eng, const unsigned char* rgba, int w, int h,
                         bool srgb);

void destroyImage(VmaAllocator alloc, VkDevice device, Image& img);
