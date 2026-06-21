#include "gpu_image.h"
#include "gpu_buffer.h"
#include "vk_engine.h"
#include <cmath>

Image createTextureImage(VulkanEngine& eng, const unsigned char* rgba, int w, int h,
                         bool srgb) {
    VmaAllocator alloc = eng.allocator();
    VkDevice device = eng.device();
    VkDeviceSize size = (VkDeviceSize)w * h * 4;
    const VkFormat format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;

    // Staging.
    Buffer staging = createBuffer(alloc, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                  VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(staging.mapped, rgba, size);

    // Device image.
    Image out{};
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {(uint32_t)w, (uint32_t)h, 1};
    // Full mip chain, generated on the GPU by successive blits.
    const uint32_t mipLevels =
        (uint32_t)std::floor(std::log2((float)(w > h ? w : h))) + 1u;
    ici.mipLevels = mipLevels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC needed: each mip is the blit source for the next.
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    VK_CHECK(vmaCreateImage(alloc, &ici, &ai, &out.image, &out.alloc, nullptr));

    eng.immediateSubmit([&](VkCommandBuffer cmd) {
        auto barrier = [&](uint32_t mip, VkImageLayout oldL, VkImageLayout newL,
                           VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                           VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
            VkImageMemoryBarrier2 b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = srcS; b.srcAccessMask = srcA;
            b.dstStageMask = dstS; b.dstAccessMask = dstA;
            b.oldLayout = oldL; b.newLayout = newL;
            b.image = out.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        };

        // Transition the WHOLE mip chain UNDEFINED -> TRANSFER_DST. Each level i>0
        // is a blit destination, so it must already be in TRANSFER_DST (not just
        // mip 0); otherwise the blit dst sits in UNDEFINED.
        {
            VkImageMemoryBarrier2 b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.image = out.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        // Mip 0: upload from staging.
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
        vkCmdCopyBufferToImage(cmd, staging.buffer, out.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // Generate the mip chain by blitting each level down to the next.
        int32_t mw = w, mh = h;
        for (uint32_t i = 1; i < mipLevels; i++) {
            // Source mip (i-1): DST -> SRC.
            barrier(i - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

            int32_t nw = mw > 1 ? mw / 2 : 1;
            int32_t nh = mh > 1 ? mh / 2 : 1;
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
            blit.srcOffsets[1] = {mw, mh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
            blit.dstOffsets[1] = {nw, nh, 1};
            vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);

            // Source mip (i-1) is done: SRC -> SHADER_READ.
            barrier(i - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            mw = nw; mh = nh;
        }
        // Last mip is still DST -> SHADER_READ.
        barrier(mipLevels - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    });

    destroyBuffer(alloc, staging);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = out.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1};
    VK_CHECK(vkCreateImageView(device, &vci, nullptr, &out.view));

    return out;
}

void destroyImage(VmaAllocator alloc, VkDevice device, Image& img) {
    if (img.view) vkDestroyImageView(device, img.view, nullptr);
    if (img.image) vmaDestroyImage(alloc, img.image, img.alloc);
    img = {};
}
