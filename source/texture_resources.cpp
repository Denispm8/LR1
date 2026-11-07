#include "texture_resources.hpp"

#include <lodepng.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace texture_resources {
namespace {

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    std::cerr << operation << " failed (VkResult " << result << ")\n";
    return false;
}

struct StagingBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;

    ~StagingBuffer() {
        if (buffer) vmaDestroyBuffer(graphics::internal::context.allocator, buffer, allocation);
    }
};

bool createStaging(const void* data, VkDeviceSize size, StagingBuffer& staging) {
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    VmaAllocationInfo mapped{};
    const auto allocator = graphics::internal::context.allocator;
    if (!check(vmaCreateBuffer(allocator, &info, &allocation_info, &staging.buffer,
                              &staging.allocation, &mapped), "vmaCreateBuffer(staging)")) return false;
    std::memcpy(mapped.pMappedData, data, static_cast<size_t>(size));
    return check(vmaFlushAllocation(allocator, staging.allocation, 0, size), "vmaFlushAllocation(staging)");
}

template <typename Record>
bool submitUpload(Record record) {
    const auto& context = graphics::internal::context;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    const VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = context.graphics_queue_index,
    };
    if (!check(vkCreateCommandPool(context.device, &pool_info, nullptr, &pool),
               "vkCreateCommandPool(upload)")) return false;
    const VkCommandBufferAllocateInfo allocate = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    bool success = check(vkAllocateCommandBuffers(context.device, &allocate, &command),
                         "vkAllocateCommandBuffers(upload)");
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (success) success = check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(upload)");
    if (success) {
        record(command);
        success = check(vkEndCommandBuffer(command), "vkEndCommandBuffer(upload)");
    }
    const VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (success) success = check(vkCreateFence(context.device, &fence_info, nullptr, &fence),
                                 "vkCreateFence(upload)");
    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command,
    };
    if (success) success = check(vkQueueSubmit(context.graphics_queue, 1, &submit, fence),
                                 "vkQueueSubmit(upload)");
    if (success && !check(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX),
                          "vkWaitForFences(upload)")) {
        check(vkQueueWaitIdle(context.graphics_queue), "vkQueueWaitIdle(upload)");
        success = false;
    }
    if (fence) vkDestroyFence(context.device, fence, nullptr);
    vkDestroyCommandPool(context.device, pool, nullptr);
    return success;
}

} // namespace

bool uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
    constexpr VkBufferUsageFlags supported = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (!data || size == 0 || size > std::numeric_limits<size_t>::max() || buffer || allocation ||
        (usage & supported) == 0 || (usage & ~supported) != 0) return false;
    const auto& context = graphics::internal::context;
    StagingBuffer staging;
    if (!createStaging(data, size, staging)) return false;
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    if (!check(vmaCreateBuffer(context.allocator, &info, &allocation_info, &buffer,
                              &allocation, nullptr), "vmaCreateBuffer(device local)")) return false;
    const bool uploaded = submitUpload([&](VkCommandBuffer command) {
        const VkBufferCopy copy = {0, 0, size};
        vkCmdCopyBuffer(command, staging.buffer, buffer, 1, &copy);
        const VkBufferMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VkAccessFlags((usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : 0) |
                                          (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT ? VK_ACCESS_INDEX_READ_BIT : 0)),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = 0,
            .size = size,
        };
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    });
    if (!uploaded) {
        vmaDestroyBuffer(context.allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }
    return uploaded;
}

bool loadTexture(const char* path, VkFormat format, Texture& texture) {
    if (!path || texture.image || texture.allocation || texture.view ||
        (format != VK_FORMAT_R8G8B8A8_SRGB && format != VK_FORMAT_R8G8B8A8_UNORM)) return false;
    const auto& context = graphics::internal::context;
    try {
        std::vector<unsigned char> pixels;
        unsigned width = 0, height = 0;
        const unsigned error = lodepng::decode(pixels, width, height, path);
        if (error) {
            std::cerr << "Cannot decode " << path << ": " << lodepng_error_text(error) << '\n';
            return false;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device, &properties);
        if (!width || !height || width > properties.limits.maxImageDimension2D ||
            height > properties.limits.maxImageDimension2D ||
            VkDeviceSize(width) * height * 4 != pixels.size()) {
            std::cerr << "Invalid texture dimensions: " << path << '\n';
            return false;
        }
        VkFormatProperties format_properties{};
        vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &format_properties);
        constexpr VkFormatFeatureFlags features = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((format_properties.optimalTilingFeatures & features) != features) {
            std::cerr << "Unsupported texture format: " << path << '\n';
            return false;
        }
        StagingBuffer staging;
        if (!createStaging(pixels.data(), pixels.size(), staging)) return false;
        const VkImageCreateInfo image = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        const VmaAllocationCreateInfo allocation_info = {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        if (!check(vmaCreateImage(context.allocator, &image, &allocation_info, &texture.image,
                                  &texture.allocation, nullptr), "vmaCreateImage(texture)")) return false;
        const bool uploaded = submitUpload([&](VkCommandBuffer command) {
            VkImageMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = texture.image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
            const VkBufferImageCopy copy = {
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {width, height, 1},
            };
            vkCmdCopyBufferToImage(command, staging.buffer, texture.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
        });
        if (!uploaded) { destroyTexture(texture); return false; }
        const VkImageViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = texture.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        if (!check(vkCreateImageView(context.device, &view, nullptr, &texture.view),
                   "vkCreateImageView(texture)")) { destroyTexture(texture); return false; }
        return true;
    } catch (const std::exception& error) {
        std::cerr << path << ": " << error.what() << '\n';
        destroyTexture(texture);
        return false;
    }
}

void destroyTexture(Texture& texture) {
    const auto& context = graphics::internal::context;
    if (texture.view) vkDestroyImageView(context.device, texture.view, nullptr);
    if (texture.image) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    texture = {};
}

} // namespace texture_resources
