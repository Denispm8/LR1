#pragma once

#include "graphics_internal.hpp"

namespace texture_resources {

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

bool uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation);
bool loadTexture(const char* path, VkFormat format, Texture& texture);
void destroyTexture(Texture& texture);

} // namespace texture_resources
