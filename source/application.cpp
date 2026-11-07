#define GLM_FORCE_DEPTH_ZERO_TO_ONE 

#include "application.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <filesystem>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "graphics_internal.hpp"
#include "texture_resources.hpp"

namespace {

    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    VkShaderModule loadShaderModule(const char* path) {
        std::string final_path = path;
        std::ifstream file(final_path, std::ios::binary | std::ios::ate);

        if (!file) {
            final_path = std::string("../") + path;
            file.open(final_path, std::ios::binary | std::ios::ate);
        }
        if (!file) {
            final_path = std::string("build-debug/") + path;
            file.open(final_path, std::ios::binary | std::ios::ate);
        }

        if (!file) {
            std::string cur_dir = std::filesystem::current_path().string();
            std::string msg = "Cannot open shader: " + std::string(path) +
                "\nCurrent working directory is:\n" + cur_dir +
                "\n\nChecked paths:\n" + std::string(path) + "\n../" + std::string(path);
            MessageBoxA(nullptr, msg.c_str(), "Shader Load Error", MB_ICONERROR | MB_OK);
            return VK_NULL_HANDLE;
        }

        const size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(size);

        file.seekg(0);
        file.read(buffer.data(), size);
        file.close();

        const VkShaderModuleCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = reinterpret_cast<const uint32_t*>(buffer.data()),
        };

        VkShaderModule result = VK_NULL_HANDLE;
        if (vkCreateShaderModule(graphics::internal::context.device,
            &info, nullptr, &result) != VK_SUCCESS) {
            std::string msg = "Failed to create VkShaderModule from: " + final_path;
            MessageBoxA(nullptr, msg.c_str(), "Shader Creation Error", MB_ICONERROR | MB_OK);
            return VK_NULL_HANDLE;
        }
        return result;
    }

} // namespace

namespace application {

    namespace {

        constexpr uint32_t MAX_LIGHTS = 32;
        constexpr uint32_t cube_count = 3;
        constexpr uint32_t material_count = 3;
        constexpr uint32_t model_capacity = cube_count + 4 * MAX_LIGHTS;

        const char* material_names[] = { "Chain portrait", "Wide portrait", "Suit portrait" };
        texture_resources::Texture albedo_textures[material_count];
        texture_resources::Texture specular_texture, black_texture;
        VkSampler vk_texture_sampler = VK_NULL_HANDLE;

        float controlsWidth() {
            return std::min(390.0f, ImGui::GetIO().DisplaySize.x * 0.42f);
        }

        uint32_t sceneWidth() {
            const auto width = graphics::internal::context.swapchain_extent.width;
            const float display_width = std::max(ImGui::GetIO().DisplaySize.x, 1.0f);
            return std::max(1u, uint32_t(width * (display_width - controlsWidth()) / display_width));
        }

        std::vector<Vertex> generateCubeVertices() {
            std::vector<Vertex> v;
            v.reserve(24);

            auto addFace = [&](glm::vec3 normal, glm::vec3 right, glm::vec3 up) {
                glm::vec3 p0 = normal - right - up;
                glm::vec3 p1 = normal + right - up;
                glm::vec3 p2 = normal + right + up;
                glm::vec3 p3 = normal - right + up;

                v.push_back({ p0, normal, { 0.0f, 1.0f } });
                v.push_back({ p1, normal, { 1.0f, 1.0f } });
                v.push_back({ p2, normal, { 1.0f, 0.0f } });
                v.push_back({ p3, normal, { 0.0f, 0.0f } });
                };

            addFace({ 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 });
            addFace({ 0, 0,-1 }, { -1, 0, 0 }, { 0, 1, 0 });
            addFace({ 1, 0, 0 }, { 0, 0,-1 }, { 0, 1, 0 });
            addFace({ -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 });
            addFace({ 0, 1, 0 }, { 1, 0, 0 }, { 0, 0,-1 });
            addFace({ 0,-1, 0 }, { 1, 0, 0 }, { 0, 0, 1 });

            assert(v.size() == 24);

            return v;
        }

        std::vector<uint32_t> generateCubeIndices() {
            std::vector<uint32_t> ind;
            ind.reserve(36);
            for (uint32_t i = 0; i < 6; ++i) {
                uint32_t base = i * 4;
                ind.push_back(base + 0);
                ind.push_back(base + 1);
                ind.push_back(base + 2);
                ind.push_back(base + 0);
                ind.push_back(base + 2);
                ind.push_back(base + 3);
            }
            return ind;
        }

        VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

        VkBuffer vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;
        uint32_t cube_index_count = 36;

        VkBuffer vk_cone_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_cone_vertex_buffer_allocation = VK_NULL_HANDLE;
        uint32_t cone_vertex_count = 0;

        struct SceneUniforms {
            glm::mat4 view_proj;
            glm::vec4 camera_position;
            glm::vec4 directional_light_direction;
            glm::vec4 directional_light_color;
            glm::vec4 ambient_light;
            uint32_t point_light_count;
            uint32_t spot_light_count;
            float _pad[2];
        };

        struct GPUSpotLight {
            glm::vec4 position;
            glm::vec4 direction;
            glm::vec4 color;
        };

        struct GPUPointLight {
            glm::vec4 position;
            glm::vec4 color;
        };

        struct ModelUniform {
            glm::mat4 model;
            glm::vec4 albedo;
            glm::vec4 specular;
            glm::vec4 emission;
            glm::ivec4 maps;
        };

        struct TexturePushConstants {
            float time;
            float strength;
            float color_modulation;
            int32_t effect;
        };
        static_assert(sizeof(ModelUniform) == 128);
        static_assert(offsetof(ModelUniform, emission) == 96);
        static_assert(offsetof(ModelUniform, maps) == 112);
        static_assert(sizeof(TexturePushConstants) == 16);
        static_assert(offsetof(TexturePushConstants, effect) == 12);

        struct PointLightData {
            glm::vec3 position{ 0.0f, 2.5f, 0.0f };
            glm::vec3 color{ 1.0f, 0.9f, 0.7f };
            float intensity = 6.0f;
        };

        struct SpotLightData {
            glm::vec3 position{ 0.0f, 3.5f, 2.0f };
            glm::vec3 direction{ 0.0f, -1.0f, -0.6f };
            glm::vec3 color{ 0.3f, 0.7f, 1.0f };
            float intensity = 15.0f;
            float inner_angle_deg = 15.0f;
            float outer_angle_deg = 25.0f;
        };

        void clampSpotAngles(SpotLightData& light) {
            light.outer_angle_deg = std::clamp(light.outer_angle_deg, 1.1f, 80.0f);
            light.inner_angle_deg = std::clamp(light.inner_angle_deg, 1.0f,
                std::min(60.0f, light.outer_angle_deg - 0.1f));
        }

        std::vector<PointLightData> point_lights;
        std::vector<SpotLightData> spot_lights;

        VkBuffer vk_scene_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_scene_uniform_buffer_allocation = VK_NULL_HANDLE;
        SceneUniforms* vk_scene_uniform_buffer_mapped = nullptr;

        VkBuffer vk_point_ssbo = VK_NULL_HANDLE;
        VmaAllocation vk_point_ssbo_allocation = VK_NULL_HANDLE;
        GPUPointLight* vk_point_ssbo_mapped = nullptr;

        VkBuffer vk_spot_ssbo = VK_NULL_HANDLE;
        VmaAllocation vk_spot_ssbo_allocation = VK_NULL_HANDLE;
        GPUSpotLight* vk_spot_ssbo_mapped = nullptr;

        VkBuffer vk_model_uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation vk_model_uniform_buffer_allocation = VK_NULL_HANDLE;
        std::byte* vk_model_uniform_buffer_mapped = nullptr;
        VkDeviceSize model_uniform_stride = 0;

        VkDescriptorSetLayout vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;

        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;
        VkDescriptorSet vk_material_descriptor_sets[material_count] = {};

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline vk_pipeline = VK_NULL_HANDLE;

        VkPipeline vk_unlit_pipeline = VK_NULL_HANDLE;
        VkPipeline vk_unlit_lines_pipeline = VK_NULL_HANDLE;

        const glm::vec3 cube_offsets[cube_count] = {
            { -3.0f, 0.0f, 0.0f },
            {  0.0f, 0.0f, 0.0f },
            {  3.0f, 0.0f, 0.0f },
        };

        struct CubeSettings {
            glm::vec3 position{ 0.0f };
            glm::vec3 rotationDeg{ 0.0f };
            glm::vec3 scale{ 0.7f };
            glm::vec3 albedo{ 1.0f };
            glm::vec3 specular{ 1.0f, 1.0f, 1.0f };
            float shininess = 32.0f;
            bool animate = true;
            float phaseOffset = 0.0f;
            int material = 0;
            int effect = 0;
            float effect_strength = 0.35f;
            float color_modulation = 0.25f;
            bool specular_map = true;
            bool emissive_map = true;
            glm::vec3 emission_tint{ 1.0f };
            float emission_strength = 0.0f;
        };

        CubeSettings cube_settings[cube_count] = {
            { glm::vec3(0.0f), glm::vec3(-8.0f, 18.0f, 0.0f), glm::vec3(0.9f), glm::vec3(1.0f), glm::vec3(0.35f), 32.0f,  true, 0.0f, 0 },
            { glm::vec3(0.0f), glm::vec3(-8.0f, -12.0f, 0.0f), glm::vec3(0.9f), glm::vec3(1.0f), glm::vec3(0.35f), 64.0f,  true, 2.094f, 1 },
            { glm::vec3(0.0f), glm::vec3(-8.0f, -22.0f, 0.0f), glm::vec3(0.9f), glm::vec3(1.0f), glm::vec3(0.35f), 128.0f, true, 4.188f, 2 }
        };

        struct Camera {
            glm::vec3 position{ 0.0f, 2.0f, 9.0f };
            float yaw = -90.0f;
            float pitch = -10.0f;
            float fov = 45.0f;
        } camera;

        glm::vec3 dir_light_direction = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));
        glm::vec3 dir_light_color = glm::vec3(0.9f, 0.9f, 0.9f);
        float dir_light_intensity = 1.0f;
        glm::vec3 ambient_light_color = glm::vec3(0.22f, 0.22f, 0.25f);
        bool show_light_visualizers = false;
        bool animate_texture = true;
        float texture_time = 0.0f;

        bool createBuffers() {
            auto vertices = generateCubeVertices();
            auto indices = generateCubeIndices();
            cube_index_count = static_cast<uint32_t>(indices.size());

            if (!texture_resources::uploadBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk_vertex_buffer, vk_vertex_buffer_allocation)) return false;
            if (!texture_resources::uploadBuffer(indices.data(), indices.size() * sizeof(uint32_t),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk_index_buffer, vk_index_buffer_allocation)) return false;

            {
                VkBufferCreateInfo buf_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(SceneUniforms),
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationCreateInfo alloc_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo res{};
                if (vmaCreateBuffer(graphics::internal::context.allocator, &buf_info, &alloc_info, &vk_scene_uniform_buffer, &vk_scene_uniform_buffer_allocation, &res) != VK_SUCCESS) return false;
                vk_scene_uniform_buffer_mapped = static_cast<SceneUniforms*>(res.pMappedData);
            }

            // Point Lights SSBO
            {
                VkBufferCreateInfo buf_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(GPUPointLight) * MAX_LIGHTS,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationCreateInfo alloc_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo res{};
                if (vmaCreateBuffer(graphics::internal::context.allocator, &buf_info, &alloc_info, &vk_point_ssbo, &vk_point_ssbo_allocation, &res) != VK_SUCCESS) return false;
                vk_point_ssbo_mapped = static_cast<GPUPointLight*>(res.pMappedData);
            }

            // Spot Lights SSBO
            {
                VkBufferCreateInfo buf_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = sizeof(GPUSpotLight) * MAX_LIGHTS,
                    .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                VmaAllocationCreateInfo alloc_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo res{};
                if (vmaCreateBuffer(graphics::internal::context.allocator, &buf_info, &alloc_info, &vk_spot_ssbo, &vk_spot_ssbo_allocation, &res) != VK_SUCCESS) return false;
                vk_spot_ssbo_mapped = static_cast<GPUSpotLight*>(res.pMappedData);
            }

            {
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
                const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
                model_uniform_stride = (sizeof(ModelUniform) + alignment - 1) & ~(alignment - 1);
                assert(model_uniform_stride % alignment == 0);
                const VkBufferCreateInfo buf_info = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = model_uniform_stride * model_capacity,
                    .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                };
                const VmaAllocationCreateInfo alloc_info = {
                    .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO,
                };
                VmaAllocationInfo result{};
                if (vmaCreateBuffer(graphics::internal::context.allocator, &buf_info, &alloc_info,
                    &vk_model_uniform_buffer, &vk_model_uniform_buffer_allocation, &result) != VK_SUCCESS) return false;
                vk_model_uniform_buffer_mapped = static_cast<std::byte*>(result.pMappedData);
            }

            {
                std::vector<glm::vec3> cone_lines;
                constexpr int segments = 16;
                for (int i = 0; i < segments; ++i) {
                    float a0 = (float(i) / segments) * 2.0f * 3.14159265f;
                    float a1 = (float(i + 1) / segments) * 2.0f * 3.14159265f;
                    cone_lines.push_back({ std::cos(a0), std::sin(a0), 1.0f });
                    cone_lines.push_back({ std::cos(a1), std::sin(a1), 1.0f });
                    if (i % 4 == 0) {
                        cone_lines.push_back({ 0.0f, 0.0f, 0.0f });
                        cone_lines.push_back({ std::cos(a0), std::sin(a0), 1.0f });
                    }
                }
                cone_vertex_count = static_cast<uint32_t>(cone_lines.size());
                if (!texture_resources::uploadBuffer(cone_lines.data(), cone_lines.size() * sizeof(glm::vec3),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk_cone_vertex_buffer, vk_cone_vertex_buffer_allocation)) return false;
            }

            return true;
        }

        bool createTextures() {
            const char* paths[] = {
                "assets/textures/portrait_chain.png",
                "assets/textures/portrait_wide.png",
                "assets/textures/portrait_suit.png",
            };
            for (uint32_t i = 0; i < material_count; ++i) {
                if (!texture_resources::loadTexture(paths[i], VK_FORMAT_R8G8B8A8_SRGB, albedo_textures[i])) return false;
            }
            if (!texture_resources::loadTexture("assets/textures/specular_grid.png", VK_FORMAT_R8G8B8A8_UNORM, specular_texture) ||
                !texture_resources::loadTexture("assets/textures/black.png", VK_FORMAT_R8G8B8A8_UNORM, black_texture)) return false;
            const VkSamplerCreateInfo sampler_info = {
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .maxLod = 0.0f,
            };
            return vkCreateSampler(graphics::internal::context.device, &sampler_info, nullptr, &vk_texture_sampler) == VK_SUCCESS;
        }

        bool createDescriptors() {
            auto& device = graphics::internal::context.device;

            const VkDescriptorSetLayoutBinding scene_bindings[] = {
                {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
                {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            };

            const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,
                .pBindings = scene_bindings,
            };

            if (vkCreateDescriptorSetLayout(device, &scene_layout_info, nullptr, &vk_scene_descriptor_set_layout) != VK_SUCCESS) return false;

            const VkDescriptorSetLayoutBinding model_bindings[] = {
                { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
                  .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            };
            const VkDescriptorSetLayoutCreateInfo model_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 4,
                .pBindings = model_bindings,
            };

            if (vkCreateDescriptorSetLayout(device, &model_layout_info, nullptr, &vk_model_descriptor_set_layout) != VK_SUCCESS) return false;

            const VkDescriptorSetLayout set_layouts[] = { vk_scene_descriptor_set_layout, vk_model_descriptor_set_layout };
            const VkPushConstantRange texture_push_range = {
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(TexturePushConstants),
            };
            const VkPipelineLayoutCreateInfo pipeline_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = set_layouts,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &texture_push_range,
            };

            if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) return false;

            const VkDescriptorPoolSize pool_sizes[] = {
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 },
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = material_count },
                {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * material_count },
                {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
            };

            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1 + material_count,
                .poolSizeCount = 4,
                .pPoolSizes = pool_sizes,
            };

            if (vkCreateDescriptorPool(device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) return false;

            const VkDescriptorSetAllocateInfo scene_alloc = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
            };
            if (vkAllocateDescriptorSets(device, &scene_alloc, &vk_scene_descriptor_set) != VK_SUCCESS) return false;

            VkDescriptorBufferInfo scene_ubo_info = { .buffer = vk_scene_uniform_buffer, .offset = 0, .range = sizeof(SceneUniforms) };
            VkDescriptorBufferInfo point_ssbo_info = { .buffer = vk_point_ssbo, .offset = 0, .range = sizeof(GPUPointLight) * MAX_LIGHTS };
            VkDescriptorBufferInfo spot_ssbo_info = { .buffer = vk_spot_ssbo, .offset = 0, .range = sizeof(GPUSpotLight) * MAX_LIGHTS };

            const VkWriteDescriptorSet scene_writes[] = {
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_ubo_info },
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &point_ssbo_info },
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_scene_descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &spot_ssbo_info },
            };
            vkUpdateDescriptorSets(device, 3, scene_writes, 0, nullptr);

            for (uint32_t i = 0; i < material_count; ++i) {
                const VkDescriptorSetAllocateInfo model_alloc = {
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = vk_descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &vk_model_descriptor_set_layout,
                };
                if (vkAllocateDescriptorSets(device, &model_alloc, &vk_material_descriptor_sets[i]) != VK_SUCCESS) return false;

                const VkDescriptorBufferInfo model_info = { .buffer = vk_model_uniform_buffer, .offset = 0, .range = sizeof(ModelUniform) };
                const VkDescriptorImageInfo images[] = {
                    { vk_texture_sampler, albedo_textures[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                    { vk_texture_sampler, specular_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                    { vk_texture_sampler, i == 0 ? black_texture.view : albedo_textures[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                };
                const VkWriteDescriptorSet writes[] = {
                    { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_material_descriptor_sets[i], .dstBinding = 0,
                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &model_info },
                    { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_material_descriptor_sets[i], .dstBinding = 1,
                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &images[0] },
                    { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_material_descriptor_sets[i], .dstBinding = 2,
                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &images[1] },
                    { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_material_descriptor_sets[i], .dstBinding = 3,
                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &images[2] },
                };
                vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
            }

            return true;
        }

        void bindMarkerUniform(VkCommandBuffer command, uint32_t& index, const glm::mat4& model, const glm::vec4& color) {
            assert(index < model_capacity);
            const uint32_t dynamic_offset = static_cast<uint32_t>(index++ * model_uniform_stride);
            const ModelUniform uniform = { model, color, glm::vec4(0.0f), glm::vec4(0.0f), glm::ivec4(0) };
            std::memcpy(vk_model_uniform_buffer_mapped + dynamic_offset, &uniform, sizeof(uniform));
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline_layout, 1, 1, &vk_material_descriptor_sets[0], 1, &dynamic_offset);
        }

        bool createPipelines() {
            auto& device = graphics::internal::context.device;

            VkShaderModule vert_m = loadShaderModule("shaders/cube.vert.spv");
            VkShaderModule frag_m = loadShaderModule("shaders/cube.frag.spv");
            if (!vert_m || !frag_m) return false;

            const VkPipelineShaderStageCreateInfo stages[] = {
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert_m, .pName = "main" },
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_m, .pName = "main" },
            };

            const VkVertexInputBindingDescription vertex_binding = {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            const VkVertexInputAttributeDescription vertex_attributes[] = {
                {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
                {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
                {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
            };

            const VkPipelineVertexInputStateCreateInfo vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 3,
                .pVertexAttributeDescriptions = vertex_attributes,
            };

            const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            const VkPipelineViewportStateCreateInfo viewport_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1, .scissorCount = 1,
            };

            const VkPipelineRasterizationStateCreateInfo rasterization = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .lineWidth = 1.0f,
            };

            const VkPipelineMultisampleStateCreateInfo multisample = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            };

            const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            const VkPipelineColorBlendAttachmentState color_attachment = {
                .blendEnable = VK_FALSE,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };

            const VkPipelineColorBlendStateCreateInfo color_blend = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
            };

            const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            const VkPipelineDynamicStateCreateInfo dynamic_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states,
            };

            const VkGraphicsPipelineCreateInfo pipeline_info = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = stages,
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterization,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blend,
                .pDynamicState = &dynamic_state,
                .layout = vk_pipeline_layout,
                .renderPass = graphics::internal::context.render_pass,
            };

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) return false;

            vkDestroyShaderModule(device, vert_m, nullptr);
            vkDestroyShaderModule(device, frag_m, nullptr);

            VkShaderModule unlit_vert_m = loadShaderModule("shaders/unlit.vert.spv");
            VkShaderModule unlit_frag_m = loadShaderModule("shaders/unlit.frag.spv");
            if (!unlit_vert_m || !unlit_frag_m) return false;

            const VkPipelineShaderStageCreateInfo unlit_stages[] = {
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = unlit_vert_m, .pName = "main" },
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = unlit_frag_m, .pName = "main" },
            };

            const VkVertexInputAttributeDescription unlit_cube_attr = {
                .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)
            };
            const VkPipelineVertexInputStateCreateInfo unlit_cube_vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 1,
                .pVertexAttributeDescriptions = &unlit_cube_attr,
            };

            VkGraphicsPipelineCreateInfo unlit_info = pipeline_info;
            unlit_info.stageCount = 2;
            unlit_info.pStages = unlit_stages;
            unlit_info.pVertexInputState = &unlit_cube_vertex_input;
            unlit_info.layout = vk_pipeline_layout;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &unlit_info, nullptr, &vk_unlit_pipeline) != VK_SUCCESS) return false;

            const VkVertexInputBindingDescription line_binding = { .binding = 0, .stride = sizeof(glm::vec3), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
            const VkVertexInputAttributeDescription line_attr = { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 };
            const VkPipelineVertexInputStateCreateInfo line_vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &line_binding,
                .vertexAttributeDescriptionCount = 1,
                .pVertexAttributeDescriptions = &line_attr,
            };

            VkPipelineInputAssemblyStateCreateInfo lines_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            };
            unlit_info.pVertexInputState = &line_vertex_input;
            unlit_info.pInputAssemblyState = &lines_assembly;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &unlit_info, nullptr, &vk_unlit_lines_pipeline) != VK_SUCCESS) return false;

            vkDestroyShaderModule(device, unlit_vert_m, nullptr);
            vkDestroyShaderModule(device, unlit_frag_m, nullptr);

            return true;
        }

    } // namespace

    bool initialize() {
#ifndef NDEBUG
        for (float inner : { -100.0f, 1.0f, 15.0f, 40.0f, 60.0f, 200.0f }) {
            for (float outer : { -100.0f, 1.1f, 15.0f, 25.0f, 80.0f, 200.0f }) {
                SpotLightData sample;
                sample.inner_angle_deg = inner;
                sample.outer_angle_deg = outer;
                clampSpotAngles(sample);
                assert(sample.inner_angle_deg >= 1.0f && sample.inner_angle_deg <= 60.0f);
                assert(sample.outer_angle_deg <= 80.0f && sample.inner_angle_deg < sample.outer_angle_deg);
                assert(std::cos(glm::radians(sample.inner_angle_deg)) > std::cos(glm::radians(sample.outer_angle_deg)));
                assert(std::tan(glm::radians(sample.inner_angle_deg)) < std::tan(glm::radians(sample.outer_angle_deg)));
            }
        }
#endif
        ImGui::StyleColorsDark();
        auto& style = ImGui::GetStyle();
        style.WindowPadding = ImVec2(16.0f, 14.0f);
        style.FramePadding = ImVec2(6.0f, 5.0f);
        style.ItemSpacing = ImVec2(8.0f, 8.0f);
        style.FrameRounding = 4.0f;
        style.GrabRounding = 4.0f;
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.065f, 0.09f, 1.0f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.13f, 0.25f, 0.31f, 1.0f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.32f, 0.37f, 1.0f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.32f, 0.85f, 0.76f, 1.0f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.32f, 0.85f, 0.76f, 1.0f);
        point_lights.push_back({ { 0.0f, 2.5f, 0.0f }, { 1.0f, 0.9f, 0.7f }, 6.0f });
        spot_lights.push_back({ { -3.0f, 3.5f, 2.0f }, glm::normalize(glm::vec3(0.5f, -1.0f, -0.5f)), { 0.3f, 0.7f, 1.0f }, 15.0f, 15.0f, 25.0f });

        if (!createBuffers()) {
            MessageBoxA(nullptr, "createBuffers() failed! Check VMA allocations.", "Init Error", MB_ICONERROR | MB_OK);
            return false;
        }

        if (!createTextures()) {
            MessageBoxA(nullptr, "Cannot load textures. Run the application from the project root (assets/textures).", "Texture Load Error", MB_ICONERROR | MB_OK);
            return false;
        }

        if (!createDescriptors()) {
            MessageBoxA(nullptr, "createDescriptors() failed! Check descriptor layouts/pool.", "Init Error", MB_ICONERROR | MB_OK);
            return false;
        }

        if (!createPipelines()) {
            MessageBoxA(nullptr, "createPipelines() failed! Check shader compilation.", "Init Error", MB_ICONERROR | MB_OK);
            return false;
        }

        return true;
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkDeviceWaitIdle(context.device);

        if (vk_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, vk_pipeline, nullptr);
        if (vk_unlit_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, vk_unlit_pipeline, nullptr);
        if (vk_unlit_lines_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, vk_unlit_lines_pipeline, nullptr);

        if (vk_pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

        if (vk_descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
        if (vk_scene_descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, vk_scene_descriptor_set_layout, nullptr);
        if (vk_model_descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, vk_model_descriptor_set_layout, nullptr);

        if (vk_scene_uniform_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_scene_uniform_buffer, vk_scene_uniform_buffer_allocation);
        if (vk_point_ssbo != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_point_ssbo, vk_point_ssbo_allocation);
        if (vk_spot_ssbo != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_spot_ssbo, vk_spot_ssbo_allocation);

        if (vk_model_uniform_buffer) vmaDestroyBuffer(context.allocator, vk_model_uniform_buffer, vk_model_uniform_buffer_allocation);
        if (vk_texture_sampler) vkDestroySampler(context.device, vk_texture_sampler, nullptr);
        for (auto& texture : albedo_textures) texture_resources::destroyTexture(texture);
        texture_resources::destroyTexture(specular_texture);
        texture_resources::destroyTexture(black_texture);
        if (vk_vertex_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
        if (vk_index_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
        if (vk_cone_vertex_buffer != VK_NULL_HANDLE) vmaDestroyBuffer(context.allocator, vk_cone_vertex_buffer, vk_cone_vertex_buffer_allocation);
    }

    void update(double time) {
        const auto& extent = graphics::internal::context.swapchain_extent;
        const float aspect = float(sceneWidth()) / float(extent.height);

        static int projection_mode = 0;
        static bool is_playing = false;
        static float anim_speed = 1.0f;
        static float anim_radius = 1.5f;
        static float anim_height = 0.5f;
        static float anim_angle = 0.0f;

        static double last_time = time;
        const double delta = time - last_time;
        last_time = time;

        if (is_playing) {
            anim_angle += float(delta) * anim_speed;
            if (anim_angle > 2.0f * 3.14159265f) {
                anim_angle -= 2.0f * 3.14159265f;
            }
        }

        if (animate_texture) texture_time += float(delta);
        const auto display = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(display.x - controlsWidth(), 0.0f));
        ImGui::SetNextWindowSize(ImVec2(controlsWidth(), display.y));
        ImGui::Begin("Texture studio", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
        ImGui::Separator();

        static int selected_cube = 0;
        if (ImGui::BeginTabBar("Cubes")) {
            for (int i = 0; i < int(cube_count); ++i) {
                const std::string label = "Cube " + std::to_string(i + 1);
                if (ImGui::BeginTabItem(label.c_str())) {
                    selected_cube = i;
                    ImGui::EndTabItem();
                }
            }
            ImGui::EndTabBar();
        }
        auto& selected = cube_settings[selected_cube];
        ImGui::Combo("Texture", &selected.material, material_names, material_count);
        ImGui::Combo("Sampling", &selected.effect, "Original\0Waves + RGB split\0");
        if (selected.effect != 0) {
            ImGui::SliderFloat("Effect strength", &selected.effect_strength, 0.0f, 1.0f);
            ImGui::SliderFloat("Color modulation", &selected.color_modulation, 0.0f, 1.0f);
        }
        ImGui::Checkbox("Animate textures", &animate_texture);
        if (ImGui::CollapsingHeader("Material maps", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo tint", &selected.albedo.x);
            ImGui::Checkbox("Specular map", &selected.specular_map);
            ImGui::ColorEdit3("Specular tint", &selected.specular.x);
            ImGui::SliderFloat("Shininess", &selected.shininess, 1.0f, 256.0f);
            ImGui::BeginDisabled(selected.material == 0);
            ImGui::Checkbox("Emissive map", &selected.emissive_map);
            ImGui::SliderFloat("Emission", &selected.emission_strength, 0.0f, 3.0f);
            ImGui::ColorEdit3("Emission tint", &selected.emission_tint.x);
            ImGui::EndDisabled();
            if (selected.material == 0) ImGui::TextDisabled("No emission map for this material.");
        }
        ImGui::Separator();

        // Проекция
        ImGui::Text("Projection:");
        ImGui::RadioButton("Perspective", &projection_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &projection_mode, 1);

        // Интерактивная камера 
        if (ImGui::CollapsingHeader("Camera (View Matrix)")) {
            ImGui::SliderFloat3("Position", &camera.position.x, -15.0f, 15.0f);
            ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);
            ImGui::SliderFloat("FOV", &camera.fov, 15.0f, 110.0f);
        }

        if (ImGui::CollapsingHeader("Global Animation (Lissajous)")) {
            ImGui::Checkbox("Play animation", &is_playing);
            ImGui::SliderFloat("Speed", &anim_speed, -3.0f, 3.0f);
            ImGui::SliderFloat("Radius", &anim_radius, 0.0f, 3.0f);
            ImGui::SliderFloat("Height", &anim_height, -1.5f, 1.5f);
            ImGui::Text("Angle: %.2f rad", anim_angle);
        }

        // Освещение сцены
        if (ImGui::CollapsingHeader("Lighting: Directional & Ambient")) {
            ImGui::SliderFloat3("Direction", &dir_light_direction.x, -1.0f, 1.0f);
            if (glm::length(dir_light_direction) > 0.001f) dir_light_direction = glm::normalize(dir_light_direction);
            ImGui::ColorEdit3("Sun Color", &dir_light_color.x);
            ImGui::SliderFloat("Sun Intensity", &dir_light_intensity, 0.0f, 5.0f);
            ImGui::Separator();
            ImGui::ColorEdit3("Ambient Light", &ambient_light_color.x);
        }

        // Массив точечных источников в SSBO
        if (ImGui::CollapsingHeader("Point Lights (SSBO Array)")) {
            if (ImGui::Button("+ Add Point Light") && point_lights.size() < MAX_LIGHTS) {
                point_lights.push_back({ { 0.0f, 2.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 5.0f });
            }
            ImGui::SameLine();
            ImGui::Text("(Count: %zu)", point_lights.size());

            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                std::string title = "Point Light #" + std::to_string(i + 1);
                if (ImGui::TreeNode(title.c_str())) {
                    ImGui::SliderFloat3("Position", &point_lights[i].position.x, -10.0f, 10.0f);
                    ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                    ImGui::SliderFloat("Intensity", &point_lights[i].intensity, 0.1f, 30.0f);
                    if (ImGui::Button("Delete")) {
                        point_lights.erase(point_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        // Массив прожекторных источников в SSBO
        if (ImGui::CollapsingHeader("Spot Lights (SSBO Array)")) {
            if (ImGui::Button("+ Add Spot Light") && spot_lights.size() < MAX_LIGHTS) {
                spot_lights.push_back({ { 0.0f, 3.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 10.0f, 15.0f, 25.0f });
            }
            ImGui::SameLine();
            ImGui::Text("(Count: %zu)", spot_lights.size());

            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(100 + i));
                std::string title = "Spot Light #" + std::to_string(i + 1);
                if (ImGui::TreeNode(title.c_str())) {
                    ImGui::SliderFloat3("Position", &spot_lights[i].position.x, -10.0f, 10.0f);
                    ImGui::SliderFloat3("Direction", &spot_lights[i].direction.x, -1.0f, 1.0f);
                    if (glm::length(spot_lights[i].direction) > 0.001f) spot_lights[i].direction = glm::normalize(spot_lights[i].direction);
                    else spot_lights[i].direction = glm::vec3(0.0f, -1.0f, 0.0f);
                    ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
                    ImGui::SliderFloat("Intensity", &spot_lights[i].intensity, 0.1f, 50.0f);
                    auto& light = spot_lights[i];
                    ImGui::SliderFloat("Inner Angle", &light.inner_angle_deg, 1.0f,
                        std::min(60.0f, light.outer_angle_deg - 0.1f), "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
                    ImGui::SliderFloat("Outer Angle", &light.outer_angle_deg,
                        light.inner_angle_deg + 0.1f, 80.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
                    clampSpotAngles(light);
                    ImGui::TextDisabled("Inner cone stays inside the outer cone.");
                    if (ImGui::Button("Delete")) {
                        spot_lights.erase(spot_lights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Objects: Transform, Anim & Materials")) {
            for (uint32_t i = 0; i < cube_count; ++i) {
                std::string label = "Cube #" + std::to_string(i + 1);
                if (ImGui::TreeNode(label.c_str())) {
                    CubeSettings& s = cube_settings[i];

                    ImGui::Checkbox("Animate", &s.animate);
                    ImGui::SliderFloat("Phase offset", &s.phaseOffset, -6.28f, 6.28f, "%.2f rad");

                    ImGui::Separator();
                    ImGui::SliderFloat3("Position", &s.position.x, -5.0f, 5.0f);
                    ImGui::SliderFloat3("Rotation", &s.rotationDeg.x, -180.0f, 180.0f);
                    ImGui::SliderFloat3("Scale", &s.scale.x, 0.1f, 3.0f);

                    ImGui::Separator();
                    ImGui::Text("Blinn-Phong Material:");
                    ImGui::ColorEdit3("Albedo (Kd)", &s.albedo.x);
                    ImGui::ColorEdit3("Specular (Ks)", &s.specular.x);
                    ImGui::SliderFloat("Shininess (Alpha)", &s.shininess, 1.0f, 256.0f);

                    ImGui::TreePop();
                }
            }
        }

        ImGui::Checkbox("Show Light Gizmos", &show_light_visualizers);
        ImGui::End();

        // Расчет направления камеры и матрицы вида (View)
        glm::vec3 front;
        front.x = cos(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
        front.y = sin(glm::radians(camera.pitch));
        front.z = sin(glm::radians(camera.yaw)) * cos(glm::radians(camera.pitch));
        front = glm::normalize(front);

        glm::mat4 view = glm::lookAt(camera.position, camera.position + front, glm::vec3(0.0f, 1.0f, 0.0f));

        // Расчет матрицы проекции (перспектива или ортографическая)
        glm::mat4 proj;
        if (projection_mode == 0) {
            proj = glm::perspective(glm::radians(camera.fov), aspect, 0.1f, 100.0f);
        }
        else {
            float ortho_size = 6.0f;
            proj = glm::ortho(-ortho_size * aspect, ortho_size * aspect, -ortho_size, ortho_size, 0.1f, 100.0f);
        }
        proj[1][1] *= -1.0f;

        // Отправка общих данных сцены
        vk_scene_uniform_buffer_mapped->view_proj = proj * view;
        vk_scene_uniform_buffer_mapped->camera_position = glm::vec4(camera.position, 1.0f);
        vk_scene_uniform_buffer_mapped->directional_light_direction = glm::vec4(dir_light_direction, 0.0f);
        vk_scene_uniform_buffer_mapped->directional_light_color = glm::vec4(dir_light_color * dir_light_intensity, 1.0f);
        vk_scene_uniform_buffer_mapped->ambient_light = glm::vec4(ambient_light_color, 1.0f);
        vk_scene_uniform_buffer_mapped->point_light_count = static_cast<uint32_t>(point_lights.size());
        vk_scene_uniform_buffer_mapped->spot_light_count = static_cast<uint32_t>(spot_lights.size());

        // Заполнение SSBO точечных источников
        for (size_t i = 0; i < point_lights.size(); ++i) {
            vk_point_ssbo_mapped[i].position = glm::vec4(point_lights[i].position, point_lights[i].intensity);
            vk_point_ssbo_mapped[i].color = glm::vec4(point_lights[i].color, 1.0f);
        }

        // Заполнение SSBO прожекторных источников
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            clampSpotAngles(spot_lights[i]);
            assert(spot_lights[i].inner_angle_deg < spot_lights[i].outer_angle_deg);
            vk_spot_ssbo_mapped[i].position = glm::vec4(spot_lights[i].position, std::cos(glm::radians(spot_lights[i].inner_angle_deg)));
            vk_spot_ssbo_mapped[i].direction = glm::vec4(spot_lights[i].direction, std::cos(glm::radians(spot_lights[i].outer_angle_deg)));
            vk_spot_ssbo_mapped[i].color = glm::vec4(spot_lights[i].color, spot_lights[i].intensity);
        }

        // Расчет матриц моделей с анимацией Лиссажу 
        for (uint32_t i = 0; i < cube_count; ++i) {
            CubeSettings& s = cube_settings[i];

            glm::vec3 final_position = cube_offsets[i] + s.position;
            glm::vec3 final_rotation = s.rotationDeg;

            if (s.animate && is_playing) {
                const float phase = anim_angle + s.phaseOffset;
                final_position.x += anim_radius * std::cos(phase);
                final_position.z += anim_radius * std::sin(phase * 2.0f);
                final_position.y += anim_height * std::sin(phase * 3.0f);
                final_rotation.y += anim_angle * 40.0f + s.phaseOffset * 30.0f;
            }

            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, final_position);
            model = glm::rotate(model, glm::radians(final_rotation.x), glm::vec3(1, 0, 0));
            model = glm::rotate(model, glm::radians(final_rotation.y), glm::vec3(0, 1, 0));
            model = glm::rotate(model, glm::radians(final_rotation.z), glm::vec3(0, 0, 1));
            model = glm::scale(model, s.scale);

            const ModelUniform uniform = {
                model, glm::vec4(s.albedo, 1.0f), glm::vec4(s.specular, s.shininess),
                glm::vec4(s.emission_tint, s.emission_strength),
                glm::ivec4(int(s.specular_map), int(s.emissive_map), 0, 0),
            };
            std::memcpy(vk_model_uniform_buffer_mapped + i * model_uniform_stride, &uniform, sizeof(uniform));
        }
        const auto allocator = graphics::internal::context.allocator;
        if (vmaFlushAllocation(allocator, vk_scene_uniform_buffer_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS ||
            vmaFlushAllocation(allocator, vk_point_ssbo_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS ||
            vmaFlushAllocation(allocator, vk_spot_ssbo_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
            throw std::runtime_error("Cannot flush scene buffers");
        }
    }

    void render(const graphics::internal::FrameData& fd) {
        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(fd.command_buffer, &begin_info);

        const VkClearValue clear_values[] = {
            {.color = {.float32 = { 0.05f, 0.05f, 0.08f, 1.0f } } },
            {.depthStencil = { 1.0f, 0 } },
        };

        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = graphics::internal::context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.extent = graphics::internal::context.swapchain_extent },
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport viewport = {
            .x = 0.0f, .y = 0.0f,
            .width = float(sceneWidth()),
            .height = float(graphics::internal::context.swapchain_extent.height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };
        const VkRect2D scissor = { .extent = { sceneWidth(), graphics::internal::context.swapchain_extent.height } };

        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        // Отрисовка освещаемых кубов
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);

        for (uint32_t i = 0; i < cube_count; ++i) {
            const auto& settings = cube_settings[i];
            const uint32_t dynamic_offset = static_cast<uint32_t>(i * model_uniform_stride);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline_layout, 1, 1, &vk_material_descriptor_sets[settings.material], 1, &dynamic_offset);
            const TexturePushConstants push = {
                texture_time, settings.effect_strength, settings.color_modulation, settings.effect,
            };
            vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDrawIndexed(fd.command_buffer, cube_index_count, 1, 0, 0, 0);
        }

        // Визуализация источников (яркие маркеры и конусы)
        if (show_light_visualizers) {
            uint32_t marker_index = cube_count;
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_unlit_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);

            // Точечные источники
            for (const auto& pl : point_lights) {
                glm::mat4 m = glm::translate(glm::mat4(1.0f), pl.position);
                m = glm::scale(m, glm::vec3(0.12f));
                bindMarkerUniform(fd.command_buffer, marker_index, m, glm::vec4(pl.color, 1.0f));
                vkCmdDrawIndexed(fd.command_buffer, cube_index_count, 1, 0, 0, 0);
            }

            // Прожекторные источники
            for (const auto& sl : spot_lights) {
                glm::mat4 m = glm::translate(glm::mat4(1.0f), sl.position);
                m = glm::scale(m, glm::vec3(0.12f));
                bindMarkerUniform(fd.command_buffer, marker_index, m, glm::vec4(sl.color, 1.0f));
                vkCmdDrawIndexed(fd.command_buffer, cube_index_count, 1, 0, 0, 0);
            }

            // Конусы прожекторов
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_unlit_lines_pipeline);
            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_cone_vertex_buffer, &offset);

            for (const auto& sl : spot_lights) {
                glm::vec3 dir = glm::normalize(sl.direction);
                glm::vec3 default_dir(0.0f, 0.0f, 1.0f);
                glm::mat4 rot(1.0f);

                if (glm::length(dir - default_dir) > 0.0001f) {
                    if (glm::length(dir + default_dir) < 0.0001f) {
                        rot = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1, 0, 0));
                    }
                    else {
                        glm::vec3 axis = glm::cross(default_dir, dir);
                        float angle = std::acos(std::clamp(glm::dot(default_dir, dir), -1.0f, 1.0f));
                        rot = glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis));
                    }
                }

                const float length = 2.5f;

                // Внешний конус (Outer Angle) — белый цвет (граница затухания)
                float radius_outer = length * std::tan(glm::radians(sl.outer_angle_deg));
                glm::mat4 m_outer = glm::translate(glm::mat4(1.0f), sl.position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(radius_outer, radius_outer, length));
                bindMarkerUniform(fd.command_buffer, marker_index, m_outer, glm::vec4(1.0f));
                vkCmdDraw(fd.command_buffer, cone_vertex_count, 1, 0, 0);

                // Внутренний конус (Inner Angle) — желтый цвет (100% яркости)
                float radius_inner = length * std::tan(glm::radians(sl.inner_angle_deg));
                glm::mat4 m_inner = glm::translate(glm::mat4(1.0f), sl.position) * rot * glm::scale(glm::mat4(1.0f), glm::vec3(radius_inner, radius_inner, length));
                bindMarkerUniform(fd.command_buffer, marker_index, m_inner, glm::vec4(1.0f, 0.85f, 0.2f, 1.0f));
                vkCmdDraw(fd.command_buffer, cone_vertex_count, 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
        // Marker uniforms are written during recording, before submission reads the shared buffer.
        if (vmaFlushAllocation(graphics::internal::context.allocator,
            vk_model_uniform_buffer_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
            throw std::runtime_error("Cannot flush model uniforms");
        }
    }

} // namespace application
