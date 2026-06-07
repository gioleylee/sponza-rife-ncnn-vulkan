// Owns descriptor set layouts, pools, allocation, and descriptor updates.
#include "VulkanRifeRendererApp.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

void VulkanRifeRendererApp::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{}; // UBO
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{}; // textures
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding
    }; // gather all bindings

    VkDescriptorSetLayoutCreateInfo layoutInfo{}; // info
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    } // creation
}

void VulkanRifeRendererApp::createLightingDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding normalInputBinding{};
    normalInputBinding.binding = 0;
    normalInputBinding.descriptorCount = 1;
    normalInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    normalInputBinding.pImmutableSamplers = nullptr;
    normalInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding albedoInputBinding{};
    albedoInputBinding.binding = 1;
    albedoInputBinding.descriptorCount = 1;
    albedoInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    albedoInputBinding.pImmutableSamplers = nullptr;
    albedoInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding positionInputBinding{};
    positionInputBinding.binding = 2;
    positionInputBinding.descriptorCount = 1;
    positionInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    positionInputBinding.pImmutableSamplers = nullptr;
    positionInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
        normalInputBinding, albedoInputBinding, positionInputBinding
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &lightingDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create lighting descriptor set layout!");
    }
}

void VulkanRifeRendererApp::createDescriptorPool() {
    uint32_t materialCount =
        static_cast<uint32_t>(std::max<size_t>(1, materials.size()));

    uint32_t totalSets = MAX_FRAMES_IN_FLIGHT * (materialCount + 1); // per-frame materials plus cube

    VkDescriptorPoolSize uboPoolSize{}; // UBO info
    uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboPoolSize.descriptorCount = totalSets;

    VkDescriptorPoolSize samplerPoolSize{}; // sampler info
    samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerPoolSize.descriptorCount = totalSets;

    std::array<VkDescriptorPoolSize, 2> poolSizes = { uboPoolSize, samplerPoolSize }; // gather

    VkDescriptorPoolCreateInfo poolInfo{}; // pool info
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = totalSets;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    } // pool creation
}

void VulkanRifeRendererApp::createDescriptorSets() {
    uint32_t materialCount = static_cast<uint32_t>(std::max<size_t>(1, materials.size()));
    uint32_t setsPerFrame = materialCount + 1;
    uint32_t totalSets = MAX_FRAMES_IN_FLIGHT * setsPerFrame; // per-frame materials plus cube

    std::vector<VkDescriptorSetLayout> layouts(totalSets, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{}; // info (layout, etc.)
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = totalSets;
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(totalSets);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    } // creation

    for (uint32_t frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
        for (uint32_t m = 0; m < materialCount; ++m) {
            uint32_t idx = frame * setsPerFrame + m;

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[frame];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};

            if (!materials.empty() &&
                materials[m].imageView != VK_NULL_HANDLE &&
                materials[m].sampler != VK_NULL_HANDLE) {
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = materials[m].imageView;
                imageInfo.sampler = materials[m].sampler;
            }
            else {
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = fallbackImageView;
                imageInfo.sampler = fallbackSampler;
            }

            std::array<VkWriteDescriptorSet, 2> writes{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descriptorSets[idx];
            writes[0].dstBinding = 0;
            writes[0].dstArrayElement = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &bufferInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = descriptorSets[idx];
            writes[1].dstBinding = 1;
            writes[1].dstArrayElement = 0;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].descriptorCount = 1;
            writes[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device,
                static_cast<uint32_t>(writes.size()), writes.data(),
                0, nullptr);
        }

        uint32_t cubeSetIndex = frame * setsPerFrame + materialCount;

        VkDescriptorBufferInfo cubeBufferInfo{};
        cubeBufferInfo.buffer = cubeUniformBuffers[frame];
        cubeBufferInfo.offset = 0;
        cubeBufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo cubeImageInfo{};
        cubeImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cubeImageInfo.imageView = fallbackImageView;
        cubeImageInfo.sampler = fallbackSampler;

        std::array<VkWriteDescriptorSet, 2> cubeWrites{};

        cubeWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        cubeWrites[0].dstSet = descriptorSets[cubeSetIndex];
        cubeWrites[0].dstBinding = 0;
        cubeWrites[0].dstArrayElement = 0;
        cubeWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cubeWrites[0].descriptorCount = 1;
        cubeWrites[0].pBufferInfo = &cubeBufferInfo;

        cubeWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        cubeWrites[1].dstSet = descriptorSets[cubeSetIndex];
        cubeWrites[1].dstBinding = 1;
        cubeWrites[1].dstArrayElement = 0;
        cubeWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cubeWrites[1].descriptorCount = 1;
        cubeWrites[1].pImageInfo = &cubeImageInfo;

        vkUpdateDescriptorSets(device,
            static_cast<uint32_t>(cubeWrites.size()), cubeWrites.data(),
            0, nullptr);
    }
}

void VulkanRifeRendererApp::createLightingDescriptorPool() {
    uint32_t imageCount = static_cast<uint32_t>(offscreenFrames.size());

    VkDescriptorPoolSize inputAttachmentPoolSize{};
    inputAttachmentPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    inputAttachmentPoolSize.descriptorCount = imageCount * 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &inputAttachmentPoolSize;
    poolInfo.maxSets = imageCount;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &lightingDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create lighting descriptor pool!");
    }
}

void VulkanRifeRendererApp::createLightingDescriptorSets() {
    uint32_t imageCount = static_cast<uint32_t>(offscreenFrames.size());

    std::vector<VkDescriptorSetLayout> layouts(imageCount, lightingDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = lightingDescriptorPool;
    allocInfo.descriptorSetCount = imageCount;
    allocInfo.pSetLayouts = layouts.data();

    lightingDescriptorSets.resize(imageCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, lightingDescriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate lighting descriptor sets!");
    }

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normalInfo.imageView = gNormalImageViews[i];

        VkDescriptorImageInfo albedoInfo{};
        albedoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        albedoInfo.imageView = gAlbedoImageViews[i];

        VkDescriptorImageInfo positionInfo{};
        positionInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        positionInfo.imageView = gPositionImageViews[i];

        std::array<VkWriteDescriptorSet, 3> writes{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = lightingDescriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &normalInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = lightingDescriptorSets[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &albedoInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = lightingDescriptorSets[i];
        writes[2].dstBinding = 2;
        writes[2].dstArrayElement = 0;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &positionInfo;

        vkUpdateDescriptorSets(device,
            static_cast<uint32_t>(writes.size()), writes.data(),
            0, nullptr);
    }
}
