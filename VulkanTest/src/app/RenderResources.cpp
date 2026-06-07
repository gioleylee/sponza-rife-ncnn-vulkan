// Owns framebuffer, depth, G-buffer attachment, and image-view resources.
#include "VulkanRifeRendererApp.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>

void VulkanRifeRendererApp::createFramebuffers() {
    offscreenFramebuffers.resize(offscreenFrames.size());

    for (size_t i = 0; i < offscreenFrames.size(); i++) {
        VkImageView attachments[] = {
            offscreenFrames[i].imageView,
            gNormalImageViews[i],
            gAlbedoImageViews[i],
            gPositionImageViews[i],
            depthImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 5;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &offscreenFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create offscreen framebuffer!");
        }
    }
}

void VulkanRifeRendererApp::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();

    depthImages.resize(offscreenFrames.size());
    depthImageMemories.resize(offscreenFrames.size());
    depthImageViews.resize(offscreenFrames.size());

    for (size_t i = 0; i < offscreenFrames.size(); ++i) {
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            depthFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            depthImages[i],
            depthImageMemories[i]
        );
        depthImageViews[i] = createImageView(depthImages[i], depthFormat, 1);
    }
}

void VulkanRifeRendererApp::createGBufferAttachments() {
    VkFormat normalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    gNormalImages.resize(offscreenFrames.size());
    gNormalImageMemories.resize(offscreenFrames.size());
    gNormalImageViews.resize(offscreenFrames.size());

    for (size_t i = 0; i < offscreenFrames.size(); ++i) {
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            normalFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            gNormalImages[i],
            gNormalImageMemories[i]
        );
        gNormalImageViews[i] = createImageView(gNormalImages[i], normalFormat, 1);
    }

    VkFormat albedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
    gAlbedoImages.resize(offscreenFrames.size());
    gAlbedoImageMemories.resize(offscreenFrames.size());
    gAlbedoImageViews.resize(offscreenFrames.size());

    for (size_t i = 0; i < offscreenFrames.size(); ++i) {
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            albedoFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            gAlbedoImages[i],
            gAlbedoImageMemories[i]
        );
        gAlbedoImageViews[i] = createImageView(gAlbedoImages[i], albedoFormat, 1);
    }

    VkFormat positionFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    gPositionImages.resize(offscreenFrames.size());
    gPositionImageMemories.resize(offscreenFrames.size());
    gPositionImageViews.resize(offscreenFrames.size());

    for (size_t i = 0; i < offscreenFrames.size(); ++i) {
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            positionFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            gPositionImages[i],
            gPositionImageMemories[i]
        );
        gPositionImageViews[i] = createImageView(gPositionImages[i], positionFormat, 1);
    }
}

