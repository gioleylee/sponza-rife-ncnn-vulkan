// Owns framebuffer, depth, G-buffer attachment, and image-view resources.
#include "HelloTriangleApplication.h"

void HelloTriangleApplication::createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapChainImageViews[i],
            gNormalImageViews[i],
            gAlbedoImageViews[i],
            gPositionImageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 5;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void HelloTriangleApplication::createDepthResources() {
    VkFormat depthFormat = findDepthFormat();

    uint32_t mipLevels = 1;

    createImage(
        swapChainExtent.width,
        swapChainExtent.height,
        mipLevels,
        depthFormat,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        depthImage,
        depthImageMemory
    );

    depthImageView = createImageView(depthImage, depthFormat, mipLevels);
}

void HelloTriangleApplication::createGBufferAttachments() {
    VkFormat normalFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    gNormalImages.resize(swapChainImages.size());
    gNormalImageMemories.resize(swapChainImages.size());
    gNormalImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); ++i) {
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
    gAlbedoImages.resize(swapChainImages.size());
    gAlbedoImageMemories.resize(swapChainImages.size());
    gAlbedoImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); ++i) {
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
    gPositionImages.resize(swapChainImages.size());
    gPositionImageMemories.resize(swapChainImages.size());
    gPositionImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); ++i) {
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

VkImageView HelloTriangleApplication::createImageView(VkImage image, VkFormat format, uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view!");
    }

    return imageView;
}
