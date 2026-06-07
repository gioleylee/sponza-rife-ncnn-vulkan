#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "VulkanNcnnRenderer.h"

namespace {

PresentationCommandMode choosePresentationCommandMode(const NcnnPresentationState& ncnnState,
                                                      size_t offscreenFrameCount) {
    // RenderFrame: render a new scene frame into the offscreen history, then present that source frame.
    PresentationCommandMode mode = PresentationCommandMode::RenderFrame;

    // DisplayInterpolatedFrame: present the completed NCNN N+0.5 output before advancing source frames.
    if (ncnnState.ncnnRealtimeInterpolationEnabled && ncnnState.hasNcnnDisplayFrame) {
        mode = PresentationCommandMode::DisplayInterpolatedFrame;
    }
    // DisplayCapturedSourceFrame: present a captured source frame after interpolation completion or failure.
    else if (ncnnState.ncnnRealtimeInterpolationEnabled &&
             ncnnState.ncnnPendingSourceDisplayIndex < offscreenFrameCount) {
        mode = PresentationCommandMode::DisplayCapturedSourceFrame;
    }
    // DisplayHeldSourceFrame: keep the last source frame visible while async NCNN inference is still running.
    else if (ncnnState.ncnnRealtimeInterpolationEnabled &&
             ncnnState.ncnnInferenceInFlight &&
             ncnnState.ncnnHeldSourceDisplayIndex < offscreenFrameCount) {
        mode = PresentationCommandMode::DisplayHeldSourceFrame;
    }
    // DisplayHeldSourceFrame: keep the held source visible during render-ahead before the interpolated frame is ready.
    else if (ncnnState.ncnnRealtimeInterpolationEnabled &&
             ncnnState.ncnnRenderAheadPending &&
             ncnnState.ncnnHeldSourceDisplayIndex < offscreenFrameCount) {
        mode = PresentationCommandMode::DisplayHeldSourceFrame;
    }

    return mode;
}

constexpr float SPONZA_FLOOR_Y = -1.264425f;
constexpr glm::vec3 CESIUM_MAN_SCENE_POSITION = glm::vec3(0.75f, SPONZA_FLOOR_Y + 1.25f, 0.75f);

}

void VulkanNcnnRenderer::run() {
#if HAS_NCNN
    std::cout << "[NCNN] enabled" << std::endl;
#else
    std::cout << "[NCNN] disabled" << std::endl;
#endif

    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void VulkanNcnnRenderer::initVulkan() {
    initializeCoreVulkan();
    initializeSwapchainResources();
    initializeRenderResources();
    initializeCommandResources();
    initializeSceneResources();
    initializeDescriptorResources();
    initializeSyncResources();
    initializeOptionalNcnn();
}

void VulkanNcnnRenderer::initializeCoreVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
}

void VulkanNcnnRenderer::initializeSwapchainResources() {
    createSwapChain();
    createImageViews();
    createFrameProcessingResources();
}

void VulkanNcnnRenderer::initializeRenderResources() {
    createRenderPass();
    createDepthResources();
    createGBufferAttachments();
    createDescriptorSetLayout();
    createLightingDescriptorSetLayout();
    createGraphicsPipeline();
    createSkinnedGraphicsPipeline();
    createLightingPipeline();
    createFramebuffers();
}

void VulkanNcnnRenderer::initializeCommandResources() {
    createCommandPool();
}

void VulkanNcnnRenderer::initializeSceneResources() {
    createFallbackTexture();
    updateCameraFrontFromAngles();
    rotatingCubePosition = cameraPos + cameraFront * 3.0f;
    loadModel("assets/sponza/sponza.obj");
    loadCesiumMan("assets/CesiumMan.glb");
    appendRotatingCubeGeometry();
    createVertexBuffer();
    createIndexBuffer();
    createSkinnedVertexBuffer();
    createSkinnedIndexBuffer();
    loadMaterialTextures();
    createUniformBuffers();
}

void VulkanNcnnRenderer::initializeDescriptorResources() {
    createDescriptorPool();
    createDescriptorSets();
    createLightingDescriptorPool();
    createLightingDescriptorSets();
}

void VulkanNcnnRenderer::initializeSyncResources() {
    createCommandBuffers();
    createSyncObjects();
}

void VulkanNcnnRenderer::initializeOptionalNcnn() {
#if HAS_NCNN
    initNcnn();
    tryLoadDefaultNcnnModel();
#endif
}

void VulkanNcnnRenderer::mainLoop() {
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime =
            std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        lastTime = currentTime;

        glfwPollEvents();
        processInput(deltaTime);
        processMouseLook();
        frameDeltaSeconds = deltaTime;
        elapsedTimeSeconds += deltaTime;
        drawFrame();
    }

    vkDeviceWaitIdle(device);
}

void VulkanNcnnRenderer::cleanup() {
#if HAS_NCNN
    waitForAsyncNcnnInference();
    shutdownNcnn();
#endif
    cleanupSwapChain();

    cleanupRenderPipelines();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, cubeUniformBuffers[i], nullptr);
        vkFreeMemory(device, cubeUniformBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, cesiumUniformBuffers[i], nullptr);
        vkFreeMemory(device, cesiumUniformBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, skinUniformBuffers[i], nullptr);
        vkFreeMemory(device, skinUniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, lightingDescriptorSetLayout, nullptr);

    for (auto& mat : materials) {
        if (mat.sampler)      vkDestroySampler(device, mat.sampler, nullptr);
        if (mat.imageView)    vkDestroyImageView(device, mat.imageView, nullptr);
        if (mat.image)        vkDestroyImage(device, mat.image, nullptr);
        if (mat.imageMemory)  vkFreeMemory(device, mat.imageMemory, nullptr);
    }

    if (fallbackSampler)     vkDestroySampler(device, fallbackSampler, nullptr);
    if (fallbackImageView)   vkDestroyImageView(device, fallbackImageView, nullptr);
    if (fallbackImage)       vkDestroyImage(device, fallbackImage, nullptr);
    if (fallbackImageMemory) vkFreeMemory(device, fallbackImageMemory, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    if (skinnedIndexBuffer) vkDestroyBuffer(device, skinnedIndexBuffer, nullptr);
    if (skinnedIndexBufferMemory) vkFreeMemory(device, skinnedIndexBufferMemory, nullptr);
    if (skinnedVertexBuffer) vkDestroyBuffer(device, skinnedVertexBuffer, nullptr);
    if (skinnedVertexBufferMemory) vkFreeMemory(device, skinnedVertexBufferMemory, nullptr);

    for (auto semaphore : renderFinishedSemaphores) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);

    if (validation::Enabled) {
        validation::destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
}

void VulkanNcnnRenderer::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

void VulkanNcnnRenderer::copyOffscreenImageToSwapchain(VkCommandBuffer commandBuffer,
                                                              uint32_t imageIndex,
                                                              uint32_t offscreenSlot) {
    auto& source = offscreenFrames[offscreenSlot];
    VkImageMemoryBarrier toTransferSrc{};
    toTransferSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferSrc.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toTransferSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferSrc.image = source.image;
    toTransferSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferSrc.subresourceRange.levelCount = 1;
    toTransferSrc.subresourceRange.layerCount = 1;
    toTransferSrc.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    toTransferSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransferSrc);

    VkImageMemoryBarrier toTransferDst{};
    toTransferDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDst.image = swapChainImages[imageIndex];
    toTransferDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferDst.subresourceRange.levelCount = 1;
    toTransferDst.subresourceRange.layerCount = 1;
    toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransferDst);

    VkImageCopy region{};
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent = { swapChainExtent.width, swapChainExtent.height, 1 };
    vkCmdCopyImage(commandBuffer, source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    toTransferSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferSrc.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toTransferSrc.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransferSrc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransferSrc);

    toTransferDst.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDst.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toTransferDst.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferDst.dstAccessMask = 0;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransferDst);
}

void VulkanNcnnRenderer::processCapturedFrameForSlot(uint32_t frameSlot) {
    if (frameSlot >= pendingCaptureSlotByFrame.size()) {
        return;
    }

    const uint32_t captureSlot = pendingCaptureSlotByFrame[frameSlot];
    pendingCaptureSlotByFrame[frameSlot] = UINT32_MAX;

    if (captureSlot == UINT32_MAX || captureSlot >= offscreenFrames.size()) {
        return;
    }

    const auto& capture = offscreenFrames[captureSlot];
    if (capture.size == 0) {
        return;
    }

#if HAS_NCNN
    if (!ncnnPresentationState.ncnnRealtimeInterpolationEnabled || !ncnnModelAttachedToRenderer) {
        return;
    }

    ncnnPresentationState.previousNcnnGpuFrameIndex = ncnnPresentationState.currentNcnnGpuFrameIndex;
    ncnnPresentationState.currentNcnnGpuFrameIndex = captureSlot;
    ncnnPresentationState.hasNcnnGpuFramePair =
        ncnnPresentationState.previousNcnnGpuFrameIndex != UINT32_MAX &&
        ncnnPresentationState.previousNcnnGpuFrameIndex != ncnnPresentationState.currentNcnnGpuFrameIndex &&
        ncnnPresentationState.previousNcnnGpuFrameIndex < offscreenFrames.size();

    ++capturedFrameCount;
#endif

}

void VulkanNcnnRenderer::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

uint32_t VulkanNcnnRenderer::recordCommandBuffer(VkCommandBuffer commandBuffer,
                                                       uint32_t imageIndex,
                                                       PresentationCommandMode mode) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    if (mode == PresentationCommandMode::DisplayInterpolatedFrame) {
        displayNcnnFrameOnSwapchain(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    if (mode == PresentationCommandMode::DisplayCapturedSourceFrame) {
        displayCapturedNcnnSourceOnSwapchain(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    if (mode == PresentationCommandMode::DisplayHeldSourceFrame) {
        displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, ncnnPresentationState.ncnnHeldSourceDisplayIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    const uint32_t offscreenSlot = findAvailableOffscreenFrameSlot();
    if (offscreenSlot == UINT32_MAX) {
        throw std::runtime_error("offscreen frame history ring is exhausted!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = offscreenFramebuffers[offscreenSlot];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    VkClearValue clearValues[5]{};
    clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
    clearValues[1].color = { { 0.0f, 0.0f, 1.0f, 1.0f } };
    clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
    clearValues[3].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
    clearValues[4].depthStencil = { 1.0f, 0 };

    renderPassInfo.clearValueCount = 5;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    VkBuffer vertexBuffers[] = { vertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    uint32_t materialCount = static_cast<uint32_t>(std::max<size_t>(1, materials.size()));
    uint32_t setsPerFrame = materialCount + 1;

    for (const auto& sm : submeshes) { // per-material change
        uint32_t matIndex = std::min(sm.materialIndex, materialCount - 1);
        uint32_t dsIndex = currentFrame * setsPerFrame + matIndex;

        vkCmdBindDescriptorSets(commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1,
            &descriptorSets[dsIndex],
            0, nullptr); // bind to corresponding descriptor set

        vkCmdDrawIndexed(commandBuffer,
            sm.indexCount, 1,
            sm.indexOffset, 0, 0); // draw
    }

    if (rotatingCubeIndexCount > 0) {
        uint32_t cubeDescriptorSetIndex = currentFrame * setsPerFrame + materialCount;

        vkCmdBindDescriptorSets(commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1,
            &descriptorSets[cubeDescriptorSetIndex],
            0, nullptr);

        vkCmdDrawIndexed(commandBuffer,
            rotatingCubeIndexCount, 1,
            rotatingCubeIndexOffset, 0, 0);
    }

    if (!cesiumMan.indices.empty() &&
        skinnedVertexBuffer != VK_NULL_HANDLE &&
        skinnedIndexBuffer != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skinnedGraphicsPipeline);

        VkBuffer skinnedVertexBuffers[] = { skinnedVertexBuffer };
        VkDeviceSize skinnedOffsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, skinnedVertexBuffers, skinnedOffsets);
        vkCmdBindIndexBuffer(commandBuffer, skinnedIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

        uint32_t cesiumMaterialIndex = materialCount - 1;
        uint32_t cesiumDescriptorSetIndex = currentFrame * setsPerFrame + cesiumMaterialIndex;
        vkCmdBindDescriptorSets(commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1,
            &descriptorSets[cesiumDescriptorSetIndex],
            0, nullptr);

        vkCmdDrawIndexed(commandBuffer,
            static_cast<uint32_t>(cesiumMan.indices.size()), 1,
            0, 0, 0);
    }

    vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lightingPipeline);

    vkCmdBindDescriptorSets(commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        lightingPipelineLayout, 0, 1,
        &lightingDescriptorSets[offscreenSlot],
        0, nullptr); // bind lighting pipeline layout

    LightingPushConstants lightingPush{}; // info
    lightingPush.lightPos = glm::vec4(0.0f, 8.0f, 0.0f, 1.0f);
    lightingPush.lightColor = glm::vec4(1.0f, 0.95f, 0.9f, 1.0f);
    lightingPush.showNormals = showNormals ? 1.0f : 0.0f;
    lightingPush.showAlbedo = showAlbedo ? 1.0f : 0.0f;
    lightingPush.showPosition = showPosition ? 1.0f : 0.0f;
    lightingPush.showSpecular = showSpecular ? 1.0f : 0.0f;
    lightingPush.cameraPos = glm::vec4(cameraPos, 1.0f);

    vkCmdPushConstants(commandBuffer,
        lightingPipelineLayout,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(LightingPushConstants),
        &lightingPush);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    // Frame flow:
    //   scene render -> sampled offscreen image -> fused NCNN tensor preprocessor

#if HAS_NCNN
    if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled &&
        ncnnModelAttachedToRenderer &&
        ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
        // Render ahead without advancing presentation
        displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, ncnnPresentationState.ncnnLastPresentedSourceIndex);
        ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
        ncnnPresentationState.ncnnRenderAheadPending = true;
    }
    else
#endif
    {
        displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, offscreenSlot);
#if HAS_NCNN
        if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled && ncnnModelAttachedToRenderer) {
            ncnnPresentationState.ncnnLastPresentedSourceIndex = offscreenSlot;
        }
#endif
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    return offscreenSlot;
}

void VulkanNcnnRenderer::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(swapChainImages.size());
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }

    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render-finished semaphore for swapchain image!");
        }
    }
}

void VulkanNcnnRenderer::updateUniformBuffer(uint32_t currentImage) {
    glm::mat4 view = glm::lookAt(
        cameraPos,
        cameraPos + cameraFront,
        cameraUp
    );

    glm::mat4 proj = glm::perspective(glm::radians(45.0f),
        swapChainExtent.width / (float)swapChainExtent.height,
        0.1f,
        1000.0f);
    proj[1][1] *= -1;

    UniformBufferObject ubo{};
    ubo.model = glm::scale(glm::mat4(1.0f), glm::vec3(modelScale));
    ubo.view = view;
    ubo.proj = proj;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));

    glm::mat4 cubeModel = glm::translate(glm::mat4(1.0f), rotatingCubePosition);
    cubeModel = glm::rotate(cubeModel, elapsedTimeSeconds * glm::half_pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f));
    cubeModel = glm::rotate(cubeModel, elapsedTimeSeconds * glm::quarter_pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
    cubeModel = glm::scale(cubeModel, glm::vec3(0.7f));

    UniformBufferObject cubeUbo{};
    cubeUbo.model = cubeModel;
    cubeUbo.view = view;
    cubeUbo.proj = proj;

    memcpy(cubeUniformBuffersMapped[currentImage], &cubeUbo, sizeof(cubeUbo));

    glm::mat4 cesiumModel = glm::translate(glm::mat4(1.0f), CESIUM_MAN_SCENE_POSITION);
    cesiumModel = glm::scale(cesiumModel, glm::vec3(1.0f));

    UniformBufferObject cesiumUbo{};
    cesiumUbo.model = cesiumModel;
    cesiumUbo.view = view;
    cesiumUbo.proj = proj;

    memcpy(cesiumUniformBuffersMapped[currentImage], &cesiumUbo, sizeof(cesiumUbo));
}

bool VulkanNcnnRenderer::acquireFrame(uint32_t& imageIndex) {
    VkResult result = vkAcquireNextImageKHR(
        device,
        swapChain,
        UINT64_MAX,
        imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE,
        &imageIndex
    );

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return false;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    return true;
}

void VulkanNcnnRenderer::updateFrameState(uint32_t frameSlot) {
    vkWaitForFences(device, 1, &inFlightFences[frameSlot], VK_TRUE, UINT64_MAX);
    for (auto& output : ncnnOutputBuffers) {
        if (output.inUseByGraphics && output.graphicsFrameSlot == frameSlot) {
            output.inUseByGraphics = false;
            output.graphicsFrameSlot = UINT32_MAX;
        }
    }
    processCapturedFrameForSlot(frameSlot);

#if HAS_NCNN
    pollAsyncNcnnInference();
    if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled) {
        if (!submitAsyncNcnnInferenceIfReady() &&
            !ncnnPresentationState.ncnnInferenceInFlight &&
            !ncnnPresentationState.hasNcnnDisplayFrame &&
            !ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair) {
            ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair = true;
        }
    }
#endif
}

uint32_t VulkanNcnnRenderer::recordMainRenderCommands(uint32_t frameSlot,
                                                         uint32_t imageIndex,
                                                         PresentationCommandMode mode) {
    vkResetFences(device, 1, &inFlightFences[frameSlot]);
    vkResetCommandBuffer(commandBuffers[frameSlot], 0);
    return recordCommandBuffer(commandBuffers[frameSlot], imageIndex, mode);
}

void VulkanNcnnRenderer::submitGraphicsWork(uint32_t frameSlot, uint32_t imageIndex, uint32_t capturedNcnnSlot) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[frameSlot] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[frameSlot];

    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VkResult submitResult = VK_SUCCESS;
    {
        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        submitResult = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[frameSlot]);
    }
    if (submitResult != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    pendingCaptureSlotByFrame[frameSlot] = capturedNcnnSlot;
}

void VulkanNcnnRenderer::handlePresentation(uint32_t imageIndex) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[imageIndex];

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    VkResult result = VK_SUCCESS;
    {
        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
    }

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }
}

void VulkanNcnnRenderer::advanceFrameIndex() {
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanNcnnRenderer::drawFrame() {
    const uint32_t frameSlot = currentFrame;
    updateFrameState(frameSlot);

    uint32_t imageIndex;
    if (!acquireFrame(imageIndex)) {
        return;
    }

    const PresentationCommandMode mode =
        choosePresentationCommandMode(ncnnPresentationState, offscreenFrames.size());

    updateSkinAnimation(frameDeltaSeconds, frameSlot);

    if (mode == PresentationCommandMode::RenderFrame) {
        updateUniformBuffer(frameSlot);
    }

    // One scheduler tick records exactly one frame source, submits it, then queues exactly one present.
    const uint32_t capturedNcnnSlot = recordMainRenderCommands(frameSlot, imageIndex, mode);
    submitGraphicsWork(frameSlot, imageIndex, capturedNcnnSlot);
    handlePresentation(imageIndex);
    advanceFrameIndex();
}
