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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "VulkanNcnnRenderer.h"

namespace {

PresentationCommandMode choosePresentationCommandMode(const NcnnPresentationState& ncnnState,
                                                      size_t offscreenFrameCount,
                                                      bool canRenderSourceFrame) {
    // RenderFrame: render a new scene frame into the offscreen history, then present that source frame.
    PresentationCommandMode mode = PresentationCommandMode::RenderFrame;

    if (canRenderSourceFrame) {
        return mode;
    }

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
             ncnnState.ncnnRunningJobCount > 0 &&
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
constexpr glm::vec3 CESIUM_MAN_SCENE_POSITION = glm::vec3(3.0f, SPONZA_FLOOR_Y + 1.25f, 0.0f);
constexpr float CESIUM_MAN_WALK_RADIUS = 1.0f;
constexpr float CESIUM_MAN_WALK_SECONDS_PER_LOOP = 4.0f;
constexpr float MAX_SIMULATION_DELTA_SECONDS = 0.1f;
constexpr float ROTATING_CUBE_YAW_RADIANS_PER_SECOND = glm::half_pi<float>();
constexpr float ROTATING_CUBE_PITCH_RADIANS_PER_SECOND = glm::quarter_pi<float>();

std::chrono::steady_clock::duration secondsPerFrame(double framesPerSecond) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / framesPerSecond));
}

bool isUnsetTimePoint(std::chrono::steady_clock::time_point timePoint) {
    return timePoint.time_since_epoch() == std::chrono::steady_clock::duration::zero();
}

void advanceScheduledTime(std::chrono::steady_clock::time_point& scheduledTime,
                          std::chrono::steady_clock::duration interval,
                          std::chrono::steady_clock::time_point now) {
    scheduledTime += interval;
    while (scheduledTime <= now) {
        scheduledTime += interval;
    }
}

void advancePresentationTime(std::chrono::steady_clock::time_point& scheduledTime,
                             std::chrono::steady_clock::duration interval,
                             std::chrono::steady_clock::time_point now) {
    scheduledTime += interval;
    if (scheduledTime + interval < now) {
        scheduledTime = now;
    }
}

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
    initializeImGuiResources();
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
    initializeFrameProcessingImageLayouts();
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
    auto lastTime = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::steady_clock::now();
        float deltaTime =
            std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        lastTime = currentTime;
        deltaTime = std::min(deltaTime, MAX_SIMULATION_DELTA_SECONDS);

        glfwPollEvents();
        processInput(deltaTime);
        processMouseLook();
        frameDeltaSeconds += deltaTime;
        elapsedTimeSeconds += deltaTime;
        rotatingCubeYawRadians += ROTATING_CUBE_YAW_RADIANS_PER_SECOND * deltaTime;
        rotatingCubePitchRadians += ROTATING_CUBE_PITCH_RADIANS_PER_SECOND * deltaTime;
        drawFrame();
    }

    vkDeviceWaitIdle(device);
}

void VulkanNcnnRenderer::cleanup() {
#if HAS_NCNN
    waitForAsyncNcnnInference();
    shutdownNcnn();
#endif
    cleanupImGui();
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
    setDebugObjectName(VK_OBJECT_TYPE_COMMAND_POOL, reinterpret_cast<uint64_t>(commandPool), "Graphics Command Pool");
}

void VulkanNcnnRenderer::loadDebugUtilsFunctions() {
    vkSetDebugUtilsObjectNameEXTFn =
        reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    vkCmdBeginDebugUtilsLabelEXTFn =
        reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT"));
    vkCmdEndDebugUtilsLabelEXTFn =
        reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT"));
    vkQueueBeginDebugUtilsLabelEXTFn =
        reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(vkGetDeviceProcAddr(device, "vkQueueBeginDebugUtilsLabelEXT"));
    vkQueueEndDebugUtilsLabelEXTFn =
        reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(vkGetDeviceProcAddr(device, "vkQueueEndDebugUtilsLabelEXT"));
}

void VulkanNcnnRenderer::setDebugObjectName(VkObjectType objectType, uint64_t objectHandle, const std::string& name) {
    if (!vkSetDebugUtilsObjectNameEXTFn || objectHandle == 0 || name.empty()) {
        return;
    }

    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = objectHandle;
    nameInfo.pObjectName = name.c_str();
    vkSetDebugUtilsObjectNameEXTFn(device, &nameInfo);
}

void VulkanNcnnRenderer::beginDebugLabel(VkCommandBuffer commandBuffer, const std::string& name, const glm::vec4& color) {
    if (!vkCmdBeginDebugUtilsLabelEXTFn || commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.c_str();
    label.color[0] = color.r;
    label.color[1] = color.g;
    label.color[2] = color.b;
    label.color[3] = color.a;
    vkCmdBeginDebugUtilsLabelEXTFn(commandBuffer, &label);
}

void VulkanNcnnRenderer::endDebugLabel(VkCommandBuffer commandBuffer) {
    if (vkCmdEndDebugUtilsLabelEXTFn && commandBuffer != VK_NULL_HANDLE) {
        vkCmdEndDebugUtilsLabelEXTFn(commandBuffer);
    }
}

void VulkanNcnnRenderer::beginQueueDebugLabel(VkQueue queue, const std::string& name, const glm::vec4& color) {
    if (!vkQueueBeginDebugUtilsLabelEXTFn || queue == VK_NULL_HANDLE) {
        return;
    }

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.c_str();
    label.color[0] = color.r;
    label.color[1] = color.g;
    label.color[2] = color.b;
    label.color[3] = color.a;
    vkQueueBeginDebugUtilsLabelEXTFn(queue, &label);
}

void VulkanNcnnRenderer::endQueueDebugLabel(VkQueue queue) {
    if (vkQueueEndDebugUtilsLabelEXTFn && queue != VK_NULL_HANDLE) {
        vkQueueEndDebugUtilsLabelEXTFn(queue);
    }
}

void VulkanNcnnRenderer::copyOffscreenImageToSwapchain(VkCommandBuffer commandBuffer,
                                                              uint32_t imageIndex,
                                                              uint32_t offscreenSlot) {
    auto& source = offscreenFrames[offscreenSlot];
    const std::string copyLabel =
        "Copy Real Frame " + debugRealFrameLabel(source.debugFrameId) +
        " to Swapchain [Offscreen Slot " + std::to_string(offscreenSlot) + "]";
    beginDebugLabel(commandBuffer, copyLabel, glm::vec4(0.2f, 0.8f, 1.0f, 1.0f));

    // Offscreen history remains shader-readable between frames so NCNN can wrap it
    // directly. Each presentation copy temporarily promotes exactly one history
    // image to TRANSFER_SRC and restores the layout before the command buffer ends.
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
    endDebugLabel(commandBuffer);
}

void VulkanNcnnRenderer::resetFrameInterpolationDebugState() {
    nextDebugFrameId = 0;
    debugLastPresentedFrameLabel = "none";
    debugLastPresentedTimelineStep = -1;

    for (auto& frame : offscreenFrames) {
        frame.debugFrameId = UINT64_MAX;
        frame.debugPresented = false;
    }

    for (auto& output : ncnnOutputBuffers) {
        output.debugPreviousFrameId = UINT64_MAX;
        output.debugCurrentFrameId = UINT64_MAX;
    }
    ncnnInterpolationTargets.clear();
}

std::string VulkanNcnnRenderer::debugRealFrameLabel(uint64_t frameId) const {
    return frameId == UINT64_MAX ? "unknown" : std::to_string(frameId);
}

std::string VulkanNcnnRenderer::debugInterpolatedFrameLabel(uint64_t previousFrameId) const {
    return previousFrameId == UINT64_MAX ? "unknown.5" : std::to_string(previousFrameId) + ".5";
}

bool VulkanNcnnRenderer::isDebugRealFramePresented(uint64_t frameId) const {
    for (const auto& frame : offscreenFrames) {
        if (frame.debugFrameId == frameId) {
            return frame.debugPresented;
        }
    }

    return false;
}

uint32_t VulkanNcnnRenderer::findEarliestUnpresentedRealFrameSlot() const {
    uint32_t selectedSlot = UINT32_MAX;
    uint64_t selectedFrameId = UINT64_MAX;

    for (uint32_t slot = 0; slot < offscreenFrames.size(); ++slot) {
        const auto& frame = offscreenFrames[slot];
        if (frame.debugFrameId == UINT64_MAX || frame.debugPresented || frame.image == VK_NULL_HANDLE || frame.size == 0) {
            continue;
        }

        const int64_t timelineStep = static_cast<int64_t>(frame.debugFrameId * 2);
        if (timelineStep <= debugLastPresentedTimelineStep) {
            continue;
        }

        if (frame.debugFrameId < selectedFrameId) {
            selectedFrameId = frame.debugFrameId;
            selectedSlot = slot;
        }
    }

    return selectedSlot;
}

uint32_t VulkanNcnnRenderer::findReadyInterpolatedOutputForPreviousFrame(uint64_t previousFrameId) const {
    const uint32_t targetIndex = findInterpolationTargetIndex(previousFrameId);
    if (targetIndex >= ncnnInterpolationTargets.size()) {
        return UINT32_MAX;
    }

    const auto& target = ncnnInterpolationTargets[targetIndex];
    if (target.state != InterpolationTargetState::Ready ||
        target.outputIndex >= ncnnOutputBuffers.size()) {
        return UINT32_MAX;
    }

    const auto& output = ncnnOutputBuffers[target.outputIndex];
    if (output.ready &&
        !output.inUseByGraphics &&
        output.gpuBuffer != VK_NULL_HANDLE &&
        output.size != 0) {
        return target.outputIndex;
    }

    return UINT32_MAX;
}

const char* VulkanNcnnRenderer::interpolationTargetStateName(InterpolationTargetState state) const {
    switch (state) {
    case InterpolationTargetState::Pending:
        return "pending";
    case InterpolationTargetState::Running:
        return "running";
    case InterpolationTargetState::Ready:
        return "ready";
    case InterpolationTargetState::Presenting:
        return "presenting";
    case InterpolationTargetState::Released:
        return "released";
    case InterpolationTargetState::Dropped:
        return "dropped";
    default:
        return "unknown";
    }
}

uint32_t VulkanNcnnRenderer::findInterpolationTargetIndex(uint64_t previousFrameId) const {
    for (uint32_t i = 0; i < ncnnInterpolationTargets.size(); ++i) {
        if (ncnnInterpolationTargets[i].previousFrameId == previousFrameId) {
            return i;
        }
    }

    return UINT32_MAX;
}

uint32_t VulkanNcnnRenderer::findInterpolationTargetIndexForOutput(uint32_t outputIndex) const {
    for (uint32_t i = 0; i < ncnnInterpolationTargets.size(); ++i) {
        if (ncnnInterpolationTargets[i].outputIndex == outputIndex &&
            ncnnInterpolationTargets[i].state != InterpolationTargetState::Released &&
            ncnnInterpolationTargets[i].state != InterpolationTargetState::Dropped) {
            return i;
        }
    }

    return UINT32_MAX;
}

uint32_t VulkanNcnnRenderer::findOffscreenSlotForDebugFrame(uint64_t frameId) const {
    for (uint32_t slot = 0; slot < offscreenFrames.size(); ++slot) {
        if (offscreenFrames[slot].debugFrameId == frameId) {
            return slot;
        }
    }

    return UINT32_MAX;
}

void VulkanNcnnRenderer::createInterpolationTargetIfNeeded(uint32_t previousSourceIndex, uint32_t currentSourceIndex) {
    if (previousSourceIndex >= offscreenFrames.size() || currentSourceIndex >= offscreenFrames.size()) {
        return;
    }

    const uint64_t previousFrameId = offscreenFrames[previousSourceIndex].debugFrameId;
    const uint64_t currentFrameId = offscreenFrames[currentSourceIndex].debugFrameId;
    if (previousFrameId == UINT64_MAX || currentFrameId == UINT64_MAX) {
        return;
    }

    if (currentFrameId != previousFrameId + 1) {
        return;
    }

    if (findInterpolationTargetIndex(previousFrameId) != UINT32_MAX) {
        return;
    }

    FrameInterpolationTarget target{};
    target.previousFrameId = previousFrameId;
    target.currentFrameId = currentFrameId;
    target.previousSourceIndex = previousSourceIndex;
    target.currentSourceIndex = currentSourceIndex;
    target.state = InterpolationTargetState::Pending;
    target.waitingForFutureSource = false;
    ncnnInterpolationTargets.push_back(std::move(target));
}

uint32_t VulkanNcnnRenderer::createWaitingInterpolationTargetIfNeeded(uint64_t previousFrameId, uint32_t previousSourceIndex) {
    const uint32_t existingTargetIndex = findInterpolationTargetIndex(previousFrameId);
    if (existingTargetIndex != UINT32_MAX) {
        return existingTargetIndex;
    }
    if (previousSourceIndex >= offscreenFrames.size() ||
        offscreenFrames[previousSourceIndex].debugFrameId != previousFrameId) {
        return UINT32_MAX;
    }

    FrameInterpolationTarget target{};
    target.previousFrameId = previousFrameId;
    target.currentFrameId = previousFrameId + 1;
    target.previousSourceIndex = previousSourceIndex;
    target.currentSourceIndex = UINT32_MAX;
    target.state = InterpolationTargetState::Pending;
    target.waitingForFutureSource = true;
    target.reason = "waiting for future real source frame";
    ncnnInterpolationTargets.push_back(std::move(target));

    const uint32_t targetIndex = static_cast<uint32_t>(ncnnInterpolationTargets.size() - 1);
    return targetIndex;
}

void VulkanNcnnRenderer::updateWaitingInterpolationTargets() {
    for (auto& target : ncnnInterpolationTargets) {
        if (!target.waitingForFutureSource ||
            target.state != InterpolationTargetState::Pending ||
            target.previousFrameId == UINT64_MAX ||
            target.currentFrameId == UINT64_MAX) {
            continue;
        }

        const uint32_t previousSlot = findOffscreenSlotForDebugFrame(target.previousFrameId);
        const uint32_t currentSlot = findOffscreenSlotForDebugFrame(target.currentFrameId);
        if (previousSlot >= offscreenFrames.size() || currentSlot >= offscreenFrames.size()) {
            continue;
        }

        target.previousSourceIndex = previousSlot;
        target.currentSourceIndex = currentSlot;
        target.waitingForFutureSource = false;
        target.reason.clear();
    }
}

void VulkanNcnnRenderer::dropInterpolationTarget(uint32_t targetIndex, const std::string& reason) {
    if (targetIndex >= ncnnInterpolationTargets.size()) {
        return;
    }

    auto& target = ncnnInterpolationTargets[targetIndex];
    if (target.state == InterpolationTargetState::Presenting ||
        target.state == InterpolationTargetState::Released ||
        target.state == InterpolationTargetState::Dropped) {
        return;
    }

    if (target.outputIndex < ncnnOutputBuffers.size() &&
        !ncnnOutputBuffers[target.outputIndex].inUseByGraphics &&
        !ncnnOutputBuffers[target.outputIndex].inUseByInference) {
        auto& output = ncnnOutputBuffers[target.outputIndex];
        output.ready = false;
        output.debugPreviousFrameId = UINT64_MAX;
        output.debugCurrentFrameId = UINT64_MAX;
        output.sequence = 0;
        target.outputIndex = UINT32_MAX;
    }

    target.state = InterpolationTargetState::Dropped;
    target.reason = reason;
}

void VulkanNcnnRenderer::releaseObsoleteNcnnOutputBuffers() {
    for (uint32_t targetIndex = 0; targetIndex < ncnnInterpolationTargets.size(); ++targetIndex) {
        auto& target = ncnnInterpolationTargets[targetIndex];
        if (target.outputIndex >= ncnnOutputBuffers.size()) {
            continue;
        }
        if (target.state != InterpolationTargetState::Ready &&
            target.state != InterpolationTargetState::Dropped &&
            target.state != InterpolationTargetState::Presenting &&
            target.state != InterpolationTargetState::Released) {
            continue;
        }
        if (target.state == InterpolationTargetState::Ready &&
            static_cast<int64_t>(target.previousFrameId * 2 + 1) > debugLastPresentedTimelineStep) {
            continue;
        }

        auto& output = ncnnOutputBuffers[target.outputIndex];
        if (output.inUseByGraphics || output.inUseByInference) {
            continue;
        }

        output.ready = false;
        output.debugPreviousFrameId = UINT64_MAX;
        output.debugCurrentFrameId = UINT64_MAX;
        output.sequence = 0;
        target.state = InterpolationTargetState::Released;
        target.outputIndex = UINT32_MAX;
    }
}

void VulkanNcnnRenderer::markDebugRealFramePresented(uint32_t sourceIndex) {
    if (sourceIndex >= offscreenFrames.size()) {
        return;
    }

    auto& frame = offscreenFrames[sourceIndex];
    const bool alreadyPresented = frame.debugPresented;
    if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled &&
        !alreadyPresented &&
        frame.debugFrameId != UINT64_MAX &&
        frame.debugFrameId > 0) {
        const std::string expectedPreviousInterp = debugInterpolatedFrameLabel(frame.debugFrameId - 1);
        if (debugLastPresentedFrameLabel != expectedPreviousInterp &&
            debugLastPresentedFrameLabel != expectedPreviousInterp + "(dropped)" &&
            debugLastPresentedFrameLabel != "none") {
        }
    }

    frame.debugPresented = true;
    debugLastPresentedFrameLabel = debugRealFrameLabel(frame.debugFrameId);
    if (frame.debugFrameId != UINT64_MAX) {
        debugLastPresentedTimelineStep = static_cast<int64_t>(frame.debugFrameId * 2);
    }
}

void VulkanNcnnRenderer::markDebugInterpolatedFramePresented(uint64_t previousFrameId) {
    if (previousFrameId != UINT64_MAX && isDebugRealFramePresented(previousFrameId + 1)) {
    }

    debugLastPresentedFrameLabel = debugInterpolatedFrameLabel(previousFrameId);
    if (previousFrameId != UINT64_MAX) {
        debugLastPresentedTimelineStep = static_cast<int64_t>(previousFrameId * 2 + 1);
    }
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

    if (ncnnPresentationState.hasNcnnGpuFramePair) {
        createInterpolationTargetIfNeeded(
            ncnnPresentationState.previousNcnnGpuFrameIndex,
            ncnnPresentationState.currentNcnnGpuFrameIndex);
    }
    else {
    }
    updateWaitingInterpolationTargets();
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

    for (uint32_t i = 0; i < commandBuffers.size(); ++i) {
        setDebugObjectName(
            VK_OBJECT_TYPE_COMMAND_BUFFER,
            reinterpret_cast<uint64_t>(commandBuffers[i]),
            "Frame Command Buffer " + std::to_string(i));
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
        renderImGuiOverlay(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    if (mode == PresentationCommandMode::DisplayCapturedSourceFrame) {
        displayCapturedNcnnSourceOnSwapchain(commandBuffer, imageIndex);
        renderImGuiOverlay(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    if (mode == PresentationCommandMode::DisplayHeldSourceFrame) {
        displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, ncnnPresentationState.ncnnHeldSourceDisplayIndex);
        renderImGuiOverlay(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    const uint32_t offscreenSlot = findAvailableOffscreenFrameSlot();
    if (offscreenSlot == UINT32_MAX) {
        throw std::runtime_error("offscreen frame history ring is exhausted!");
    }

    const uint64_t realFrameId = nextDebugFrameId++;
    const std::string realFrameLabel =
        "Render Real Frame " + std::to_string(realFrameId) +
        " [Offscreen Slot " + std::to_string(offscreenSlot) + "]";
    beginDebugLabel(commandBuffer, realFrameLabel, glm::vec4(0.1f, 0.9f, 0.2f, 1.0f));
    const bool renderedFrameIsExpected =
        static_cast<int64_t>(realFrameId * 2) == debugLastPresentedTimelineStep + 1;
    offscreenFrames[offscreenSlot].debugFrameId = realFrameId;
    offscreenFrames[offscreenSlot].debugPresented = false;

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

    for (const auto& sm : submeshes) {
        uint32_t matIndex = std::min(sm.materialIndex, materialCount - 1);
        uint32_t dsIndex = currentFrame * setsPerFrame + matIndex;

        vkCmdBindDescriptorSets(commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1,
            &descriptorSets[dsIndex],
            0, nullptr);

        vkCmdDrawIndexed(commandBuffer,
            sm.indexCount, 1, sm.indexOffset, 0, 0);
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
        0, nullptr);

    LightingPushConstants lightingPush{};
    lightingPush.lightPos = glm::vec4(0.0f, 8.0f, 0.0f, 1.0f);
    lightingPush.lightColor = glm::vec4(1.0f, 0.95f, 0.9f, 1.0f);
    lightingPush.showNormals = showNormals ? 1.0f : 0.0f;
    lightingPush.showAlbedo = showAlbedo ? 1.0f : 0.0f;
    lightingPush.showPosition = showPosition ? 1.0f : 0.0f;
    lightingPush.showSpecular = showSpecular ? 1.0f : 0.0f;
    lightingPush.cameraPos = glm::vec4(cameraPos, 1.0f);
    const float panelWidth = std::min(260.0f, std::max(120.0f, static_cast<float>(swapChainExtent.width) * 0.28f));
    const float panelHeight = std::min(140.0f, std::max(72.0f, static_cast<float>(swapChainExtent.height) * 0.18f));
    const float panelTravelX = std::max(1.0f, static_cast<float>(swapChainExtent.width) - panelWidth - 24.0f);
    const float panelTravelY = std::max(1.0f, static_cast<float>(swapChainExtent.height) - panelHeight - 24.0f);
    const float panelX = 12.0f + panelTravelX * (0.5f + 0.5f * std::sin(elapsedTimeSeconds * 0.45f));
    const float panelY = 12.0f + panelTravelY * (0.5f + 0.5f * std::sin(elapsedTimeSeconds * 0.31f + 1.2f));
    lightingPush.debugPanelRect = glm::vec4(panelX, panelY, panelWidth, panelHeight);
    lightingPush.debugPanelEnabled = showInterpolationDebugPanel ? 1.0f : 0.0f;
    lightingPush.debugPanelFrameId = static_cast<float>(realFrameId);

    vkCmdPushConstants(commandBuffer,
        lightingPipelineLayout,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(LightingPushConstants),
        &lightingPush);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);
    endDebugLabel(commandBuffer);

#if HAS_NCNN
    if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled && ncnnModelAttachedToRenderer) {
        const int64_t expectedTimelineStep = debugLastPresentedTimelineStep + 1;
        const bool pendingInterpIsExpected =
            ncnnPresentationState.hasNcnnDisplayFrame &&
            ncnnPresentationState.ncnnPendingInterpolatedOutputIndex < ncnnOutputBuffers.size() &&
            static_cast<int64_t>(
                ncnnOutputBuffers[ncnnPresentationState.ncnnPendingInterpolatedOutputIndex].debugPreviousFrameId * 2 + 1) ==
                expectedTimelineStep;
        const bool pendingSourceIsExpected =
            ncnnPresentationState.ncnnPendingSourceDisplayIndex < offscreenFrames.size() &&
            offscreenFrames[ncnnPresentationState.ncnnPendingSourceDisplayIndex].debugFrameId != UINT64_MAX &&
            static_cast<int64_t>(
                offscreenFrames[ncnnPresentationState.ncnnPendingSourceDisplayIndex].debugFrameId * 2) ==
                expectedTimelineStep;
        if (renderedFrameIsExpected) {
            displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, offscreenSlot);
            ncnnPresentationState.ncnnLastPresentedSourceIndex = offscreenSlot;
            ncnnPresentationState.ncnnHeldSourceDisplayIndex = UINT32_MAX;
            ncnnPresentationState.ncnnRenderAheadPending = false;
        }
        else if (pendingInterpIsExpected) {
            displayNcnnFrameOnSwapchain(commandBuffer, imageIndex);
        }
        else if (pendingSourceIsExpected) {
            displayCapturedNcnnSourceOnSwapchain(commandBuffer, imageIndex);
        }
        else if (ncnnPresentationState.hasNcnnDisplayFrame) {
            if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
                displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, ncnnPresentationState.ncnnLastPresentedSourceIndex);
                ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
                ncnnPresentationState.ncnnRenderAheadPending = true;
            }
            else {
                displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, offscreenSlot);
                ncnnPresentationState.ncnnLastPresentedSourceIndex = offscreenSlot;
            }
        }
        else if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
            // Render ahead without advancing presentation.
            displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, ncnnPresentationState.ncnnLastPresentedSourceIndex);
            ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
            ncnnPresentationState.ncnnRenderAheadPending = true;
        }
        else {
            displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, offscreenSlot);
            ncnnPresentationState.ncnnLastPresentedSourceIndex = offscreenSlot;
        }
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

    renderImGuiOverlay(commandBuffer, imageIndex);

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
        setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE, reinterpret_cast<uint64_t>(imageAvailableSemaphores[i]),
            "Image Available Semaphore Frame " + std::to_string(i));
        setDebugObjectName(VK_OBJECT_TYPE_FENCE, reinterpret_cast<uint64_t>(inFlightFences[i]),
            "In Flight Fence Frame " + std::to_string(i));
    }

    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render-finished semaphore for swapchain image!");
        }
        setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE, reinterpret_cast<uint64_t>(renderFinishedSemaphores[i]),
            "Render Finished Semaphore Swapchain Image " + std::to_string(i));
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
    cubeModel = glm::rotate(cubeModel, rotatingCubeYawRadians, glm::vec3(0.0f, 1.0f, 0.0f));
    cubeModel = glm::rotate(cubeModel, rotatingCubePitchRadians, glm::vec3(1.0f, 0.0f, 0.0f));
    cubeModel = glm::scale(cubeModel, glm::vec3(0.7f));

    UniformBufferObject cubeUbo{};
    cubeUbo.model = cubeModel;
    cubeUbo.view = view;
    cubeUbo.proj = proj;

    memcpy(cubeUniformBuffersMapped[currentImage], &cubeUbo, sizeof(cubeUbo));

    float walkAngle = elapsedTimeSeconds * glm::two_pi<float>() / CESIUM_MAN_WALK_SECONDS_PER_LOOP;
    glm::vec3 cesiumPosition = CESIUM_MAN_SCENE_POSITION +
        glm::vec3(std::cos(walkAngle) * CESIUM_MAN_WALK_RADIUS,
                  0.0f,
                  std::sin(walkAngle) * CESIUM_MAN_WALK_RADIUS);
    glm::vec3 walkTangent = glm::normalize(glm::vec3(-std::sin(walkAngle), 0.0f, std::cos(walkAngle)));
    float facingAngle = std::atan2(walkTangent.x, walkTangent.z);

    glm::mat4 cesiumModel = glm::translate(glm::mat4(1.0f), cesiumPosition);
    cesiumModel = glm::rotate(cesiumModel, facingAngle, glm::vec3(0.0f, 1.0f, 0.0f));
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
    for (uint32_t outputIndex = 0; outputIndex < ncnnOutputBuffers.size(); ++outputIndex) {
        auto& output = ncnnOutputBuffers[outputIndex];
        if (output.inUseByGraphics && output.graphicsFrameSlot == frameSlot) {
            output.inUseByGraphics = false;
            output.graphicsFrameSlot = UINT32_MAX;
        }
    }
    releaseObsoleteNcnnOutputBuffers();
    processCapturedFrameForSlot(frameSlot);
    updateWaitingInterpolationTargets();

#if HAS_NCNN
    pollAsyncNcnnInference();
    if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled) {
        if (!submitAsyncNcnnInferenceIfReady() &&
            ncnnPresentationState.ncnnRunningJobCount == 0 &&
            !ncnnPresentationState.hasNcnnDisplayFrame &&
            !ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair) {
            ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair = true;
        }
    }
#endif
}

void VulkanNcnnRenderer::setBenchmarkModeEnabled(bool enabled) {
    if (benchmarkModeEnabled == enabled) {
        return;
    }

    benchmarkModeEnabled = enabled;
    const auto now = std::chrono::steady_clock::now();
    benchmarkNextPresentTime = now;
    benchmarkNextRealFrameTime = now;

    std::cout << "[BENCH] 30 FPS source lock "
              << (benchmarkModeEnabled ? "enabled" : "disabled")
              << std::endl;
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
        const std::string submitLabel = "Queue Submit Logical Frame " + debugLastPresentedFrameLabel;
        beginQueueDebugLabel(graphicsQueue, submitLabel, glm::vec4(0.2f, 1.0f, 0.4f, 1.0f));
        submitResult = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[frameSlot]);
        endQueueDebugLabel(graphicsQueue);
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

    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
        recordPresentedFrameStats(pendingPresentedFrameKind);
    }
    pendingPresentedFrameKind = PresentedFrameKind::None;

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }
}

void VulkanNcnnRenderer::recordPresentedFrameStats(PresentedFrameKind frameKind) {
    if (frameKind == PresentedFrameKind::None) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (isUnsetTimePoint(fpsStatsWindowStart)) {
        fpsStatsWindowStart = now;
    }

    ++fpsPresentedFrameCount;
    if (frameKind == PresentedFrameKind::Real) {
        ++fpsRealFrameCount;
    }
    else if (frameKind == PresentedFrameKind::Interpolated) {
        ++fpsInterpolatedFrameCount;
    }

    const std::chrono::duration<float> elapsed = now - fpsStatsWindowStart;
    if (elapsed.count() < 1.0f) {
        return;
    }

    displayedPresentedFps = static_cast<float>(fpsPresentedFrameCount) / elapsed.count();
    displayedRealFps = static_cast<float>(fpsRealFrameCount) / elapsed.count();
    displayedInterpolatedFps = static_cast<float>(fpsInterpolatedFrameCount) / elapsed.count();
    fpsPresentedFrameCount = 0;
    fpsRealFrameCount = 0;
    fpsInterpolatedFrameCount = 0;
    fpsStatsWindowStart = now;
}

void VulkanNcnnRenderer::advanceFrameIndex() {
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanNcnnRenderer::drawFrame() {
    bool useChronologicalInterpolation = false;
#if HAS_NCNN
    useChronologicalInterpolation =
        ncnnPresentationState.ncnnRealtimeInterpolationEnabled && ncnnModelAttachedToRenderer;
#endif

    const auto schedulerNow = std::chrono::steady_clock::now();
    const auto realFrameInterval = secondsPerFrame(30.0);
    const auto presentationInterval = secondsPerFrame(useChronologicalInterpolation ? 60.0 : 30.0);
    bool benchmarkCanRenderSourceFrame = true;
    if (benchmarkModeEnabled) {
        if (isUnsetTimePoint(benchmarkNextPresentTime)) {
            benchmarkNextPresentTime = schedulerNow;
        }
        if (isUnsetTimePoint(benchmarkNextRealFrameTime)) {
            benchmarkNextRealFrameTime = schedulerNow;
        }

        if (schedulerNow < benchmarkNextPresentTime) {
            return;
        }

        benchmarkCanRenderSourceFrame = schedulerNow >= benchmarkNextRealFrameTime;
        if (!useChronologicalInterpolation && !benchmarkCanRenderSourceFrame) {
            return;
        }
    }

    const uint32_t frameSlot = currentFrame;
    updateFrameState(frameSlot);

    uint32_t imageIndex;
    if (!acquireFrame(imageIndex)) {
        return;
    }

    const bool canRenderSourceFrame =
        benchmarkCanRenderSourceFrame && findAvailableOffscreenFrameSlot() != UINT32_MAX;
    PresentationCommandMode mode = PresentationCommandMode::RenderFrame;
    const int64_t expectedTimelineStep = debugLastPresentedTimelineStep + 1;
    const bool expectedIsReal = (expectedTimelineStep % 2) == 0;

    if (!useChronologicalInterpolation) {
        mode = choosePresentationCommandMode(ncnnPresentationState, offscreenFrames.size(), canRenderSourceFrame);
    }
    else if (expectedIsReal) {
        const uint64_t expectedFrameId = static_cast<uint64_t>(expectedTimelineStep / 2);
        const uint32_t pendingRealSlot = findEarliestUnpresentedRealFrameSlot();
        if (pendingRealSlot < offscreenFrames.size() &&
            offscreenFrames[pendingRealSlot].debugFrameId == expectedFrameId) {
            ncnnPresentationState.ncnnPendingSourceDisplayIndex = pendingRealSlot;
            mode = PresentationCommandMode::DisplayCapturedSourceFrame;
        }
        else if (canRenderSourceFrame && nextDebugFrameId == expectedFrameId) {
            mode = PresentationCommandMode::RenderFrame;
        }
        else if (ncnnPresentationState.ncnnHeldSourceDisplayIndex < offscreenFrames.size()) {
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
        }
        else if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
            ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
        }
        else {
            mode = PresentationCommandMode::RenderFrame;
        }
    }
    else {
        const uint64_t expectedPreviousFrameId = static_cast<uint64_t>(expectedTimelineStep / 2);
        uint32_t expectedTargetIndex = findInterpolationTargetIndex(expectedPreviousFrameId);
        if (expectedTargetIndex >= ncnnInterpolationTargets.size()) {
            const uint32_t previousSlot = findOffscreenSlotForDebugFrame(expectedPreviousFrameId);
            const uint32_t currentSlot = findOffscreenSlotForDebugFrame(expectedPreviousFrameId + 1);
            if (previousSlot < offscreenFrames.size() && currentSlot < offscreenFrames.size()) {
                createInterpolationTargetIfNeeded(previousSlot, currentSlot);
                expectedTargetIndex = findInterpolationTargetIndex(expectedPreviousFrameId);
            }
            else if (previousSlot < offscreenFrames.size() &&
                     currentSlot >= offscreenFrames.size() &&
                     nextDebugFrameId == expectedPreviousFrameId + 1) {
                expectedTargetIndex = createWaitingInterpolationTargetIfNeeded(expectedPreviousFrameId, previousSlot);
                mode = canRenderSourceFrame ? PresentationCommandMode::RenderFrame : PresentationCommandMode::DisplayHeldSourceFrame;
                if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
                    ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
                }
            }
            else {
                debugLastPresentedFrameLabel = debugInterpolatedFrameLabel(expectedPreviousFrameId) + "(dropped)";
                debugLastPresentedTimelineStep = expectedTimelineStep;
                mode = PresentationCommandMode::DisplayHeldSourceFrame;
                if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
                    ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
                }
            }
        }

        if (mode != PresentationCommandMode::DisplayHeldSourceFrame &&
            expectedTargetIndex < ncnnInterpolationTargets.size() &&
            ncnnInterpolationTargets[expectedTargetIndex].state == InterpolationTargetState::Dropped) {
            debugLastPresentedFrameLabel = debugInterpolatedFrameLabel(expectedPreviousFrameId) + "(dropped)";
            debugLastPresentedTimelineStep = expectedTimelineStep;
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
            if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
                ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
            }
        }
        const uint32_t outputIndex = findReadyInterpolatedOutputForPreviousFrame(expectedPreviousFrameId);
        if (mode != PresentationCommandMode::DisplayHeldSourceFrame && outputIndex != UINT32_MAX) {
            ncnnPresentationState.ncnnPendingInterpolatedOutputIndex = outputIndex;
            ncnnPresentationState.hasNcnnDisplayFrame = true;
            mode = PresentationCommandMode::DisplayInterpolatedFrame;
        }
        else if (mode != PresentationCommandMode::DisplayHeldSourceFrame && canRenderSourceFrame) {
            mode = PresentationCommandMode::RenderFrame;
        }
        else if (mode != PresentationCommandMode::DisplayHeldSourceFrame &&
                 ncnnPresentationState.ncnnHeldSourceDisplayIndex < offscreenFrames.size()) {
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
        }
        else if (mode != PresentationCommandMode::DisplayHeldSourceFrame &&
                 ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
            ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
        }
        else if (mode != PresentationCommandMode::DisplayHeldSourceFrame) {
            mode = PresentationCommandMode::RenderFrame;
        }
    }

    if (useChronologicalInterpolation &&
        canRenderSourceFrame &&
        mode != PresentationCommandMode::RenderFrame) {
        mode = PresentationCommandMode::RenderFrame;
    }
    else if (benchmarkModeEnabled &&
             !benchmarkCanRenderSourceFrame &&
             mode == PresentationCommandMode::RenderFrame) {
        if (ncnnPresentationState.ncnnHeldSourceDisplayIndex < offscreenFrames.size()) {
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
        }
        else if (ncnnPresentationState.ncnnLastPresentedSourceIndex < offscreenFrames.size()) {
            ncnnPresentationState.ncnnHeldSourceDisplayIndex = ncnnPresentationState.ncnnLastPresentedSourceIndex;
            mode = PresentationCommandMode::DisplayHeldSourceFrame;
        }
    }

    if (mode == PresentationCommandMode::RenderFrame) {
        updateSkinAnimation(frameDeltaSeconds, frameSlot);
        updateUniformBuffer(frameSlot);
        frameDeltaSeconds = 0.0f;
    }

    beginImGuiFrame();

    // One scheduler tick records exactly one frame source, submits it, then queues exactly one present.
    const uint32_t capturedNcnnSlot = recordMainRenderCommands(frameSlot, imageIndex, mode);
    submitGraphicsWork(frameSlot, imageIndex, capturedNcnnSlot);
    handlePresentation(imageIndex);

    if (benchmarkModeEnabled) {
        const auto afterPresent = std::chrono::steady_clock::now();
        advancePresentationTime(benchmarkNextPresentTime, presentationInterval, afterPresent);
        if (capturedNcnnSlot != UINT32_MAX) {
            advanceScheduledTime(benchmarkNextRealFrameTime, realFrameInterval, afterPresent);
        }
    }

    advanceFrameIndex();
}
