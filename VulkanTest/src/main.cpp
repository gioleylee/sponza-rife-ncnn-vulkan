#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/vector3.h>
#include <assimp/types.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <ratio>

#if __has_include(<ncnn/net.h>) && __has_include(<ncnn/gpu.h>) && __has_include(<ncnn/layer.h>)
#include <ncnn/net.h>
#include <ncnn/gpu.h>
// #include <ncnn/allocator.h>
// #include <ncnn/layer.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

#include "AppTypes.h"
#include "RifeRunner.h"
#include "validation_layers.h"

#include "VulkanRifeRendererApp.h"

void VulkanRifeRendererApp::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void VulkanRifeRendererApp::initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createFrameProcessingResources();
    createRenderPass();
    createDepthResources();
    createGBufferAttachments();
    createDescriptorSetLayout();
    createLightingDescriptorSetLayout();
    createGraphicsPipeline();
    createLightingPipeline();
    createFramebuffers();
    createCommandPool();
    createFallbackTexture();
    updateCameraFrontFromAngles();
    rotatingCubePosition = cameraPos + cameraFront * 3.0f;
    loadModel("assets/sponza/sponza.obj");
    appendRotatingCubeGeometry();
    createVertexBuffer();
    createIndexBuffer();
    loadMaterialTextures();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createLightingDescriptorPool();
    createLightingDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
#if HAS_NCNN
    initNcnn();
    tryLoadDefaultNcnnModel();
#endif
}

void VulkanRifeRendererApp::mainLoop() {
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime =
            std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        lastTime = currentTime;

        glfwPollEvents();
        processInput(deltaTime);
        processMouseLook();
        elapsedTimeSeconds += deltaTime;
        drawFrame();
    }

    vkDeviceWaitIdle(device);
}

void VulkanRifeRendererApp::cleanup() {
#if HAS_NCNN
    waitForAsyncRifeInference();
    shutdownNcnn();
#endif
    cleanupSwapChain();

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipeline(device, lightingPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, lightingPipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        vkDestroyBuffer(device, cubeUniformBuffers[i], nullptr);
        vkFreeMemory(device, cubeUniformBuffersMemory[i], nullptr);
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

#if HAS_NCNN
int VulkanRifeRendererApp::findNcnnDeviceIndexForRenderer() const {
    VkPhysicalDeviceProperties rendererProperties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &rendererProperties);

    const int gpuCount = ncnn::get_gpu_count();
    int vendorMatch = -1;
    for (int i = 0; i < gpuCount; ++i) {
        const ncnn::GpuInfo& gpuInfo = ncnn::get_gpu_info(i);

        if (gpuInfo.vendor_id() == rendererProperties.vendorID &&
            gpuInfo.device_id() == rendererProperties.deviceID) {
            if (std::strcmp(gpuInfo.device_name(), rendererProperties.deviceName) != 0) {
                std::cout << "[NCNN] matched renderer GPU by vendor/device id: "
                          << rendererProperties.deviceName << " -> index " << i << std::endl;
            }
            return i;
        }

        if (vendorMatch < 0 && gpuInfo.vendor_id() == rendererProperties.vendorID) {
            vendorMatch = i;
        }
    }

    if (vendorMatch >= 0) {
        std::cout << "[NCNN] exact renderer GPU id not found; using same-vendor GPU index "
                  << vendorMatch << " for " << rendererProperties.deviceName << std::endl;
        return vendorMatch;
    }

    std::cout << "[NCNN] renderer GPU was not found in NCNN's Vulkan list; falling back to index 0" << std::endl;
    return gpuCount > 0 ? 0 : -1;
}

void VulkanRifeRendererApp::applyNcnnVulkanOptions() {
    const int gpuCount = ncnn::get_gpu_count();
    const bool hasSelectedGpu =
        ncnnRendererDeviceIndex >= 0 && ncnnRendererDeviceIndex < gpuCount;
    const ncnn::GpuInfo* gpuInfo =
        hasSelectedGpu ? &ncnn::get_gpu_info(ncnnRendererDeviceIndex) : nullptr;

    net.opt.use_vulkan_compute = hasSelectedGpu && HAS_RIFE_WARP_VK;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_packing_layout = true;

    net.opt.use_cooperative_matrix = true;
}

void VulkanRifeRendererApp::initNcnn() {
    if (ncnnInitialized) {
        return;
    }

    ncnn::create_gpu_instance();

    const int gpuCount = ncnn::get_gpu_count();
    ncnnRendererDeviceIndex = findNcnnDeviceIndexForRenderer();
    applyNcnnVulkanOptions();
    net.opt.num_threads = 1;
    if (gpuCount > 0 && !HAS_RIFE_WARP_VK) {
        std::cout << "[NCNN] rife warp vulkan shaders not found; forcing CPU inference path" << std::endl;
    }
    ncnnInitialized = true;

    std::cout << "[NCNN] initialized (gpu_count=" << gpuCount
              << ", renderer_gpu_index=" << ncnnRendererDeviceIndex
              << ", vulkan_compute=" << (net.opt.use_vulkan_compute ? "on" : "off") << ")" << std::endl;
}

void VulkanRifeRendererApp::shutdownNcnn() {
    if (!ncnnInitialized) {
        return;
    }

    net.clear();
    rifeRunner.reset();
    ncnn::destroy_gpu_instance();

    ncnnInitialized = false;
    ncnnRendererDeviceIndex = -1;
    ncnnModelLoaded = false;
    rifeModelAttachedToRenderer = false;
}

bool VulkanRifeRendererApp::loadNcnnModel(const std::string& paramPath, const std::string& binPath) {
    if (!ncnnInitialized) {
        initNcnn();
    }

    // NCNN consumes these options while loading the model and creating its
    // Vulkan pipelines. Reapply them before every load.
    applyNcnnVulkanOptions();

    if (!rifeRunner.initialize(net, paramPath, binPath, device, ncnnRendererDeviceIndex)) {
        return false;
    }

    ncnnModelLoaded = true;

    std::cout << "[NCNN] model loaded: " << paramPath << " + " << binPath << std::endl;
    return true;
}

void VulkanRifeRendererApp::tryLoadDefaultNcnnModel() {
    const std::string optimizedParam = "assets/models/rife-v4/flownet-opt.param";
    const std::string optimizedBin = "assets/models/rife-v4/flownet-opt.bin";
    const std::string defaultParam = "assets/models/rife-v4/flownet.param";
    const std::string defaultBin = "assets/models/rife-v4/flownet.bin";

    const bool hasOptimizedModel =
        std::filesystem::exists(optimizedParam) && std::filesystem::exists(optimizedBin);
    const std::string& paramPath = hasOptimizedModel ? optimizedParam : defaultParam;
    const std::string& binPath = hasOptimizedModel ? optimizedBin : defaultBin;

    if (!std::filesystem::exists(paramPath) || !std::filesystem::exists(binPath)) {
        std::cout << "[NCNN] default RIFE model not found: "
                  << paramPath << " and " << binPath << std::endl;
        return;
    }

    std::cout << "[NCNN] using " << (hasOptimizedModel ? "output" : "original model")
              << ": " << paramPath << " + " << binPath << std::endl;

    if (loadNcnnModel(paramPath, binPath)) {
        ncnn::Extractor extractor = net.create_extractor();
        (void)extractor;
        std::cout << "[NCNN] extractor created successfully" << std::endl;
        rifeModelAttachedToRenderer = rifeRunner.isReady();
        if (rifeModelAttachedToRenderer) {
            std::cout << "[RIFE] model attached to Vulkan renderer" << std::endl;
        }
    }
    else {
        std::cout << "[NCNN] failed to load default RIFE model" << std::endl;
    }
}

#endif

void VulkanRifeRendererApp::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

bool VulkanRifeRendererApp::createExportableFrameBuffer(VkDeviceSize size,
                                 VkBuffer& buffer,
                                 VkDeviceMemory& bufferMemory,
                                 HANDLE& externalHandle) {
    buffer = VK_NULL_HANDLE;
    bufferMemory = VK_NULL_HANDLE;
    externalHandle = nullptr;

    VkExternalMemoryBufferCreateInfo externalBufferInfo{};
    externalBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    externalBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.pNext = &externalBufferInfo;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    uint32_t memoryTypeIndex = 0;
    if (!tryFindMemoryType(
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memoryTypeIndex)) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportInfo;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(device, buffer, bufferMemory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        bufferMemory = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory = bufferMemory;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    if (vkGetMemoryWin32HandleKHRFn(device, &handleInfo, &externalHandle) != VK_SUCCESS || !externalHandle) {
        vkFreeMemory(device, bufferMemory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        bufferMemory = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void VulkanRifeRendererApp::createFrameProcessingResources() {
    cleanupFrameProcessingResources();

    const VkDeviceSize frameSize =
        static_cast<VkDeviceSize>(swapChainExtent.width) *
        static_cast<VkDeviceSize>(swapChainExtent.height) * 4;

    offscreenFrames.resize(OFFSCREEN_FRAME_HISTORY_COUNT);
    rifeOutputBuffers.resize(RIFE_OUTPUT_BUFFER_COUNT);

    rifeDisplayBufferSize = frameSize;
    hasRifeDisplayFrame = false;
    nextRifeOutputSequence = 1;
    rifePendingInterpolatedOutputIndex = UINT32_MAX;
    rifePendingSourceDisplayIndex = UINT32_MAX;
    rifeHeldSourceDisplayIndex = UINT32_MAX;
    rifeLastPresentedSourceIndex = UINT32_MAX;
    rifeRenderAheadPending = false;

    const auto toUnormFormat = [](VkFormat format) {
        switch (format) {
        case VK_FORMAT_B8G8R8A8_SRGB:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case VK_FORMAT_R8G8B8A8_SRGB:
            return VK_FORMAT_R8G8B8A8_UNORM;
        default:
            return format;
        }
    };
    const VkFormat rifeInputFormat = toUnormFormat(swapChainImageFormat);
    const VkImageCreateFlags offscreenFlags =
        rifeInputFormat != swapChainImageFormat ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;

    for (auto& frame : offscreenFrames) {
        createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            1,
            swapChainImageFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            frame.image,
            frame.imageMemory,
            offscreenFlags
        );
        frame.imageView = createImageView(frame.image, swapChainImageFormat, 1);
        frame.rifeInputImageView = createImageView(frame.image, rifeInputFormat, 1);

        frame.size = frameSize;
    }

    for (auto& output : rifeOutputBuffers) {
        createBuffer(
            frameSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            output.gpuBuffer,
            output.gpuMemory
        );

        output.size = frameSize;
        output.ready = false;
        output.inUseByInference = false;
        output.inUseByGraphics = false;
        output.graphicsFrameSlot = UINT32_MAX;
        output.sequence = 0;
    }

    pendingCaptureSlotByFrame.fill(UINT32_MAX);
    hasRifeGpuFramePair = false;
    currentRifeGpuFrameIndex = UINT32_MAX;
    previousRifeGpuFrameIndex = UINT32_MAX;
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
}

void VulkanRifeRendererApp::cleanupFrameProcessingResources() {
#if HAS_NCNN
    waitForAsyncRifeInference();
#endif

    for (auto& output : rifeOutputBuffers) {
        if (output.gpuBuffer) {
            vkDestroyBuffer(device, output.gpuBuffer, nullptr);
            output.gpuBuffer = VK_NULL_HANDLE;
        }

        if (output.gpuMemory) {
            vkFreeMemory(device, output.gpuMemory, nullptr);
            output.gpuMemory = VK_NULL_HANDLE;
        }

        output.size = 0;
        output.ready = false;
        output.inUseByInference = false;
        output.inUseByGraphics = false;
        output.graphicsFrameSlot = UINT32_MAX;
        output.sequence = 0;
    }

    rifeOutputBuffers.clear();
    rifeDisplayBufferSize = 0;
    hasRifeDisplayFrame = false;
    nextRifeOutputSequence = 1;
    rifePendingInterpolatedOutputIndex = UINT32_MAX;
    rifePendingSourceDisplayIndex = UINT32_MAX;
    rifeHeldSourceDisplayIndex = UINT32_MAX;
    rifeLastPresentedSourceIndex = UINT32_MAX;
    rifeRenderAheadPending = false;
#if HAS_NCNN
    rifeInferenceInFlight = false;
    asyncRifePrevFrameIndex = UINT32_MAX;
    asyncRifeCurrFrameIndex = UINT32_MAX;
    asyncRifeOutputIndex = UINT32_MAX;
    rifeInferenceScaleDivisor = RIFE_INITIAL_INFERENCE_SCALE_DIVISOR;
    rifeCompletedInferenceCount = 0;
#endif

    for (auto& frame : offscreenFrames) {
        if (frame.gpuBuffer) {
            vkDestroyBuffer(device, frame.gpuBuffer, nullptr);
            frame.gpuBuffer = VK_NULL_HANDLE;
        }

        if (frame.gpuMemory) {
            vkFreeMemory(device, frame.gpuMemory, nullptr);
            frame.gpuMemory = VK_NULL_HANDLE;
        }

        if (frame.imageView) {
            vkDestroyImageView(device, frame.imageView, nullptr);
            frame.imageView = VK_NULL_HANDLE;
        }

        if (frame.rifeInputImageView) {
            vkDestroyImageView(device, frame.rifeInputImageView, nullptr);
            frame.rifeInputImageView = VK_NULL_HANDLE;
        }

        if (frame.image) {
            vkDestroyImage(device, frame.image, nullptr);
            frame.image = VK_NULL_HANDLE;
        }

        if (frame.imageMemory) {
            vkFreeMemory(device, frame.imageMemory, nullptr);
            frame.imageMemory = VK_NULL_HANDLE;
        }

        frame.size = 0;
    }

    offscreenFrames.clear();
    pendingCaptureSlotByFrame.fill(UINT32_MAX);
    hasRifeGpuFramePair = false;
    currentRifeGpuFrameIndex = UINT32_MAX;
    previousRifeGpuFrameIndex = UINT32_MAX;
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
}

uint32_t VulkanRifeRendererApp::findAvailableOffscreenFrameSlot() const {
    for (uint32_t slot = 0; slot < offscreenFrames.size(); ++slot) {
#if HAS_NCNN
        if (rifeInferenceInFlight &&
            (slot == asyncRifePrevFrameIndex || slot == asyncRifeCurrFrameIndex)) {
            continue;
        }
#endif
        if (slot == currentRifeGpuFrameIndex) {
            continue;
        }
        if (slot == rifePendingSourceDisplayIndex) {
            continue;
        }
        if (slot == rifeHeldSourceDisplayIndex) {
            continue;
        }
        if (slot == rifeLastPresentedSourceIndex) {
            continue;
        }

        bool pending = false;
        for (uint32_t pendingSlot : pendingCaptureSlotByFrame) {
            if (pendingSlot == slot) {
                pending = true;
                break;
            }
        }

        if (!pending) {
            return slot;
        }
    }

    return UINT32_MAX;
}

void VulkanRifeRendererApp::copyOffscreenImageToSwapchain(VkCommandBuffer commandBuffer,
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

void VulkanRifeRendererApp::copyRifeBufferToSwapchain(VkCommandBuffer commandBuffer,
                                                         uint32_t imageIndex,
                                                         VkBuffer sourceBuffer,
                                                         VkAccessFlags sourceAccessMask,
                                                         VkPipelineStageFlags sourceStageMask) {
    VkImageMemoryBarrier toTransferDstBarrier{};
    toTransferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // Presentation overwrites the entire acquired image, so its old contents
    // are intentionally discarded. This also handles a swapchain image's first
    // use without treating it as a render target.
    toTransferDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstBarrier.image = swapChainImages[imageIndex];
    toTransferDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferDstBarrier.subresourceRange.baseMipLevel = 0;
    toTransferDstBarrier.subresourceRange.levelCount = 1;
    toTransferDstBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferDstBarrier.subresourceRange.layerCount = 1;
    toTransferDstBarrier.srcAccessMask = 0;
    toTransferDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toTransferDstBarrier
    );

    VkBufferMemoryBarrier rifeOutputBarrier{};
    rifeOutputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    rifeOutputBarrier.srcAccessMask = sourceAccessMask;
    rifeOutputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    rifeOutputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rifeOutputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rifeOutputBarrier.buffer = sourceBuffer;
    rifeOutputBarrier.offset = 0;
    rifeOutputBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStageMask,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &rifeOutputBarrier,
        0,
        nullptr
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { swapChainExtent.width, swapChainExtent.height, 1 };

    vkCmdCopyBufferToImage(
        commandBuffer,
        sourceBuffer,
        swapChainImages[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    VkImageMemoryBarrier backToPresentBarrier{};
    backToPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    backToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    backToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    backToPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToPresentBarrier.image = swapChainImages[imageIndex];
    backToPresentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    backToPresentBarrier.subresourceRange.baseMipLevel = 0;
    backToPresentBarrier.subresourceRange.levelCount = 1;
    backToPresentBarrier.subresourceRange.baseArrayLayer = 0;
    backToPresentBarrier.subresourceRange.layerCount = 1;
    backToPresentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    backToPresentBarrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &backToPresentBarrier
    );
}

void VulkanRifeRendererApp::displayRifeFrameOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (!hasRifeDisplayFrame ||
        rifePendingInterpolatedOutputIndex >= rifeOutputBuffers.size() ||
        rifeOutputBuffers.empty()) {
        return;
    }

    auto& selectedOutput = rifeOutputBuffers[rifePendingInterpolatedOutputIndex];
    if (!selectedOutput.ready ||
        selectedOutput.inUseByGraphics ||
        selectedOutput.gpuBuffer == VK_NULL_HANDLE ||
        selectedOutput.size == 0) {
        hasRifeDisplayFrame = false;
        rifePendingInterpolatedOutputIndex = UINT32_MAX;
        return;
    }

    copyRifeBufferToSwapchain(
        commandBuffer,
        imageIndex,
        selectedOutput.gpuBuffer,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    selectedOutput.ready = false;
    selectedOutput.inUseByGraphics = true;
    selectedOutput.graphicsFrameSlot = currentFrame;

    hasRifeDisplayFrame = false;
    rifePendingInterpolatedOutputIndex = UINT32_MAX;
    rifeHeldSourceDisplayIndex = UINT32_MAX;
}

void VulkanRifeRendererApp::displayRifeSourceBufferOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t sourceIndex) {
    if (sourceIndex >= offscreenFrames.size()) {
        return;
    }

    const auto& source = offscreenFrames[sourceIndex];
    if (source.image == VK_NULL_HANDLE || source.size == 0) {
        return;
    }

    copyOffscreenImageToSwapchain(commandBuffer, imageIndex, sourceIndex);
}

void VulkanRifeRendererApp::displayCapturedRifeSourceOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (rifePendingSourceDisplayIndex >= offscreenFrames.size()) {
        rifePendingSourceDisplayIndex = UINT32_MAX;
        return;
    }

    displayRifeSourceBufferOnSwapchain(commandBuffer, imageIndex, rifePendingSourceDisplayIndex);
    rifeLastPresentedSourceIndex = rifePendingSourceDisplayIndex;
    rifePendingSourceDisplayIndex = UINT32_MAX;
    rifeHeldSourceDisplayIndex = UINT32_MAX;
    rifeRenderAheadPending = false;
}

void VulkanRifeRendererApp::processCapturedFrameForSlot(uint32_t frameSlot) {
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
    if (!rifeRealtimeInterpolationEnabled || !rifeModelAttachedToRenderer) {
        return;
    }

    previousRifeGpuFrameIndex = currentRifeGpuFrameIndex;
    currentRifeGpuFrameIndex = captureSlot;
    hasRifeGpuFramePair =
        previousRifeGpuFrameIndex != UINT32_MAX &&
        previousRifeGpuFrameIndex != currentRifeGpuFrameIndex &&
        previousRifeGpuFrameIndex < offscreenFrames.size();

    ++capturedFrameCount;
#endif

}

void VulkanRifeRendererApp::createCommandBuffers() {
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

uint32_t VulkanRifeRendererApp::recordCommandBuffer(VkCommandBuffer commandBuffer,
                                                       uint32_t imageIndex,
                                                       PresentationCommandMode mode) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    if (mode == PresentationCommandMode::DisplayInterpolatedFrame) {
        displayRifeFrameOnSwapchain(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    if (mode == PresentationCommandMode::DisplayCapturedSourceFrame) {
        displayCapturedRifeSourceOnSwapchain(commandBuffer, imageIndex);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        return UINT32_MAX;
    }

    if (mode == PresentationCommandMode::DisplayHeldSourceFrame) {
        displayRifeSourceBufferOnSwapchain(commandBuffer, imageIndex, rifeHeldSourceDisplayIndex);

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

    // Frame flow for a real frame:
    //   scene render -> sampled offscreen image -> fused NCNN tensor preprocessor
    //   Real-frame presentation copies the same image directly to the swapchain.
    // The swapchain is only a presentation target and is never an inference
    // capture source. Once this submission's fence signals, the history manager
    // may pair this frame with its predecessor and start RIFE.
#if HAS_NCNN
    if (rifeRealtimeInterpolationEnabled &&
        rifeModelAttachedToRenderer &&
        rifeLastPresentedSourceIndex < offscreenFrames.size()) {
        // Render ahead without advancing presentation: N+1 is now available
        // for inference, but N stays visible until N+0.5 has been presented.
        displayRifeSourceBufferOnSwapchain(commandBuffer, imageIndex, rifeLastPresentedSourceIndex);
        rifeHeldSourceDisplayIndex = rifeLastPresentedSourceIndex;
        rifeRenderAheadPending = true;
    }
    else
#endif
    {
        displayRifeSourceBufferOnSwapchain(commandBuffer, imageIndex, offscreenSlot);
#if HAS_NCNN
        if (rifeRealtimeInterpolationEnabled && rifeModelAttachedToRenderer) {
            rifeLastPresentedSourceIndex = offscreenSlot;
        }
#endif
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    return offscreenSlot;
}

void VulkanRifeRendererApp::createSyncObjects() {
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

#if HAS_NCNN
void VulkanRifeRendererApp::waitForAsyncRifeInference() {
    if (asyncRifeInference.valid()) {
        asyncRifeInference.wait();
        AsyncRifeResult result = asyncRifeInference.get();
        if (result.outputIndex < rifeOutputBuffers.size()) {
            rifeOutputBuffers[result.outputIndex].inUseByInference = false;
            rifeOutputBuffers[result.outputIndex].ready = false;
        }
    }

    rifeInferenceInFlight = false;
    rifePendingInterpolatedOutputIndex = UINT32_MAX;
    rifePendingSourceDisplayIndex = UINT32_MAX;
    rifeHeldSourceDisplayIndex = UINT32_MAX;
    rifeLastPresentedSourceIndex = UINT32_MAX;
    rifeRenderAheadPending = false;
    asyncRifePrevFrameIndex = UINT32_MAX;
    asyncRifeCurrFrameIndex = UINT32_MAX;
    asyncRifeOutputIndex = UINT32_MAX;
}

void VulkanRifeRendererApp::pollAsyncRifeInference() {
    if (!asyncRifeInference.valid()) {
        rifeInferenceInFlight = false;
        return;
    }

    if (asyncRifeInference.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }

    const AsyncRifeResult result = asyncRifeInference.get();
    rifeInferenceInFlight = false;
    if (result.outputIndex < rifeOutputBuffers.size()) {
        rifeOutputBuffers[result.outputIndex].inUseByInference = false;
    }
    asyncRifePrevFrameIndex = UINT32_MAX;
    asyncRifeCurrFrameIndex = UINT32_MAX;
    asyncRifeOutputIndex = UINT32_MAX;

    if (result.processRet == 0) {
        if (result.outputIndex >= rifeOutputBuffers.size()) {
            std::cerr << "[RIFE] async GPU interpolation finished with invalid output slot" << std::endl;
            return;
        }

        ++rifeCompletedInferenceCount;
        const int previousDivisor = rifeInferenceScaleDivisor;
        if (result.inferenceMs > RIFE_TARGET_INFERENCE_MS && rifeInferenceScaleDivisor < RIFE_MAX_INFERENCE_SCALE_DIVISOR) {
            ++rifeInferenceScaleDivisor;
        }
        else if (result.inferenceMs < RIFE_FAST_INFERENCE_MS && rifeInferenceScaleDivisor > RIFE_MIN_INFERENCE_SCALE_DIVISOR) {
            --rifeInferenceScaleDivisor;
        }

        rifeOutputBuffers[result.outputIndex].ready = true;
        rifeOutputBuffers[result.outputIndex].sequence = nextRifeOutputSequence++;
        rifePendingInterpolatedOutputIndex = result.outputIndex;
        rifePendingSourceDisplayIndex = result.currentSourceIndex;
        hasRifeDisplayFrame = true;
        if (previousDivisor != rifeInferenceScaleDivisor || (rifeCompletedInferenceCount % 120) == 1) {
            std::cout << "[RIFE] async inference"
                      << " display=" << result.inputW << "x" << result.inputH
                      << ", inference=" << result.inferenceW << "x" << result.inferenceH
                      << ", scale_divisor=" << rifeInferenceScaleDivisor
                      << ", inference_ms=" << result.inferenceMs << std::endl;
        }
        return;
    }

    std::cerr << "[RIFE] async GPU interpolation failed"
              << " (code=" << result.processRet
              << ", inference_ms=" << result.inferenceMs << ")" << std::endl;
    // Do not leave the scheduler holding N forever if interpolation fails.
    // Advance to N+1 on the next presentation tick and resume render-ahead.
    rifePendingSourceDisplayIndex = result.currentSourceIndex;
}

bool VulkanRifeRendererApp::submitAsyncRifeInferenceIfReady() {
    if (!rifeRealtimeInterpolationEnabled ||
        !rifeModelAttachedToRenderer ||
        rifeInferenceInFlight ||
        !hasRifeGpuFramePair ||
        capturedFrameCount < 2 ||
        previousRifeGpuFrameIndex >= offscreenFrames.size() ||
        currentRifeGpuFrameIndex >= offscreenFrames.size() ||
        rifeOutputBuffers.empty() ||
        rifeDisplayBufferSize == 0) {
        return false;
    }

    uint32_t outputIndex = UINT32_MAX;
    for (uint32_t i = 0; i < rifeOutputBuffers.size(); ++i) {
        const auto& output = rifeOutputBuffers[i];
        if (output.gpuBuffer != VK_NULL_HANDLE &&
            output.size >= rifeDisplayBufferSize &&
            !output.ready &&
            !output.inUseByInference &&
            !output.inUseByGraphics) {
            outputIndex = i;
            break;
        }
    }

    if (outputIndex == UINT32_MAX) {
        return false;
    }

    const uint32_t prevIndex = previousRifeGpuFrameIndex;
    const uint32_t currIndex = currentRifeGpuFrameIndex;
    const VkImage prevImage = offscreenFrames[prevIndex].image;
    const VkImageView prevImageView = offscreenFrames[prevIndex].rifeInputImageView;
    const VkDeviceMemory prevMemory = offscreenFrames[prevIndex].imageMemory;
    const VkImage currImage = offscreenFrames[currIndex].image;
    const VkImageView currImageView = offscreenFrames[currIndex].rifeInputImageView;
    const VkDeviceMemory currMemory = offscreenFrames[currIndex].imageMemory;
    const VkFormat inputFormat =
        swapChainImageFormat == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM :
        swapChainImageFormat == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM :
        swapChainImageFormat;
    const VkBuffer outBuffer = rifeOutputBuffers[outputIndex].gpuBuffer;
    const VkDeviceMemory outMemory = rifeOutputBuffers[outputIndex].gpuMemory;
    const VkDeviceSize outSize = rifeOutputBuffers[outputIndex].size;
    const int inputW = static_cast<int>(swapChainExtent.width);
    const int inputH = static_cast<int>(swapChainExtent.height);
    const int divisor = std::clamp(
        rifeInferenceScaleDivisor,
        RIFE_MIN_INFERENCE_SCALE_DIVISOR,
        RIFE_MAX_INFERENCE_SCALE_DIVISOR
    );
    const int inferenceW = std::min(inputW, std::max(32, inputW / divisor));
    const int inferenceH = std::min(inputH, std::max(32, inputH / divisor));

    rifeInferenceInFlight = true;
    asyncRifePrevFrameIndex = prevIndex;
    asyncRifeCurrFrameIndex = currIndex;
    asyncRifeOutputIndex = outputIndex;
    rifeOutputBuffers[outputIndex].inUseByInference = true;
    rifeOutputBuffers[outputIndex].ready = false;
    rifeOutputBuffers[outputIndex].sequence = 0;
    hasRifeGpuFramePair = false;
    rifeHeldSourceDisplayIndex = prevIndex;
    rifeInferenceRequestWaitingForFramePair = false;

    asyncRifeInference = std::async(std::launch::async, [this,
                                                         prevImage,
                                                         prevImageView,
                                                         prevMemory,
                                                         currImage,
                                                         currImageView,
                                                         currMemory,
                                                         inputFormat,
                                                         outBuffer,
                                                         outMemory,
                                                         outSize,
                                                         inputW,
                                                         inputH,
                                                         inferenceW,
                                                         inferenceH,
                                                         currIndex,
                                                         outputIndex]() {
        AsyncRifeResult result{};
        result.inputW = inputW;
        result.inputH = inputH;
        result.inferenceW = inferenceW;
        result.inferenceH = inferenceH;
        result.outputIndex = outputIndex;
        result.currentSourceIndex = currIndex;

        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        const auto start = std::chrono::high_resolution_clock::now();
        result.processRet = rifeRunner.processGpuRgbaFrames(
            prevImage,
            prevImageView,
            prevMemory,
            currImage,
            currImageView,
            currMemory,
            inputFormat,
            outBuffer,
            outMemory,
            outSize,
            inputW,
            inputH,
            inferenceW,
            inferenceH,
            0.5f
        );
        result.inferenceMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
        return result;
    });

    return true;
}

#endif

void VulkanRifeRendererApp::updateUniformBuffer(uint32_t currentImage) {
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
}

void VulkanRifeRendererApp::drawFrame() {
    const auto prepareFrameSlot = [this](uint32_t frameSlot) {
        vkWaitForFences(device, 1, &inFlightFences[frameSlot], VK_TRUE, UINT64_MAX);
        for (auto& output : rifeOutputBuffers) {
            if (output.inUseByGraphics && output.graphicsFrameSlot == frameSlot) {
                output.inUseByGraphics = false;
                output.graphicsFrameSlot = UINT32_MAX;
            }
        }
        processCapturedFrameForSlot(frameSlot);
    };

    const auto submitCommandBuffer = [this](uint32_t frameSlot,
                                            uint32_t imageIndex,
                                            PresentationCommandMode mode) {
        vkResetFences(device, 1, &inFlightFences[frameSlot]);

        vkResetCommandBuffer(commandBuffers[frameSlot], 0);
        const uint32_t capturedRifeSlot = recordCommandBuffer(commandBuffers[frameSlot], imageIndex, mode);

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

        pendingCaptureSlotByFrame[frameSlot] = capturedRifeSlot;
    };

    const auto presentImage = [this](uint32_t imageIndex, VkSemaphore waitSemaphore) {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &waitSemaphore;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        return vkQueuePresentKHR(presentQueue, &presentInfo);
    };

    prepareFrameSlot(currentFrame);
#if HAS_NCNN
    pollAsyncRifeInference();
    if (rifeRealtimeInterpolationEnabled) {
        // Start RIFE work before acquiring this tick's swapchain image so compute can overlap graphics rendering.
        if (!submitAsyncRifeInferenceIfReady() &&
            !rifeInferenceInFlight &&
            !hasRifeDisplayFrame &&
            !rifeInferenceRequestWaitingForFramePair) {
            rifeInferenceRequestWaitingForFramePair = true;
        }
    }
#endif

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    PresentationCommandMode mode = PresentationCommandMode::RenderFrame;
#if HAS_NCNN
    if (rifeRealtimeInterpolationEnabled && hasRifeDisplayFrame) {
        mode = PresentationCommandMode::DisplayInterpolatedFrame;
    }
    else if (rifeRealtimeInterpolationEnabled &&
             rifePendingSourceDisplayIndex < offscreenFrames.size()) {
        mode = PresentationCommandMode::DisplayCapturedSourceFrame;
    }
    else if (rifeRealtimeInterpolationEnabled &&
             rifeInferenceInFlight &&
             rifeHeldSourceDisplayIndex < offscreenFrames.size()) {
        mode = PresentationCommandMode::DisplayHeldSourceFrame;
    }
    else if (rifeRealtimeInterpolationEnabled &&
             rifeRenderAheadPending &&
             rifeHeldSourceDisplayIndex < offscreenFrames.size()) {
        mode = PresentationCommandMode::DisplayHeldSourceFrame;
    }
#endif

    if (mode == PresentationCommandMode::RenderFrame) {
        updateUniformBuffer(currentFrame);
    }

    // One scheduler tick records exactly one frame source, submits it, then queues exactly one present.
    submitCommandBuffer(currentFrame, imageIndex, mode);

    result = presentImage(imageIndex, renderFinishedSemaphores[imageIndex]);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

int main() {
#if HAS_NCNN
    std::cout << "[NCNN] enabled" << std::endl;
#else
    std::cout << "[NCNN] disabled" << std::endl;
#endif

    VulkanRifeRendererApp app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
