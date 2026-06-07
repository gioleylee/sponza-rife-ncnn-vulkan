// Owns NCNN initialization, frame-processing resources, async inference, and cleanup.
#include "VulkanNcnnRenderer.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>

namespace {

void resetNcnnDisplayState(NcnnPresentationState& state) {
    state.hasNcnnDisplayFrame = false;
    state.nextNcnnOutputSequence = 1;
    state.ncnnPendingInterpolatedOutputIndex = UINT32_MAX;
    state.ncnnPendingSourceDisplayIndex = UINT32_MAX;
    state.ncnnHeldSourceDisplayIndex = UINT32_MAX;
    state.ncnnLastPresentedSourceIndex = UINT32_MAX;
    state.ncnnRenderAheadPending = false;
}

void resetNcnnFramePairState(NcnnPresentationState& state) {
    state.hasNcnnGpuFramePair = false;
    state.currentNcnnGpuFrameIndex = UINT32_MAX;
    state.previousNcnnGpuFrameIndex = UINT32_MAX;
}

void resetNcnnAsyncState(NcnnPresentationState& state) {
    state.ncnnInferenceInFlight = false;
    state.asyncNcnnPrevFrameIndex = UINT32_MAX;
    state.asyncNcnnCurrFrameIndex = UINT32_MAX;
    state.asyncNcnnOutputIndex = UINT32_MAX;
}

void resetNcnnAdaptiveInferenceState(NcnnPresentationState& state) {
    resetNcnnAsyncState(state);
    state.ncnnInferenceScaleDivisor = NCNN_INITIAL_INFERENCE_SCALE_DIVISOR;
    state.ncnnCompletedInferenceCount = 0;
}

void resetNcnnPendingPresentationState(NcnnPresentationState& state) {
    state.ncnnPendingInterpolatedOutputIndex = UINT32_MAX;
    state.ncnnPendingSourceDisplayIndex = UINT32_MAX;
    state.ncnnHeldSourceDisplayIndex = UINT32_MAX;
    state.ncnnLastPresentedSourceIndex = UINT32_MAX;
    state.ncnnRenderAheadPending = false;
}

}

#if HAS_NCNN
int VulkanNcnnRenderer::findNcnnDeviceIndexForRenderer() const {
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

void VulkanNcnnRenderer::applyNcnnVulkanOptions() {
    const int gpuCount = ncnn::get_gpu_count();
    const bool hasSelectedGpu =
        ncnnRendererDeviceIndex >= 0 && ncnnRendererDeviceIndex < gpuCount;

    net.opt.use_vulkan_compute = hasSelectedGpu && HAS_NCNN_WARP_VK;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_packing_layout = true;

    net.opt.use_cooperative_matrix = true;
}

void VulkanNcnnRenderer::initNcnn() {
    if (ncnnInitialized) {
        return;
    }

    ncnn::create_gpu_instance();

    const int gpuCount = ncnn::get_gpu_count();
    ncnnRendererDeviceIndex = findNcnnDeviceIndexForRenderer();
    applyNcnnVulkanOptions();
    net.opt.num_threads = 1;
    if (gpuCount > 0 && !HAS_NCNN_WARP_VK) {
        std::cout << "[NCNN] NCNN warp Vulkan shaders not found; forcing CPU inference path" << std::endl;
    }
    ncnnInitialized = true;

    std::cout << "[NCNN] initialized (gpu_count=" << gpuCount
              << ", renderer_gpu_index=" << ncnnRendererDeviceIndex
              << ", vulkan_compute=" << (net.opt.use_vulkan_compute ? "on" : "off") << ")" << std::endl;
}

void VulkanNcnnRenderer::shutdownNcnn() {
    if (!ncnnInitialized) {
        return;
    }

    net.clear();
    ncnnFrameInterpolator.reset();
    ncnn::destroy_gpu_instance();

    ncnnInitialized = false;
    ncnnRendererDeviceIndex = -1;
    ncnnModelLoaded = false;
    ncnnModelAttachedToRenderer = false;
}

bool VulkanNcnnRenderer::loadNcnnModel(const std::string& paramPath, const std::string& binPath) {
    if (!ncnnInitialized) {
        initNcnn();
    }

    // NCNN consumes these options while loading the model and creating its
    // Vulkan pipelines. Reapply them before every load.
    applyNcnnVulkanOptions();

    if (!ncnnFrameInterpolator.initialize(net, paramPath, binPath, device, ncnnRendererDeviceIndex)) {
        return false;
    }

    ncnnModelLoaded = true;

    std::cout << "[NCNN] model loaded: " << paramPath << " + " << binPath << std::endl;
    return true;
}

void VulkanNcnnRenderer::tryLoadDefaultNcnnModel() {
    const std::string optimizedParam = "assets/nn_models/rife-v4/flownet-opt.param";
    const std::string optimizedBin = "assets/nn_models/rife-v4/flownet-opt.bin";
    const std::string defaultParam = "assets/nn_models/rife-v4/flownet.param";
    const std::string defaultBin = "assets/nn_models/rife-v4/flownet.bin";

    const bool hasOptimizedModel =
        std::filesystem::exists(optimizedParam) && std::filesystem::exists(optimizedBin);
    const std::string& paramPath = hasOptimizedModel ? optimizedParam : defaultParam;
    const std::string& binPath = hasOptimizedModel ? optimizedBin : defaultBin;

    if (!std::filesystem::exists(paramPath) || !std::filesystem::exists(binPath)) {
        std::cout << "[NCNN] default interpolation model not found: "
                  << paramPath << " and " << binPath << std::endl;
        return;
    }

    std::cout << "[NCNN] using " << (hasOptimizedModel ? "output" : "original model")
              << ": " << paramPath << " + " << binPath << std::endl;

    if (loadNcnnModel(paramPath, binPath)) {
        ncnn::Extractor extractor = net.create_extractor();
        (void)extractor;
        std::cout << "[NCNN] extractor created successfully" << std::endl;
        ncnnModelAttachedToRenderer = ncnnFrameInterpolator.isReady();
        if (ncnnModelAttachedToRenderer) {
            std::cout << "[NCNN] model attached to Vulkan renderer" << std::endl;
        }
    }
    else {
        std::cout << "[NCNN] failed to load default interpolation model" << std::endl;
    }
}
#endif

#if defined(_WIN32)
bool VulkanNcnnRenderer::createExportableFrameBuffer(VkDeviceSize size,
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
#endif

void VulkanNcnnRenderer::createFrameProcessingResources() {
    cleanupFrameProcessingResources();

    const VkDeviceSize frameSize =
        static_cast<VkDeviceSize>(swapChainExtent.width) *
        static_cast<VkDeviceSize>(swapChainExtent.height) * 4;

    offscreenFrames.resize(OFFSCREEN_FRAME_HISTORY_COUNT);
    ncnnOutputBuffers.resize(NCNN_OUTPUT_BUFFER_COUNT);

    ncnnDisplayBufferSize = frameSize;
    resetNcnnDisplayState(ncnnPresentationState);

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
    const VkFormat ncnnInputFormat = toUnormFormat(swapChainImageFormat);
    const VkImageCreateFlags offscreenFlags =
        ncnnInputFormat != swapChainImageFormat ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;

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
        frame.ncnnInputImageView = createImageView(frame.image, ncnnInputFormat, 1);

        frame.size = frameSize;
    }

    for (auto& output : ncnnOutputBuffers) {
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
    resetNcnnFramePairState(ncnnPresentationState);
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
}

void VulkanNcnnRenderer::cleanupFrameProcessingResources() {
#if HAS_NCNN
    waitForAsyncNcnnInference();
#endif

    for (auto& output : ncnnOutputBuffers) {
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

    ncnnOutputBuffers.clear();
    ncnnDisplayBufferSize = 0;
    resetNcnnDisplayState(ncnnPresentationState);
#if HAS_NCNN
    resetNcnnAdaptiveInferenceState(ncnnPresentationState);
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

        if (frame.ncnnInputImageView) {
            vkDestroyImageView(device, frame.ncnnInputImageView, nullptr);
            frame.ncnnInputImageView = VK_NULL_HANDLE;
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
    resetNcnnFramePairState(ncnnPresentationState);
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
}

uint32_t VulkanNcnnRenderer::findAvailableOffscreenFrameSlot() const {
    for (uint32_t slot = 0; slot < offscreenFrames.size(); ++slot) {
#if HAS_NCNN
        if (ncnnPresentationState.ncnnInferenceInFlight &&
            (slot == ncnnPresentationState.asyncNcnnPrevFrameIndex || slot == ncnnPresentationState.asyncNcnnCurrFrameIndex)) {
            continue;
        }
#endif
        if (slot == ncnnPresentationState.currentNcnnGpuFrameIndex) {
            continue;
        }
        if (slot == ncnnPresentationState.ncnnPendingSourceDisplayIndex) {
            continue;
        }
        if (slot == ncnnPresentationState.ncnnHeldSourceDisplayIndex) {
            continue;
        }
        if (slot == ncnnPresentationState.ncnnLastPresentedSourceIndex) {
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

void VulkanNcnnRenderer::copyNcnnBufferToSwapchain(VkCommandBuffer commandBuffer,
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

    VkBufferMemoryBarrier ncnnOutputBarrier{};
    ncnnOutputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    ncnnOutputBarrier.srcAccessMask = sourceAccessMask;
    ncnnOutputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ncnnOutputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ncnnOutputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ncnnOutputBarrier.buffer = sourceBuffer;
    ncnnOutputBarrier.offset = 0;
    ncnnOutputBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStageMask,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &ncnnOutputBarrier,
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

void VulkanNcnnRenderer::displayNcnnFrameOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (!ncnnPresentationState.hasNcnnDisplayFrame ||
        ncnnPresentationState.ncnnPendingInterpolatedOutputIndex >= ncnnOutputBuffers.size() ||
        ncnnOutputBuffers.empty()) {
        return;
    }

    auto& selectedOutput = ncnnOutputBuffers[ncnnPresentationState.ncnnPendingInterpolatedOutputIndex];
    if (!selectedOutput.ready ||
        selectedOutput.inUseByGraphics ||
        selectedOutput.gpuBuffer == VK_NULL_HANDLE ||
        selectedOutput.size == 0) {
        ncnnPresentationState.hasNcnnDisplayFrame = false;
        ncnnPresentationState.ncnnPendingInterpolatedOutputIndex = UINT32_MAX;
        return;
    }

    copyNcnnBufferToSwapchain(
        commandBuffer,
        imageIndex,
        selectedOutput.gpuBuffer,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    selectedOutput.ready = false;
    selectedOutput.inUseByGraphics = true;
    selectedOutput.graphicsFrameSlot = currentFrame;

    ncnnPresentationState.hasNcnnDisplayFrame = false;
    ncnnPresentationState.ncnnPendingInterpolatedOutputIndex = UINT32_MAX;
    ncnnPresentationState.ncnnHeldSourceDisplayIndex = UINT32_MAX;
}

void VulkanNcnnRenderer::displayNcnnSourceBufferOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t sourceIndex) {
    if (sourceIndex >= offscreenFrames.size()) {
        return;
    }

    const auto& source = offscreenFrames[sourceIndex];
    if (source.image == VK_NULL_HANDLE || source.size == 0) {
        return;
    }

    copyOffscreenImageToSwapchain(commandBuffer, imageIndex, sourceIndex);
}

void VulkanNcnnRenderer::displayCapturedNcnnSourceOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (ncnnPresentationState.ncnnPendingSourceDisplayIndex >= offscreenFrames.size()) {
        ncnnPresentationState.ncnnPendingSourceDisplayIndex = UINT32_MAX;
        return;
    }

    displayNcnnSourceBufferOnSwapchain(commandBuffer, imageIndex, ncnnPresentationState.ncnnPendingSourceDisplayIndex);
    ncnnPresentationState.ncnnLastPresentedSourceIndex = ncnnPresentationState.ncnnPendingSourceDisplayIndex;
    ncnnPresentationState.ncnnPendingSourceDisplayIndex = UINT32_MAX;
    ncnnPresentationState.ncnnHeldSourceDisplayIndex = UINT32_MAX;
    ncnnPresentationState.ncnnRenderAheadPending = false;
}

#if HAS_NCNN
void VulkanNcnnRenderer::waitForAsyncNcnnInference() {
    if (asyncNcnnInference.valid()) {
        asyncNcnnInference.wait();
        AsyncNcnnResult result = asyncNcnnInference.get();
        if (result.outputIndex < ncnnOutputBuffers.size()) {
            ncnnOutputBuffers[result.outputIndex].inUseByInference = false;
            ncnnOutputBuffers[result.outputIndex].ready = false;
        }
    }

    resetNcnnAsyncState(ncnnPresentationState);
    resetNcnnPendingPresentationState(ncnnPresentationState);
}

void VulkanNcnnRenderer::pollAsyncNcnnInference() {
    if (!asyncNcnnInference.valid()) {
        ncnnPresentationState.ncnnInferenceInFlight = false;
        return;
    }

    if (asyncNcnnInference.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }

    const AsyncNcnnResult result = asyncNcnnInference.get();
    if (result.outputIndex < ncnnOutputBuffers.size()) {
        ncnnOutputBuffers[result.outputIndex].inUseByInference = false;
    }
    resetNcnnAsyncState(ncnnPresentationState);

    if (result.processRet == 0) {
        if (result.outputIndex >= ncnnOutputBuffers.size()) {
            std::cerr << "[NCNN] async GPU interpolation finished with invalid output slot" << std::endl;
            return;
        }

        ++ncnnPresentationState.ncnnCompletedInferenceCount;
        const int previousDivisor = ncnnPresentationState.ncnnInferenceScaleDivisor;
        if (result.inferenceMs > NCNN_TARGET_INFERENCE_MS && ncnnPresentationState.ncnnInferenceScaleDivisor < NCNN_MAX_INFERENCE_SCALE_DIVISOR) {
            ++ncnnPresentationState.ncnnInferenceScaleDivisor;
        }
        else if (result.inferenceMs < NCNN_FAST_INFERENCE_MS && ncnnPresentationState.ncnnInferenceScaleDivisor > NCNN_MIN_INFERENCE_SCALE_DIVISOR) {
            --ncnnPresentationState.ncnnInferenceScaleDivisor;
        }

        ncnnOutputBuffers[result.outputIndex].ready = true;
        ncnnOutputBuffers[result.outputIndex].sequence = ncnnPresentationState.nextNcnnOutputSequence++;
        ncnnPresentationState.ncnnPendingInterpolatedOutputIndex = result.outputIndex;
        ncnnPresentationState.ncnnPendingSourceDisplayIndex = result.currentSourceIndex;
        ncnnPresentationState.hasNcnnDisplayFrame = true;
        if (previousDivisor != ncnnPresentationState.ncnnInferenceScaleDivisor || (ncnnPresentationState.ncnnCompletedInferenceCount % 120) == 1) {
            std::cout << "[NCNN] async inference"
                      << " display=" << result.inputW << "x" << result.inputH
                      << ", inference=" << result.inferenceW << "x" << result.inferenceH
                      << ", scale_divisor=" << ncnnPresentationState.ncnnInferenceScaleDivisor
                      << ", inference_ms=" << result.inferenceMs << std::endl;
        }
        return;
    }

    std::cerr << "[NCNN] async GPU interpolation failed"
              << " (code=" << result.processRet
              << ", inference_ms=" << result.inferenceMs << ")" << std::endl;
    // Do not leave the scheduler holding N forever if interpolation fails.
    // Advance to N+1 on the next presentation tick and resume render-ahead.
    ncnnPresentationState.ncnnPendingSourceDisplayIndex = result.currentSourceIndex;
}

bool VulkanNcnnRenderer::submitAsyncNcnnInferenceIfReady() {
    if (!ncnnPresentationState.ncnnRealtimeInterpolationEnabled ||
        !ncnnModelAttachedToRenderer ||
        ncnnPresentationState.ncnnInferenceInFlight ||
        !ncnnPresentationState.hasNcnnGpuFramePair ||
        capturedFrameCount < 2 ||
        ncnnPresentationState.previousNcnnGpuFrameIndex >= offscreenFrames.size() ||
        ncnnPresentationState.currentNcnnGpuFrameIndex >= offscreenFrames.size() ||
        ncnnOutputBuffers.empty() ||
        ncnnDisplayBufferSize == 0) {
        return false;
    }

    uint32_t outputIndex = UINT32_MAX;
    for (uint32_t i = 0; i < ncnnOutputBuffers.size(); ++i) {
        const auto& output = ncnnOutputBuffers[i];
        if (output.gpuBuffer != VK_NULL_HANDLE &&
            output.size >= ncnnDisplayBufferSize &&
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

    const uint32_t prevIndex = ncnnPresentationState.previousNcnnGpuFrameIndex;
    const uint32_t currIndex = ncnnPresentationState.currentNcnnGpuFrameIndex;
    const VkImage prevImage = offscreenFrames[prevIndex].image;
    const VkImageView prevImageView = offscreenFrames[prevIndex].ncnnInputImageView;
    const VkDeviceMemory prevMemory = offscreenFrames[prevIndex].imageMemory;
    const VkImage currImage = offscreenFrames[currIndex].image;
    const VkImageView currImageView = offscreenFrames[currIndex].ncnnInputImageView;
    const VkDeviceMemory currMemory = offscreenFrames[currIndex].imageMemory;
    const VkFormat inputFormat =
        swapChainImageFormat == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM :
        swapChainImageFormat == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM :
        swapChainImageFormat;
    const VkBuffer outBuffer = ncnnOutputBuffers[outputIndex].gpuBuffer;
    const VkDeviceMemory outMemory = ncnnOutputBuffers[outputIndex].gpuMemory;
    const VkDeviceSize outSize = ncnnOutputBuffers[outputIndex].size;
    const int inputW = static_cast<int>(swapChainExtent.width);
    const int inputH = static_cast<int>(swapChainExtent.height);
    const int divisor = std::clamp(
        ncnnPresentationState.ncnnInferenceScaleDivisor,
        NCNN_MIN_INFERENCE_SCALE_DIVISOR,
        NCNN_MAX_INFERENCE_SCALE_DIVISOR
    );
    const int inferenceW = std::min(inputW, std::max(32, inputW / divisor));
    const int inferenceH = std::min(inputH, std::max(32, inputH / divisor));

    ncnnPresentationState.ncnnInferenceInFlight = true;
    ncnnPresentationState.asyncNcnnPrevFrameIndex = prevIndex;
    ncnnPresentationState.asyncNcnnCurrFrameIndex = currIndex;
    ncnnPresentationState.asyncNcnnOutputIndex = outputIndex;
    ncnnOutputBuffers[outputIndex].inUseByInference = true;
    ncnnOutputBuffers[outputIndex].ready = false;
    ncnnOutputBuffers[outputIndex].sequence = 0;
    ncnnPresentationState.hasNcnnGpuFramePair = false;
    ncnnPresentationState.ncnnHeldSourceDisplayIndex = prevIndex;
    ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair = false;

    asyncNcnnInference = std::async(std::launch::async, [this,
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
        AsyncNcnnResult result{};
        result.inputW = inputW;
        result.inputH = inputH;
        result.inferenceW = inferenceW;
        result.inferenceH = inferenceH;
        result.outputIndex = outputIndex;
        result.currentSourceIndex = currIndex;

        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        const auto start = std::chrono::high_resolution_clock::now();
        result.processRet = ncnnFrameInterpolator.processGpuRgbaFrames(
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
