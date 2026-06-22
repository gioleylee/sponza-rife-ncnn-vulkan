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

constexpr uint32_t INTERPOLATED_FRAME_MARKER_MIN_WIDTH = 24;
constexpr uint32_t INTERPOLATED_FRAME_MARKER_WIDTH_DIVISOR = 40;

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
    state.ncnnRunningJobCount = 0;
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

    uint32_t ncnnComputeQueueFamilyIndex = UINT32_MAX;
    uint32_t ncnnTransferQueueFamilyIndex = UINT32_MAX;
    if (ncnnRendererDeviceIndex >= 0 && ncnnRendererDeviceIndex < gpuCount) {
        const ncnn::GpuInfo& gpuInfo = ncnn::get_gpu_info(ncnnRendererDeviceIndex);
        ncnnComputeQueueFamilyIndex = gpuInfo.compute_queue_family_index();
        ncnnTransferQueueFamilyIndex = gpuInfo.transfer_queue_family_index();
    }

    const auto rendererQueueUsesNcnnQueueZero = [&](uint32_t familyIndex, uint32_t queueIndex) {
        if (queueIndex != 0) {
            return false;
        }

        return familyIndex == ncnnComputeQueueFamilyIndex ||
               familyIndex == ncnnTransferQueueFamilyIndex;
    };

    const bool ncnnUsesGraphicsQueueFamily =
        ncnnComputeQueueFamilyIndex == graphicsQueueFamilyIndex &&
        ncnnTransferQueueFamilyIndex == graphicsQueueFamilyIndex;

    ncnnCanRunWithoutQueueMutex =
        ncnnRendererDeviceIndex >= 0 &&
        ncnnUsesGraphicsQueueFamily &&
        !rendererQueueUsesNcnnQueueZero(graphicsQueueFamilyIndex, graphicsQueueIndex) &&
        !rendererQueueUsesNcnnQueueZero(presentQueueFamilyIndex, presentQueueIndex);

    if (gpuCount > 0 && !HAS_NCNN_WARP_VK) {
        std::cout << "[NCNN] NCNN warp Vulkan shaders not found; forcing CPU inference path" << std::endl;
    }
    ncnnInitialized = true;

    std::cout << "[NCNN] initialized (vulkan_compute=" << (net.opt.use_vulkan_compute ? "on" : "off") << ")" << std::endl;

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
    ncnnCanRunWithoutQueueMutex = false;
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

    for (uint32_t i = 0; i < offscreenFrames.size(); ++i) {
        auto& frame = offscreenFrames[i];
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
        setDebugObjectName(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(frame.image),
            "Offscreen Source Image Slot " + std::to_string(i));
        setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<uint64_t>(frame.imageView),
            "Offscreen Source SRGB View Slot " + std::to_string(i));
        setDebugObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, reinterpret_cast<uint64_t>(frame.ncnnInputImageView),
            "Offscreen Source NCNN Input View Slot " + std::to_string(i));

        frame.size = frameSize;
    }

    for (uint32_t i = 0; i < ncnnOutputBuffers.size(); ++i) {
        auto& output = ncnnOutputBuffers[i];
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
        output.debugPreviousFrameId = UINT64_MAX;
        output.debugCurrentFrameId = UINT64_MAX;
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(output.gpuBuffer),
            "NCNN Interpolated Output Buffer Slot " + std::to_string(i));
    }

    pendingCaptureSlotByFrame.fill(UINT32_MAX);
    resetNcnnFramePairState(ncnnPresentationState);
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
    resetFrameInterpolationDebugState();
}

void VulkanNcnnRenderer::initializeFrameProcessingImageLayouts() {
    for (auto& frame : offscreenFrames) {
        if (frame.image == VK_NULL_HANDLE) {
            continue;
        }

        transitionImageLayout(frame.image, swapChainImageFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    }
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
        output.debugPreviousFrameId = UINT64_MAX;
        output.debugCurrentFrameId = UINT64_MAX;
    }

    ncnnOutputBuffers.clear();
    ncnnInterpolationTargets.clear();
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
        frame.debugFrameId = UINT64_MAX;
        frame.debugPresented = false;
    }

    offscreenFrames.clear();
    pendingCaptureSlotByFrame.fill(UINT32_MAX);
    resetNcnnFramePairState(ncnnPresentationState);
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
    resetFrameInterpolationDebugState();
}

uint32_t VulkanNcnnRenderer::findAvailableOffscreenFrameSlot() const {
    for (uint32_t slot = 0; slot < offscreenFrames.size(); ++slot) {
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
        if (offscreenFrames[slot].debugFrameId != UINT64_MAX && !offscreenFrames[slot].debugPresented) {
            continue;
        }

        bool usedByPendingInterpolation = false;
        for (const auto& target : ncnnInterpolationTargets) {
            if ((target.state == InterpolationTargetState::Pending ||
                 target.state == InterpolationTargetState::Running ||
                 target.state == InterpolationTargetState::Ready ||
                 target.state == InterpolationTargetState::Presenting) &&
                !target.waitingForFutureSource &&
                (slot == target.previousSourceIndex || slot == target.currentSourceIndex)) {
                usedByPendingInterpolation = true;
                break;
            }
        }
        if (usedByPendingInterpolation) {
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
                                                   VkPipelineStageFlags sourceStageMask,
                                                   const std::string& logicalFrameLabel) {
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

    beginDebugLabel(commandBuffer, "Output Conversion " + logicalFrameLabel, glm::vec4(0.9f, 0.4f, 1.0f, 1.0f));
    uint32_t markerWidth = 0;
    if (markInterpolatedFrames) {
        markerWidth = std::min(
            std::max(
                INTERPOLATED_FRAME_MARKER_MIN_WIDTH,
                swapChainExtent.width / INTERPOLATED_FRAME_MARKER_WIDTH_DIVISOR),
            swapChainExtent.width);

        VkClearColorValue markerColor{};
        markerColor.float32[0] = 0.0f;
        markerColor.float32[1] = 1.0f;
        markerColor.float32[2] = 0.0f;
        markerColor.float32[3] = 1.0f;

        VkImageSubresourceRange markerSubresource{};
        markerSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        markerSubresource.baseMipLevel = 0;
        markerSubresource.levelCount = 1;
        markerSubresource.baseArrayLayer = 0;
        markerSubresource.layerCount = 1;

        // Clear first, then copy the interpolated image over everything except
        // the left marker strip. The marker never feeds back into frame history.
        vkCmdClearColorImage(
            commandBuffer,
            swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &markerColor,
            1,
            &markerSubresource
        );
    }

    const uint32_t copyWidth = swapChainExtent.width - markerWidth;
    if (copyWidth > 0) {
        VkBufferImageCopy region{};
        region.bufferOffset = static_cast<VkDeviceSize>(markerWidth) * 4;
        region.bufferRowLength = swapChainExtent.width;
        region.bufferImageHeight = swapChainExtent.height;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { static_cast<int32_t>(markerWidth), 0, 0 };
        region.imageExtent = { copyWidth, swapChainExtent.height, 1 };

        vkCmdCopyBufferToImage(
            commandBuffer,
            sourceBuffer,
            swapChainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );
    }
    endDebugLabel(commandBuffer);

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

    const uint64_t presentedPreviousFrameId = selectedOutput.debugPreviousFrameId;
    const uint32_t targetIndex = findInterpolationTargetIndex(presentedPreviousFrameId);
    pushNvtxRange(
        "CPU Copy Record: Logical Frame " + debugInterpolatedFrameLabel(presentedPreviousFrameId) +
        " interpolated [outputSlot=" +
        std::to_string(ncnnPresentationState.ncnnPendingInterpolatedOutputIndex) +
        ", swapchain=" + std::to_string(imageIndex) + "]");
    const std::string copyLabel =
        "Copy Logical Frame " + debugInterpolatedFrameLabel(presentedPreviousFrameId) +
        " to Swapchain [Output Slot " + std::to_string(ncnnPresentationState.ncnnPendingInterpolatedOutputIndex) + "]";
    beginDebugLabel(commandBuffer, copyLabel, glm::vec4(0.9f, 0.4f, 1.0f, 1.0f));
    copyNcnnBufferToSwapchain(
        commandBuffer,
        imageIndex,
        selectedOutput.gpuBuffer,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        debugInterpolatedFrameLabel(presentedPreviousFrameId)
    );
    endDebugLabel(commandBuffer);
    popNvtxRange();
    pendingPresentedFrameKind = PresentedFrameKind::Interpolated;

    selectedOutput.ready = false;
    selectedOutput.inUseByGraphics = true;
    selectedOutput.graphicsFrameSlot = currentFrame;
    markNvtxInstant(
        std::string("Slot Transition: output ") +
        std::to_string(ncnnPresentationState.ncnnPendingInterpolatedOutputIndex) +
        " ready -> presenting [interp=" + debugInterpolatedFrameLabel(presentedPreviousFrameId) +
        ", src=" + debugRealFrameLabel(selectedOutput.debugPreviousFrameId) +
        "+" + debugRealFrameLabel(selectedOutput.debugCurrentFrameId) + "]");
    markDebugInterpolatedFramePresented(presentedPreviousFrameId);
    if (targetIndex < ncnnInterpolationTargets.size()) {
        auto& target = ncnnInterpolationTargets[targetIndex];
        target.state = InterpolationTargetState::Presenting;
    }

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

    markDebugRealFramePresented(sourceIndex);
    pendingPresentedFrameKind = PresentedFrameKind::Real;
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
void VulkanNcnnRenderer::setNcnnRealtimeInterpolationEnabled(bool enabled) {
    if (ncnnPresentationState.ncnnRealtimeInterpolationEnabled == enabled) {
        return;
    }

    waitForAsyncNcnnInference();

    if (enabled) {
        ncnnPresentationState = NcnnPresentationState{};
        ncnnPresentationState.ncnnRealtimeInterpolationEnabled = true;
    }
    else {
        ncnnPresentationState.ncnnRealtimeInterpolationEnabled = false;
        ncnnPresentationState.hasNcnnGpuFramePair = false;
        ncnnPresentationState.currentNcnnGpuFrameIndex = UINT32_MAX;
        ncnnPresentationState.previousNcnnGpuFrameIndex = UINT32_MAX;
        ncnnPresentationState.hasNcnnDisplayFrame = false;
        ncnnPresentationState.ncnnPendingInterpolatedOutputIndex = UINT32_MAX;
        ncnnPresentationState.ncnnPendingSourceDisplayIndex = UINT32_MAX;
        ncnnPresentationState.ncnnHeldSourceDisplayIndex = UINT32_MAX;
        ncnnPresentationState.ncnnLastPresentedSourceIndex = UINT32_MAX;
        ncnnPresentationState.ncnnRenderAheadPending = false;
        ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair = false;
    }

    for (auto& output : ncnnOutputBuffers) {
        if (!output.inUseByInference) {
            output.ready = false;
            output.sequence = 0;
        }
    }

    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
    resetFrameInterpolationDebugState();

    std::cout << "[NCNN] realtime interpolation "
              << (ncnnPresentationState.ncnnRealtimeInterpolationEnabled ? "enabled" : "disabled")
              << std::endl;
}

void VulkanNcnnRenderer::waitForAsyncNcnnInference() {
    for (uint32_t targetIndex = 0; targetIndex < ncnnInterpolationTargets.size(); ++targetIndex) {
        auto& target = ncnnInterpolationTargets[targetIndex];
        if (target.state != InterpolationTargetState::Running || !target.future.valid()) {
            continue;
        }

        target.future.wait();
        AsyncNcnnResult result = target.future.get();
        if (result.outputIndex < ncnnOutputBuffers.size()) {
            ncnnOutputBuffers[result.outputIndex].inUseByInference = false;
            ncnnOutputBuffers[result.outputIndex].ready = false;
            ncnnOutputBuffers[result.outputIndex].debugPreviousFrameId = UINT64_MAX;
            ncnnOutputBuffers[result.outputIndex].debugCurrentFrameId = UINT64_MAX;
        }
        dropInterpolationTarget(targetIndex, "async wait abandoned result");
    }

    resetNcnnAsyncState(ncnnPresentationState);
    resetNcnnPendingPresentationState(ncnnPresentationState);
}

void VulkanNcnnRenderer::pollAsyncNcnnInference() {
    uint32_t runningCount = 0;
    uint32_t readyCount = 0;
    uint32_t pendingCount = 0;

    for (uint32_t targetIndex = 0; targetIndex < ncnnInterpolationTargets.size(); ++targetIndex) {
        auto& target = ncnnInterpolationTargets[targetIndex];
        if (target.state == InterpolationTargetState::Pending) {
            ++pendingCount;
            continue;
        }
        if (target.state == InterpolationTargetState::Ready) {
            ++readyCount;
            continue;
        }
        if (target.state != InterpolationTargetState::Running) {
            continue;
        }

        if (!target.future.valid()) {
            dropInterpolationTarget(targetIndex, "running job has no future");
            continue;
        }
        if (target.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++runningCount;
            continue;
        }

        pushNvtxRange(
            "CPU Interpolation Job " + debugInterpolatedFrameLabel(target.previousFrameId) +
            ": wait for completion result [outputSlot=" +
            std::to_string(target.outputIndex) + "]");
        const AsyncNcnnResult result = target.future.get();
        popNvtxRange();
        if (result.outputIndex < ncnnOutputBuffers.size()) {
            ncnnOutputBuffers[result.outputIndex].inUseByInference = false;
        }

        if (!ncnnPresentationState.ncnnRealtimeInterpolationEnabled) {
            if (result.outputIndex < ncnnOutputBuffers.size()) {
                ncnnOutputBuffers[result.outputIndex].ready = false;
                ncnnOutputBuffers[result.outputIndex].sequence = 0;
                ncnnOutputBuffers[result.outputIndex].debugPreviousFrameId = UINT64_MAX;
                ncnnOutputBuffers[result.outputIndex].debugCurrentFrameId = UINT64_MAX;
            }
            dropInterpolationTarget(targetIndex, "realtime interpolation disabled");
            resetNcnnFramePairState(ncnnPresentationState);
            resetNcnnPendingPresentationState(ncnnPresentationState);
            continue;
        }

        if (result.processRet != 0) {
            std::cerr << "[NCNN] async GPU interpolation failed"
                      << " (code=" << result.processRet
                      << ", process_ms=" << result.rifeProcessMs << ")" << std::endl;
            ncnnPresentationState.ncnnPendingSourceDisplayIndex = result.currentSourceIndex;
            dropInterpolationTarget(targetIndex, "NCNN inference failed");
            continue;
        }

        if (result.outputIndex >= ncnnOutputBuffers.size()) {
            std::cerr << "[NCNN] async GPU interpolation finished with invalid output slot" << std::endl;
            dropInterpolationTarget(targetIndex, "invalid NCNN output slot");
            continue;
        }

        ++ncnnPresentationState.ncnnCompletedInferenceCount;
        const int previousDivisor = ncnnPresentationState.ncnnInferenceScaleDivisor;
        if (result.inferenceMs > NCNN_TARGET_INFERENCE_MS && ncnnPresentationState.ncnnInferenceScaleDivisor < NCNN_MAX_INFERENCE_SCALE_DIVISOR) {
            ++ncnnPresentationState.ncnnInferenceScaleDivisor;
        }
        else if (result.inferenceMs < NCNN_FAST_INFERENCE_MS && ncnnPresentationState.ncnnInferenceScaleDivisor > NCNN_MIN_INFERENCE_SCALE_DIVISOR) {
            --ncnnPresentationState.ncnnInferenceScaleDivisor;
        }

        pushNvtxRange(
            "CPU Interpolation Job " + debugInterpolatedFrameLabel(result.previousFrameId) +
            ": mark output ready [outputSlot=" + std::to_string(result.outputIndex) +
            ", src=" + debugRealFrameLabel(result.previousFrameId) +
            "+" + debugRealFrameLabel(result.currentFrameId) + "]");
        ncnnOutputBuffers[result.outputIndex].ready = true;
        ncnnOutputBuffers[result.outputIndex].sequence = ncnnPresentationState.nextNcnnOutputSequence++;
        ncnnOutputBuffers[result.outputIndex].debugPreviousFrameId = result.previousFrameId;
        ncnnOutputBuffers[result.outputIndex].debugCurrentFrameId = result.currentFrameId;
        target.state = InterpolationTargetState::Ready;
        target.outputIndex = result.outputIndex;
        target.previousSourceIndex = result.previousSourceIndex;
        target.currentSourceIndex = result.currentSourceIndex;
        ncnnPresentationState.ncnnPendingInterpolatedOutputIndex = result.outputIndex;
        ncnnPresentationState.ncnnPendingSourceDisplayIndex = result.currentSourceIndex;
        ncnnPresentationState.hasNcnnDisplayFrame = true;
        markNvtxInstant(
            std::string("Slot Transition: output ") + std::to_string(result.outputIndex) +
            " inference -> ready [interp=" + debugInterpolatedFrameLabel(result.previousFrameId) +
            ", src=" + debugRealFrameLabel(result.previousFrameId) +
            "+" + debugRealFrameLabel(result.currentFrameId) + "]");
        popNvtxRange();
        ++readyCount;

        if (previousDivisor != ncnnPresentationState.ncnnInferenceScaleDivisor || (ncnnPresentationState.ncnnCompletedInferenceCount % 120) == 1) {
            std::cout << "[NCNN] display=" << result.inputW << "x" << result.inputH
                      << ", input=" << result.inferenceW << "x" << result.inferenceH
                      << ", scale_divisor=" << ncnnPresentationState.ncnnInferenceScaleDivisor
                      << ", process_ms=" << result.rifeProcessMs << std::endl;
        }
    }

    ncnnPresentationState.ncnnRunningJobCount = runningCount;
}

bool VulkanNcnnRenderer::submitAsyncNcnnInferenceIfReady() {
    releaseObsoleteNcnnOutputBuffers();

    if (!ncnnPresentationState.ncnnRealtimeInterpolationEnabled) {
        return false;
    }
    if (!ncnnModelAttachedToRenderer) {
        return false;
    }
    if (ncnnOutputBuffers.empty() || ncnnDisplayBufferSize == 0) {
        return false;
    }

    uint32_t runningCount = 0;
    uint32_t pendingCount = 0;
    uint32_t readyCount = 0;
    for (const auto& target : ncnnInterpolationTargets) {
        if (target.state == InterpolationTargetState::Running) {
            ++runningCount;
        }
        else if (target.state == InterpolationTargetState::Pending) {
            ++pendingCount;
        }
        else if (target.state == InterpolationTargetState::Ready) {
            ++readyCount;
        }
    }
    ncnnPresentationState.ncnnRunningJobCount = runningCount;

    if (runningCount >= MAX_NCNN_IN_FLIGHT) {
        return false;
    }

    bool startedAny = false;
    while (runningCount < MAX_NCNN_IN_FLIGHT) {
        uint32_t targetIndex = UINT32_MAX;
        for (uint32_t i = 0; i < ncnnInterpolationTargets.size(); ++i) {
            const auto& target = ncnnInterpolationTargets[i];
            if (target.state == InterpolationTargetState::Pending &&
                !target.waitingForFutureSource &&
                target.previousSourceIndex < offscreenFrames.size() &&
                target.currentSourceIndex < offscreenFrames.size()) {
                targetIndex = i;
                break;
            }
        }

        if (targetIndex == UINT32_MAX) {
            break;
        }

        const uint64_t pendingPrevFrameId = ncnnInterpolationTargets[targetIndex].previousFrameId;
        const uint64_t pendingCurrFrameId = ncnnInterpolationTargets[targetIndex].currentFrameId;
        pushNvtxRange(
            "CPU Interpolation Job " + debugInterpolatedFrameLabel(pendingPrevFrameId) +
            ": wait for source frames [src=" + debugRealFrameLabel(pendingPrevFrameId) +
            "+" + debugRealFrameLabel(pendingCurrFrameId) + "]");
        popNvtxRange();

        pushNvtxRange(
            "CPU Interpolation Job " + debugInterpolatedFrameLabel(pendingPrevFrameId) +
            ": acquire output slot [src=" + debugRealFrameLabel(pendingPrevFrameId) +
            "+" + debugRealFrameLabel(pendingCurrFrameId) + "]");
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
            uint32_t reusableFutureTargetIndex = UINT32_MAX;
            uint64_t reusableFutureFrameId = 0;
            for (uint32_t i = 0; i < ncnnInterpolationTargets.size(); ++i) {
                const auto& candidate = ncnnInterpolationTargets[i];
                if (candidate.state != InterpolationTargetState::Ready ||
                    candidate.outputIndex >= ncnnOutputBuffers.size() ||
                    candidate.previousFrameId <= ncnnInterpolationTargets[targetIndex].previousFrameId) {
                    continue;
                }

                const auto& output = ncnnOutputBuffers[candidate.outputIndex];
                if (output.inUseByGraphics || output.inUseByInference) {
                    continue;
                }

                if (candidate.previousFrameId >= reusableFutureFrameId) {
                    reusableFutureFrameId = candidate.previousFrameId;
                    reusableFutureTargetIndex = i;
                }
            }

            if (reusableFutureTargetIndex < ncnnInterpolationTargets.size()) {
                outputIndex = ncnnInterpolationTargets[reusableFutureTargetIndex].outputIndex;
                dropInterpolationTarget(reusableFutureTargetIndex, "dropped to free NCNN output buffer for earlier target");
            }
        }

        if (outputIndex == UINT32_MAX) {
            popNvtxRange();
            break;
        }
        popNvtxRange();

        auto& target = ncnnInterpolationTargets[targetIndex];
        const uint32_t prevIndex = target.previousSourceIndex;
        const uint32_t currIndex = target.currentSourceIndex;
        const uint64_t prevFrameId = target.previousFrameId;
        const uint64_t currFrameId = target.currentFrameId;
        if (prevIndex >= offscreenFrames.size() ||
            currIndex >= offscreenFrames.size() ||
            offscreenFrames[prevIndex].debugFrameId != prevFrameId ||
            offscreenFrames[currIndex].debugFrameId != currFrameId) {
            dropInterpolationTarget(targetIndex, "source offscreen slot was overwritten before inference");
            continue;
        }

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

        ncnnOutputBuffers[outputIndex].inUseByInference = true;
        ncnnOutputBuffers[outputIndex].ready = false;
        ncnnOutputBuffers[outputIndex].sequence = 0;
        ncnnOutputBuffers[outputIndex].debugPreviousFrameId = prevFrameId;
        ncnnOutputBuffers[outputIndex].debugCurrentFrameId = currFrameId;
        markNvtxInstant(
            std::string("Slot Transition: output ") + std::to_string(outputIndex) +
            " free -> inference [interp=" + debugInterpolatedFrameLabel(prevFrameId) +
            ", srcSlots=" + std::to_string(prevIndex) +
            "+" + std::to_string(currIndex) + "]");
        ncnnPresentationState.hasNcnnGpuFramePair = false;
        ncnnPresentationState.ncnnHeldSourceDisplayIndex = prevIndex;
        ncnnPresentationState.ncnnInferenceRequestWaitingForFramePair = false;
        target.state = InterpolationTargetState::Running;
        target.outputIndex = outputIndex;
        ++runningCount;
        --pendingCount;
        startedAny = true;
        ncnnPresentationState.ncnnRunningJobCount = runningCount;

        markNvtxInstant(
            std::string("CPU Interpolation Job ") + debugInterpolatedFrameLabel(prevFrameId) +
            ": async job queued [outputSlot=" + std::to_string(outputIndex) + "]");

        target.future = std::async(std::launch::async, [this,
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
                                                        outputIndex,
                                                        prevIndex,
                                                        prevFrameId,
                                                        currFrameId,
                                                        targetIndex]() {
            AsyncNcnnResult result{};
            result.inputW = inputW;
            result.inputH = inputH;
            result.inferenceW = inferenceW;
            result.inferenceH = inferenceH;
            result.outputIndex = outputIndex;
            result.currentSourceIndex = currIndex;
            result.previousSourceIndex = prevIndex;
            result.previousFrameId = prevFrameId;
            result.currentFrameId = currFrameId;
            result.interpolationTargetIndex = targetIndex;

            const auto start = std::chrono::high_resolution_clock::now();
            pushNvtxRange(
                "CPU Interpolation Job " + debugInterpolatedFrameLabel(prevFrameId) +
                " full lifetime [src=" + debugRealFrameLabel(prevFrameId) +
                "+" + debugRealFrameLabel(currFrameId) +
                ", srcSlots=" + std::to_string(prevIndex) +
                "+" + std::to_string(currIndex) +
                ", outputSlot=" + std::to_string(outputIndex) + "]");

            const auto processFrames = [&]() {
                pushNvtxRange(
                    "CPU Interpolation Submit+Wait: RIFE " +
                    debugInterpolatedFrameLabel(prevFrameId) +
                    " [outputSlot=" + std::to_string(outputIndex) + "]");
                const auto processStart = std::chrono::high_resolution_clock::now();
                const int processRet = ncnnFrameInterpolator.processGpuRgbaFrames(
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
                result.rifeProcessMs = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - processStart).count();
                popNvtxRange();
                return processRet;
            };

            const auto queueWaitStart = std::chrono::high_resolution_clock::now();
            if (computeQueue == graphicsQueue || computeQueue == presentQueue) {
                pushNvtxRange("CPU Sync: Wait Shared Vulkan Queue Mutex for RIFE");
                std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
                result.queueWaitMs = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - queueWaitStart).count();
                popNvtxRange();
                result.processRet = processFrames();
            }
            else {
                pushNvtxRange("CPU Sync: Wait Compute Queue Mutex for RIFE");
                std::lock_guard<std::mutex> queueLock(ncnnComputeQueueMutex);
                result.queueWaitMs = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - queueWaitStart).count();
                popNvtxRange();
                result.processRet = processFrames();
            }

            result.inferenceMs = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - start).count();
            popNvtxRange();
            return result;
        });
    }

    return startedAny;
}
#endif
