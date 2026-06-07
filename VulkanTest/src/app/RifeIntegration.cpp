// Owns NCNN/RIFE initialization, frame-processing resources, async inference, and RIFE cleanup.
#include "VulkanRifeRendererApp.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>

namespace {

void resetRifeDisplayState(RifePresentationState& state) {
    state.hasRifeDisplayFrame = false;
    state.nextRifeOutputSequence = 1;
    state.rifePendingInterpolatedOutputIndex = UINT32_MAX;
    state.rifePendingSourceDisplayIndex = UINT32_MAX;
    state.rifeHeldSourceDisplayIndex = UINT32_MAX;
    state.rifeLastPresentedSourceIndex = UINT32_MAX;
    state.rifeRenderAheadPending = false;
}

void resetRifeFramePairState(RifePresentationState& state) {
    state.hasRifeGpuFramePair = false;
    state.currentRifeGpuFrameIndex = UINT32_MAX;
    state.previousRifeGpuFrameIndex = UINT32_MAX;
}

void resetRifeAsyncState(RifePresentationState& state) {
    state.rifeInferenceInFlight = false;
    state.asyncRifePrevFrameIndex = UINT32_MAX;
    state.asyncRifeCurrFrameIndex = UINT32_MAX;
    state.asyncRifeOutputIndex = UINT32_MAX;
}

void resetRifeAdaptiveInferenceState(RifePresentationState& state) {
    resetRifeAsyncState(state);
    state.rifeInferenceScaleDivisor = RIFE_INITIAL_INFERENCE_SCALE_DIVISOR;
    state.rifeCompletedInferenceCount = 0;
}

void resetRifePendingPresentationState(RifePresentationState& state) {
    state.rifePendingInterpolatedOutputIndex = UINT32_MAX;
    state.rifePendingSourceDisplayIndex = UINT32_MAX;
    state.rifeHeldSourceDisplayIndex = UINT32_MAX;
    state.rifeLastPresentedSourceIndex = UINT32_MAX;
    state.rifeRenderAheadPending = false;
}

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
    const std::string optimizedParam = "assets/nn_models/rife-v4/flownet-opt.param";
    const std::string optimizedBin = "assets/nn_models/rife-v4/flownet-opt.bin";
    const std::string defaultParam = "assets/nn_models/rife-v4/flownet.param";
    const std::string defaultBin = "assets/nn_models/rife-v4/flownet.bin";

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

#if defined(_WIN32)
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
#endif

void VulkanRifeRendererApp::createFrameProcessingResources() {
    cleanupFrameProcessingResources();

    const VkDeviceSize frameSize =
        static_cast<VkDeviceSize>(swapChainExtent.width) *
        static_cast<VkDeviceSize>(swapChainExtent.height) * 4;

    offscreenFrames.resize(OFFSCREEN_FRAME_HISTORY_COUNT);
    rifeOutputBuffers.resize(RIFE_OUTPUT_BUFFER_COUNT);

    rifeDisplayBufferSize = frameSize;
    resetRifeDisplayState(rifePresentationState);

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
    resetRifeFramePairState(rifePresentationState);
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
    resetRifeDisplayState(rifePresentationState);
#if HAS_NCNN
    resetRifeAdaptiveInferenceState(rifePresentationState);
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
    resetRifeFramePairState(rifePresentationState);
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
}

uint32_t VulkanRifeRendererApp::findAvailableOffscreenFrameSlot() const {
    for (uint32_t slot = 0; slot < offscreenFrames.size(); ++slot) {
#if HAS_NCNN
        if (rifePresentationState.rifeInferenceInFlight &&
            (slot == rifePresentationState.asyncRifePrevFrameIndex || slot == rifePresentationState.asyncRifeCurrFrameIndex)) {
            continue;
        }
#endif
        if (slot == rifePresentationState.currentRifeGpuFrameIndex) {
            continue;
        }
        if (slot == rifePresentationState.rifePendingSourceDisplayIndex) {
            continue;
        }
        if (slot == rifePresentationState.rifeHeldSourceDisplayIndex) {
            continue;
        }
        if (slot == rifePresentationState.rifeLastPresentedSourceIndex) {
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
    if (!rifePresentationState.hasRifeDisplayFrame ||
        rifePresentationState.rifePendingInterpolatedOutputIndex >= rifeOutputBuffers.size() ||
        rifeOutputBuffers.empty()) {
        return;
    }

    auto& selectedOutput = rifeOutputBuffers[rifePresentationState.rifePendingInterpolatedOutputIndex];
    if (!selectedOutput.ready ||
        selectedOutput.inUseByGraphics ||
        selectedOutput.gpuBuffer == VK_NULL_HANDLE ||
        selectedOutput.size == 0) {
        rifePresentationState.hasRifeDisplayFrame = false;
        rifePresentationState.rifePendingInterpolatedOutputIndex = UINT32_MAX;
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

    rifePresentationState.hasRifeDisplayFrame = false;
    rifePresentationState.rifePendingInterpolatedOutputIndex = UINT32_MAX;
    rifePresentationState.rifeHeldSourceDisplayIndex = UINT32_MAX;
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
    if (rifePresentationState.rifePendingSourceDisplayIndex >= offscreenFrames.size()) {
        rifePresentationState.rifePendingSourceDisplayIndex = UINT32_MAX;
        return;
    }

    displayRifeSourceBufferOnSwapchain(commandBuffer, imageIndex, rifePresentationState.rifePendingSourceDisplayIndex);
    rifePresentationState.rifeLastPresentedSourceIndex = rifePresentationState.rifePendingSourceDisplayIndex;
    rifePresentationState.rifePendingSourceDisplayIndex = UINT32_MAX;
    rifePresentationState.rifeHeldSourceDisplayIndex = UINT32_MAX;
    rifePresentationState.rifeRenderAheadPending = false;
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

    resetRifeAsyncState(rifePresentationState);
    resetRifePendingPresentationState(rifePresentationState);
}

void VulkanRifeRendererApp::pollAsyncRifeInference() {
    if (!asyncRifeInference.valid()) {
        rifePresentationState.rifeInferenceInFlight = false;
        return;
    }

    if (asyncRifeInference.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        return;
    }

    const AsyncRifeResult result = asyncRifeInference.get();
    if (result.outputIndex < rifeOutputBuffers.size()) {
        rifeOutputBuffers[result.outputIndex].inUseByInference = false;
    }
    resetRifeAsyncState(rifePresentationState);

    if (result.processRet == 0) {
        if (result.outputIndex >= rifeOutputBuffers.size()) {
            std::cerr << "[RIFE] async GPU interpolation finished with invalid output slot" << std::endl;
            return;
        }

        ++rifePresentationState.rifeCompletedInferenceCount;
        const int previousDivisor = rifePresentationState.rifeInferenceScaleDivisor;
        if (result.inferenceMs > RIFE_TARGET_INFERENCE_MS && rifePresentationState.rifeInferenceScaleDivisor < RIFE_MAX_INFERENCE_SCALE_DIVISOR) {
            ++rifePresentationState.rifeInferenceScaleDivisor;
        }
        else if (result.inferenceMs < RIFE_FAST_INFERENCE_MS && rifePresentationState.rifeInferenceScaleDivisor > RIFE_MIN_INFERENCE_SCALE_DIVISOR) {
            --rifePresentationState.rifeInferenceScaleDivisor;
        }

        rifeOutputBuffers[result.outputIndex].ready = true;
        rifeOutputBuffers[result.outputIndex].sequence = rifePresentationState.nextRifeOutputSequence++;
        rifePresentationState.rifePendingInterpolatedOutputIndex = result.outputIndex;
        rifePresentationState.rifePendingSourceDisplayIndex = result.currentSourceIndex;
        rifePresentationState.hasRifeDisplayFrame = true;
        if (previousDivisor != rifePresentationState.rifeInferenceScaleDivisor || (rifePresentationState.rifeCompletedInferenceCount % 120) == 1) {
            std::cout << "[RIFE] async inference"
                      << " display=" << result.inputW << "x" << result.inputH
                      << ", inference=" << result.inferenceW << "x" << result.inferenceH
                      << ", scale_divisor=" << rifePresentationState.rifeInferenceScaleDivisor
                      << ", inference_ms=" << result.inferenceMs << std::endl;
        }
        return;
    }

    std::cerr << "[RIFE] async GPU interpolation failed"
              << " (code=" << result.processRet
              << ", inference_ms=" << result.inferenceMs << ")" << std::endl;
    // Do not leave the scheduler holding N forever if interpolation fails.
    // Advance to N+1 on the next presentation tick and resume render-ahead.
    rifePresentationState.rifePendingSourceDisplayIndex = result.currentSourceIndex;
}

bool VulkanRifeRendererApp::submitAsyncRifeInferenceIfReady() {
    if (!rifePresentationState.rifeRealtimeInterpolationEnabled ||
        !rifeModelAttachedToRenderer ||
        rifePresentationState.rifeInferenceInFlight ||
        !rifePresentationState.hasRifeGpuFramePair ||
        capturedFrameCount < 2 ||
        rifePresentationState.previousRifeGpuFrameIndex >= offscreenFrames.size() ||
        rifePresentationState.currentRifeGpuFrameIndex >= offscreenFrames.size() ||
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

    const uint32_t prevIndex = rifePresentationState.previousRifeGpuFrameIndex;
    const uint32_t currIndex = rifePresentationState.currentRifeGpuFrameIndex;
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
        rifePresentationState.rifeInferenceScaleDivisor,
        RIFE_MIN_INFERENCE_SCALE_DIVISOR,
        RIFE_MAX_INFERENCE_SCALE_DIVISOR
    );
    const int inferenceW = std::min(inputW, std::max(32, inputW / divisor));
    const int inferenceH = std::min(inputH, std::max(32, inputH / divisor));

    rifePresentationState.rifeInferenceInFlight = true;
    rifePresentationState.asyncRifePrevFrameIndex = prevIndex;
    rifePresentationState.asyncRifeCurrFrameIndex = currIndex;
    rifePresentationState.asyncRifeOutputIndex = outputIndex;
    rifeOutputBuffers[outputIndex].inUseByInference = true;
    rifeOutputBuffers[outputIndex].ready = false;
    rifeOutputBuffers[outputIndex].sequence = 0;
    rifePresentationState.hasRifeGpuFramePair = false;
    rifePresentationState.rifeHeldSourceDisplayIndex = prevIndex;
    rifePresentationState.rifeInferenceRequestWaitingForFramePair = false;

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
