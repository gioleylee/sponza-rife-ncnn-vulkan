#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "HelloTriangleApplication.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

void HelloTriangleApplication::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void HelloTriangleApplication::initVulkan() {
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

void HelloTriangleApplication::mainLoop() {
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

void HelloTriangleApplication::cleanup() {
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
int HelloTriangleApplication::findNcnnDeviceIndexForRenderer() const {
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

void HelloTriangleApplication::initNcnn() {
    if (ncnnInitialized) {
        return;
    }

    ncnn::create_gpu_instance();

    const int gpuCount = ncnn::get_gpu_count();
    ncnnRendererDeviceIndex = findNcnnDeviceIndexForRenderer();
    net.opt.use_vulkan_compute = gpuCount > 0 && HAS_RIFE_WARP_VK;
    net.opt.use_cooperative_matrix = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = 1;
    if (gpuCount > 0 && !HAS_RIFE_WARP_VK) {
        std::cout << "[NCNN] rife warp vulkan shaders not found; forcing CPU inference path" << std::endl;
    }
    ncnnInitialized = true;

    std::cout << "[NCNN] initialized (gpu_count=" << gpuCount
              << ", renderer_gpu_index=" << ncnnRendererDeviceIndex
              << ", vulkan_compute=" << (net.opt.use_vulkan_compute ? "on" : "off") << ")" << std::endl;
}

void HelloTriangleApplication::shutdownNcnn() {
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

bool HelloTriangleApplication::loadNcnnModel(const std::string& paramPath, const std::string& binPath) {
    if (!ncnnInitialized) {
        initNcnn();
    }

    if (!rifeRunner.initialize(net, paramPath, binPath, device, ncnnRendererDeviceIndex)) {
        return false;
    }

    ncnnModelLoaded = true;

    std::cout << "[NCNN] model loaded: " << paramPath << " + " << binPath << std::endl;
    return true;
}

void HelloTriangleApplication::tryLoadDefaultNcnnModel() {
    const std::string defaultParam = "assets/models/rife-v4/flownet.param";
    const std::string defaultBin = "assets/models/rife-v4/flownet.bin";

    if (!std::filesystem::exists(defaultParam) || !std::filesystem::exists(defaultBin)) {
        std::cout << "[NCNN] default RIFE model not found: "
                  << defaultParam << " and " << defaultBin << std::endl;
        return;
    }

    if (loadNcnnModel(defaultParam, defaultBin)) {
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

void HelloTriangleApplication::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

void HelloTriangleApplication::createDescriptorPool() {
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

void HelloTriangleApplication::createDescriptorSets() {
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

void HelloTriangleApplication::createLightingDescriptorPool() {
    uint32_t imageCount = static_cast<uint32_t>(swapChainImages.size());

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

void HelloTriangleApplication::createLightingDescriptorSets() {
    uint32_t imageCount = static_cast<uint32_t>(swapChainImages.size());

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

void HelloTriangleApplication::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                   VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

bool HelloTriangleApplication::createExportableFrameBuffer(VkDeviceSize size,
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

void HelloTriangleApplication::createFrameProcessingResources() {
    cleanupFrameProcessingResources();

    const VkDeviceSize frameSize =
        static_cast<VkDeviceSize>(swapChainExtent.width) *
        static_cast<VkDeviceSize>(swapChainExtent.height) * 4;

    frameCaptureBuffers.resize(std::max<uint32_t>(RIFE_CAPTURE_BUFFER_COUNT, MAX_FRAMES_IN_FLIGHT + 2));
    rifeOutputBuffers.resize(RIFE_OUTPUT_BUFFER_COUNT);

    rifeDisplayBufferSize = frameSize;
    hasRifeDisplayFrame = false;
    nextRifeOutputSequence = 1;

    for (auto& capture : frameCaptureBuffers) {
        createBuffer(
            frameSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            capture.gpuBuffer,
            capture.gpuMemory
        );

        capture.size = frameSize;
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

void HelloTriangleApplication::cleanupFrameProcessingResources() {
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
#if HAS_NCNN
    rifeInferenceInFlight = false;
    asyncRifePrevFrameIndex = UINT32_MAX;
    asyncRifeCurrFrameIndex = UINT32_MAX;
    asyncRifeOutputIndex = UINT32_MAX;
    rifeInferenceScaleDivisor = RIFE_INITIAL_INFERENCE_SCALE_DIVISOR;
    rifeCompletedInferenceCount = 0;
#endif

    for (auto& capture : frameCaptureBuffers) {
        if (capture.gpuBuffer) {
            vkDestroyBuffer(device, capture.gpuBuffer, nullptr);
            capture.gpuBuffer = VK_NULL_HANDLE;
        }

        if (capture.gpuMemory) {
            vkFreeMemory(device, capture.gpuMemory, nullptr);
            capture.gpuMemory = VK_NULL_HANDLE;
        }

        capture.size = 0;
    }

    frameCaptureBuffers.clear();
    pendingCaptureSlotByFrame.fill(UINT32_MAX);
    hasRifeGpuFramePair = false;
    currentRifeGpuFrameIndex = UINT32_MAX;
    previousRifeGpuFrameIndex = UINT32_MAX;
    capturedFrameCount = 0;
    previousFrameCaptureProcessMs = 0.0;
    lastFrameCaptureProcessMs = 0.0;
    lastFramePairCaptureProcessMs = 0.0;
}

uint32_t HelloTriangleApplication::findAvailableRifeCaptureSlot() const {
#if HAS_NCNN
    if (!rifeRealtimeInterpolationEnabled || !rifeModelAttachedToRenderer) {
        return UINT32_MAX;
    }
#else
    return UINT32_MAX;
#endif

    for (uint32_t slot = 0; slot < frameCaptureBuffers.size(); ++slot) {
#if HAS_NCNN
        if (rifeInferenceInFlight &&
            (slot == asyncRifePrevFrameIndex || slot == asyncRifeCurrFrameIndex)) {
            continue;
        }
#endif
        if (slot == currentRifeGpuFrameIndex) {
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

bool HelloTriangleApplication::captureSwapchainImageForRife(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t captureSlot) {
    if (imageIndex >= swapChainImages.size() || captureSlot >= frameCaptureBuffers.size()) {
        return false;
    }

    VkImageMemoryBarrier toTransferBarrier{};
    toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.image = swapChainImages[imageIndex];
    toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferBarrier.subresourceRange.baseMipLevel = 0;
    toTransferBarrier.subresourceRange.levelCount = 1;
    toTransferBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferBarrier.subresourceRange.layerCount = 1;
    toTransferBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toTransferBarrier
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

    vkCmdCopyImageToBuffer(
        commandBuffer,
        swapChainImages[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        frameCaptureBuffers[captureSlot].gpuBuffer,
        1,
        &region
    );

    VkBufferMemoryBarrier gpuBufferBarrier{};
    gpuBufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    gpuBufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    gpuBufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    gpuBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    gpuBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    gpuBufferBarrier.buffer = frameCaptureBuffers[captureSlot].gpuBuffer;
    gpuBufferBarrier.offset = 0;
    gpuBufferBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &gpuBufferBarrier,
        0,
        nullptr
    );

    VkImageMemoryBarrier backToPresentBarrier{};
    backToPresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    backToPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    backToPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    backToPresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToPresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToPresentBarrier.image = swapChainImages[imageIndex];
    backToPresentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    backToPresentBarrier.subresourceRange.baseMipLevel = 0;
    backToPresentBarrier.subresourceRange.levelCount = 1;
    backToPresentBarrier.subresourceRange.baseArrayLayer = 0;
    backToPresentBarrier.subresourceRange.layerCount = 1;
    backToPresentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
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

    return true;
}

void HelloTriangleApplication::displayRifeFrameOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (!hasRifeDisplayFrame || rifeOutputBuffers.empty()) {
        return;
    }

    uint32_t outputIndex = UINT32_MAX;
    uint64_t newestSequence = 0;
    for (uint32_t i = 0; i < rifeOutputBuffers.size(); ++i) {
        const auto& output = rifeOutputBuffers[i];
        if (output.ready && !output.inUseByGraphics && output.sequence >= newestSequence) {
            outputIndex = i;
            newestSequence = output.sequence;
        }
    }

    if (outputIndex == UINT32_MAX) {
        hasRifeDisplayFrame = false;
        for (const auto& output : rifeOutputBuffers) {
            if (output.ready) {
                hasRifeDisplayFrame = true;
                break;
            }
        }
        return;
    }

    auto& selectedOutput = rifeOutputBuffers[outputIndex];
    if (selectedOutput.gpuBuffer == VK_NULL_HANDLE || selectedOutput.size == 0) {
        selectedOutput.ready = false;
        return;
    }

    VkImageMemoryBarrier toTransferDstBarrier{};
    toTransferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDstBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
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
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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
    rifeOutputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    rifeOutputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    rifeOutputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rifeOutputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    rifeOutputBarrier.buffer = selectedOutput.gpuBuffer;
    rifeOutputBarrier.offset = 0;
    rifeOutputBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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
        selectedOutput.gpuBuffer,
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

    selectedOutput.ready = false;
    selectedOutput.inUseByGraphics = true;
    selectedOutput.graphicsFrameSlot = currentFrame;

    hasRifeDisplayFrame = false;
    for (auto& output : rifeOutputBuffers) {
        if (output.ready && output.sequence < newestSequence) {
            output.ready = false;
        }
        hasRifeDisplayFrame = hasRifeDisplayFrame || output.ready;
    }
}

void HelloTriangleApplication::processCapturedFrameForSlot(uint32_t frameSlot) {
    if (frameSlot >= pendingCaptureSlotByFrame.size()) {
        return;
    }

    const uint32_t captureSlot = pendingCaptureSlotByFrame[frameSlot];
    pendingCaptureSlotByFrame[frameSlot] = UINT32_MAX;

    if (captureSlot == UINT32_MAX || captureSlot >= frameCaptureBuffers.size()) {
        return;
    }

    const auto& capture = frameCaptureBuffers[captureSlot];
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
        previousRifeGpuFrameIndex < frameCaptureBuffers.size();

    ++capturedFrameCount;
#endif

}

void HelloTriangleApplication::createCommandBuffers() {
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

uint32_t HelloTriangleApplication::recordCommandBuffer(VkCommandBuffer commandBuffer,
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

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
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
        &lightingDescriptorSets[imageIndex],
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

    const uint32_t captureSlot = findAvailableRifeCaptureSlot();
    const bool capturedForRife = captureSlot != UINT32_MAX &&
        captureSwapchainImageForRife(commandBuffer, imageIndex, captureSlot);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    return capturedForRife ? captureSlot : UINT32_MAX;
}

void HelloTriangleApplication::createSyncObjects() {
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
void HelloTriangleApplication::waitForAsyncRifeInference() {
    if (asyncRifeInference.valid()) {
        asyncRifeInference.wait();
        AsyncRifeResult result = asyncRifeInference.get();
        if (result.outputIndex < rifeOutputBuffers.size()) {
            rifeOutputBuffers[result.outputIndex].inUseByInference = false;
            rifeOutputBuffers[result.outputIndex].ready = false;
        }
    }

    rifeInferenceInFlight = false;
    asyncRifePrevFrameIndex = UINT32_MAX;
    asyncRifeCurrFrameIndex = UINT32_MAX;
    asyncRifeOutputIndex = UINT32_MAX;
}

void HelloTriangleApplication::pollAsyncRifeInference() {
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
}

bool HelloTriangleApplication::submitAsyncRifeInferenceIfReady() {
    if (!rifeRealtimeInterpolationEnabled ||
        !rifeModelAttachedToRenderer ||
        rifeInferenceInFlight ||
        !hasRifeGpuFramePair ||
        capturedFrameCount < 2 ||
        previousRifeGpuFrameIndex >= frameCaptureBuffers.size() ||
        currentRifeGpuFrameIndex >= frameCaptureBuffers.size() ||
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
    const auto prevGpu = frameCaptureBuffers[prevIndex];
    const auto currGpu = frameCaptureBuffers[currIndex];
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
    rifeInferenceRequestWaitingForFramePair = false;

    asyncRifeInference = std::async(std::launch::async, [this,
                                                         prevGpu,
                                                         currGpu,
                                                         outBuffer,
                                                         outMemory,
                                                         outSize,
                                                         inputW,
                                                         inputH,
                                                         inferenceW,
                                                         inferenceH,
                                                         outputIndex]() {
        AsyncRifeResult result{};
        result.inputW = inputW;
        result.inputH = inputH;
        result.inferenceW = inferenceW;
        result.inferenceH = inferenceH;
        result.outputIndex = outputIndex;

        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        const auto start = std::chrono::high_resolution_clock::now();
        result.processRet = rifeRunner.processGpuRgbaFrames(
            prevGpu.gpuBuffer,
            prevGpu.gpuMemory,
            prevGpu.size,
            currGpu.gpuBuffer,
            currGpu.gpuMemory,
            currGpu.size,
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

void HelloTriangleApplication::updateUniformBuffer(uint32_t currentImage) {
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

void HelloTriangleApplication::drawFrame() {
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

void HelloTriangleApplication::loadModel(const std::string& path) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices |
        aiProcess_PreTransformVertices |
        aiProcess_ImproveCacheLocality
    );

    if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        throw std::runtime_error(std::string("Failed to load model: ") + importer.GetErrorString());
    }

    modelVertices.clear();
    modelIndices.clear();
    materials.clear();
    submeshes.clear();

    materials.resize(scene->mNumMaterials);
    std::string baseDir;
    {
        size_t slash = path.find_last_of("/\\");
        baseDir = (slash == std::string::npos) ? "" : path.substr(0, slash + 1);
    }

    for (unsigned int m = 0; m < scene->mNumMaterials; ++m) {
        aiMaterial* mat = scene->mMaterials[m];
        Material material{};

        aiString texPath;
        if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
            material.diffusePath = baseDir + texPath.C_Str();
        }
        else {
            material.diffusePath.clear();
        }

        materials[m] = material;
    }

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh* mesh = scene->mMeshes[m];

        Submesh sm{};
        sm.indexOffset = static_cast<uint32_t>(modelIndices.size());
        sm.materialIndex = mesh->mMaterialIndex;

        size_t baseVertex = modelVertices.size();

        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            Vertex vertex{};
            const aiVector3D& pos = mesh->mVertices[i];

            vertex.pos = glm::vec3(pos.x, pos.y, pos.z);

            if (mesh->mTextureCoords[0]) {
                const aiVector3D& uv = mesh->mTextureCoords[0][i];
                float u = uv.x;
                float v = 1.0f - uv.y;
                vertex.texCoord = glm::vec2(u, v);
            }
            else {
                vertex.texCoord = glm::vec2(0.0f, 0.0f);
            }

            if (mesh->mNormals) {
                const aiVector3D& n = mesh->mNormals[i];
                vertex.normal = glm::normalize(glm::vec3(n.x, n.y, n.z));
            }
            else {
                vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            modelVertices.push_back(vertex);
        }

        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            const aiFace& face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue;

            modelIndices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[0]));
            modelIndices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[1]));
            modelIndices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[2]));
        }

        sm.indexCount = static_cast<uint32_t>(modelIndices.size()) - sm.indexOffset;
        if (sm.indexCount > 0) {
            submeshes.push_back(sm);
        }
    }

    if (modelVertices.empty() || modelIndices.empty() || submeshes.empty()) {
        throw std::runtime_error("Loaded model has no geometry.");
    }

}

void HelloTriangleApplication::appendRotatingCubeGeometry() {
    rotatingCubeIndexOffset = static_cast<uint32_t>(modelIndices.size());

    const float h = 0.5f;
    const float uvTile = 4.0f;
    const glm::vec2 uv00(0.0f, 0.0f);
    const glm::vec2 uv10(uvTile, 0.0f);
    const glm::vec2 uv11(uvTile, uvTile);
    const glm::vec2 uv01(0.0f, uvTile);

    auto addFace = [&](const glm::vec3& normal,
                       const glm::vec3& a,
                       const glm::vec3& b,
                       const glm::vec3& c,
                       const glm::vec3& d) {
        uint32_t base = static_cast<uint32_t>(modelVertices.size());
        modelVertices.push_back({ a, uv00, normal });
        modelVertices.push_back({ b, uv10, normal });
        modelVertices.push_back({ c, uv11, normal });
        modelVertices.push_back({ d, uv01, normal });
        modelIndices.push_back(base + 0);
        modelIndices.push_back(base + 1);
        modelIndices.push_back(base + 2);
        modelIndices.push_back(base + 2);
        modelIndices.push_back(base + 3);
        modelIndices.push_back(base + 0);
    };

    addFace(glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(-h, -h, h), glm::vec3(h, -h, h), glm::vec3(h, h, h), glm::vec3(-h, h, h));
    addFace(glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(h, -h, -h), glm::vec3(-h, -h, -h), glm::vec3(-h, h, -h), glm::vec3(h, h, -h));
    addFace(glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(h, -h, h), glm::vec3(h, -h, -h), glm::vec3(h, h, -h), glm::vec3(h, h, h));
    addFace(glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(-h, -h, -h), glm::vec3(-h, -h, h), glm::vec3(-h, h, h), glm::vec3(-h, h, -h));
    addFace(glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(-h, h, h), glm::vec3(h, h, h), glm::vec3(h, h, -h), glm::vec3(-h, h, -h));
    addFace(glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(-h, -h, -h), glm::vec3(h, -h, -h), glm::vec3(h, -h, h), glm::vec3(-h, -h, h));

    rotatingCubeIndexCount = static_cast<uint32_t>(modelIndices.size()) - rotatingCubeIndexOffset;
}

void HelloTriangleApplication::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format,
    VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

VkCommandBuffer HelloTriangleApplication::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void HelloTriangleApplication::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    {
        std::lock_guard<std::mutex> queueLock(vulkanQueueMutex);
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void HelloTriangleApplication::transitionImageLayout(VkImage image, VkFormat format,
    VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
        oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
        format == VK_FORMAT_D32_SFLOAT ||
        format == VK_FORMAT_D24_UNORM_S8_UINT ||
        format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else {
        throw std::runtime_error("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(cmd,
        srcStage, dstStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    endSingleTimeCommands(cmd);
}

void HelloTriangleApplication::copyBufferToImage(VkBuffer buffer, VkImage image,
    uint32_t width, uint32_t height) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(cmd, buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);

    endSingleTimeCommands(cmd);
}

void HelloTriangleApplication::loadMaterialTextures() {
    const std::string fallback = "";

    for (auto& mat : materials) {

        std::string path = mat.diffusePath.empty() ? fallback : mat.diffusePath;
        if (path.empty()) {
            continue;
        }

        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            std::cerr << "Failed to load texture: " << path << "\n";
            continue;
        }

        uint32_t mipLevels = static_cast<uint32_t>(
            std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        VkDeviceSize imageSize = texWidth * texHeight * 4;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            mipLevels,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            mat.image, mat.imageMemory);

        transitionImageLayout(mat.image, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);

        copyBufferToImage(stagingBuffer, mat.image,
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight));

        generateMipmaps(mat.image, VK_FORMAT_R8G8B8A8_SRGB,
            texWidth, texHeight, mipLevels);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        mat.imageView = createImageView(mat.image, VK_FORMAT_R8G8B8A8_SRGB, mipLevels);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = deviceFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
        samplerInfo.maxAnisotropy = deviceFeatures.samplerAnisotropy ? 16.0f : 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);

        if (vkCreateSampler(device, &samplerInfo, nullptr, &mat.sampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create sampler for material!");
        }
    }
}

void HelloTriangleApplication::generateMipmaps(VkImage image, VkFormat imageFormat,
    int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {

    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;

        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { std::max(mipWidth / 2, 1), std::max(mipHeight / 2, 1), 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit,
            VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    endSingleTimeCommands(cmd);
}

void HelloTriangleApplication::createFallbackTexture() {
    const uint32_t textureWidth = 64;
    const uint32_t textureHeight = 64;
    const auto rgba = [](uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
        return static_cast<uint32_t>(r) |
            (static_cast<uint32_t>(g) << 8) |
            (static_cast<uint32_t>(b) << 16) |
            (static_cast<uint32_t>(a) << 24);
    };

    std::vector<uint32_t> pixels(textureWidth * textureHeight);
    for (uint32_t y = 0; y < textureHeight; ++y) {
        for (uint32_t x = 0; x < textureWidth; ++x) {
            const bool checker = ((x / 8) + (y / 8)) % 2 == 0;
            uint32_t color = checker ? rgba(245, 245, 245) : rgba(12, 12, 12);

            if ((x % 16) < 3) {
                color = rgba(235, 35, 35);
            }
            if ((y % 16) < 3) {
                color = rgba(30, 220, 80);
            }
            if (x >= 29 && x <= 34) {
                color = rgba(40, 120, 255);
            }
            if (y >= 29 && y <= 34) {
                color = rgba(255, 210, 20);
            }

            pixels[y * textureWidth + x] = color;
        }
    }

    VkDeviceSize imageSize = sizeof(pixels[0]) * pixels.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data = nullptr;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels.data(), static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = textureWidth;
    imageInfo.extent.height = textureHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &fallbackImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fallback image!");
    }

    VkMemoryRequirements memRequirements{};
    vkGetImageMemoryRequirements(device, fallbackImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &fallbackImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate fallback image memory!");
    }

    vkBindImageMemory(device, fallbackImage, fallbackImageMemory, 0);

    transitionImageLayout(fallbackImage, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);

    copyBufferToImage(stagingBuffer, fallbackImage, textureWidth, textureHeight);

    transitionImageLayout(fallbackImage, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    fallbackImageView = createImageView(fallbackImage, VK_FORMAT_R8G8B8A8_SRGB, 1);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = deviceFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
    samplerInfo.maxAnisotropy = deviceFeatures.samplerAnisotropy ? 16.0f : 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &fallbackSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create fallback sampler!");
    }
}

int main() {
#if HAS_NCNN
    std::cout << "[NCNN] enabled" << std::endl;
#else
    std::cout << "[NCNN] disabled" << std::endl;
#endif

    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
