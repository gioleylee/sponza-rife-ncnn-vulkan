#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <filesystem>
#include <memory>
#include <string>
#include <future>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "rife_src/rife.h"

#if __has_include(<ncnn/net.h>) && __has_include(<ncnn/gpu.h>) && __has_include(<ncnn/layer.h>)
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/allocator.h>
#include <ncnn/layer.h>
#include "rife_src/rife_ops.h"
#define HAS_NCNN 1
#else
#define HAS_NCNN 0
#endif

#if HAS_NCNN && defined(HAS_WARP_VK_SHADER)
#define HAS_RIFE_WARP_VK HAS_WARP_VK_SHADER
#elif __has_include("rife_src/warp.comp.hex.h") && __has_include("rife_src/warp_pack4.comp.hex.h") && __has_include("rife_src/warp_pack8.comp.hex.h")
#define HAS_RIFE_WARP_VK 1
#else
#define HAS_RIFE_WARP_VK 0
#endif

#include "../src/rife/RifeRunner.h"

#include "AppTypes.h"
#include "validation_layers.h"

class HelloTriangleApplication {
public:
    void run();

private:
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
#if defined(_WIN32)
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHRFn = nullptr;
#endif

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    std::vector<VkImage> gNormalImages;
    std::vector<VkDeviceMemory> gNormalImageMemories;
    std::vector<VkImageView> gNormalImageViews;

    std::vector<VkImage> gAlbedoImages;
    std::vector<VkDeviceMemory> gAlbedoImageMemories;
    std::vector<VkImageView> gAlbedoImageViews;

    std::vector<VkImage> gPositionImages;
    std::vector<VkDeviceMemory> gPositionImageMemories;
    std::vector<VkImageView> gPositionImageViews;

    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout lightingDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout lightingPipelineLayout = VK_NULL_HANDLE;
    VkPipeline lightingPipeline = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

    VkImage fallbackImage = VK_NULL_HANDLE;
    VkDeviceMemory fallbackImageMemory = VK_NULL_HANDLE;
    VkImageView fallbackImageView = VK_NULL_HANDLE;
    VkSampler fallbackSampler = VK_NULL_HANDLE;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    std::vector<VkBuffer> cubeUniformBuffers;
    std::vector<VkDeviceMemory> cubeUniformBuffersMemory;
    std::vector<void*> cubeUniformBuffersMapped;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorPool lightingDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> lightingDescriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    std::vector<FrameCaptureBuffer> frameCaptureBuffers;
    std::vector<RifeOutputBuffer> rifeOutputBuffers;
    std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> pendingCaptureSlotByFrame = { UINT32_MAX, UINT32_MAX };
    bool hasRifeGpuFramePair = false;
    uint32_t currentRifeGpuFrameIndex = UINT32_MAX;
    uint32_t previousRifeGpuFrameIndex = UINT32_MAX;
    uint64_t capturedFrameCount = 0;
    double previousFrameCaptureProcessMs = 0.0;
    double lastFrameCaptureProcessMs = 0.0;
    double lastFramePairCaptureProcessMs = 0.0;
    VkDeviceSize rifeDisplayBufferSize = 0;
    bool hasRifeDisplayFrame = false;
    uint64_t nextRifeOutputSequence = 1;
#if HAS_NCNN
    std::future<AsyncRifeResult> asyncRifeInference;
    bool rifeInferenceInFlight = false;
    uint32_t asyncRifePrevFrameIndex = UINT32_MAX;
    uint32_t asyncRifeCurrFrameIndex = UINT32_MAX;
    uint32_t asyncRifeOutputIndex = UINT32_MAX;
    int rifeInferenceScaleDivisor = RIFE_INITIAL_INFERENCE_SCALE_DIVISOR;
    uint64_t rifeCompletedInferenceCount = 0;
#endif

    bool framebufferResized = false;

    std::vector<Vertex> modelVertices;
    std::vector<uint32_t> modelIndices;
    std::vector<Material> materials;
    std::vector<Submesh> submeshes;
    uint32_t rotatingCubeIndexOffset = 0;
    uint32_t rotatingCubeIndexCount = 0;
    glm::vec3 rotatingCubePosition = glm::vec3(0.0f);

    glm::vec3 cameraPos = glm::vec3(-3.5f, 1.0f, 0.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    float cameraYaw = -1.5f;
    float cameraPitch = 0.5f;
    float cameraSpeed = 2.5f;

    float modelScale = 0.01f;
    float elapsedTimeSeconds = 0.0f;

    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    float mouseSensitivity = 0.1f;

    bool showNormals = false;
    bool nKeyPressed = false;
    bool showAlbedo = false;
    bool bKeyPressed = false;
    bool showPosition = false;
    bool vKeyPressed = false;
    bool showSpecular = false;
    bool mKeyPressed = false;

    bool autoPanEnabled = false;
    bool tKeyPressed = false;
    bool oneKeyPressed = false;
    bool twoKeyPressed = false;
    float autoPanSpeedDegreesPerSecond = 8.0f;

#if HAS_NCNN
    RifeRunner rifeRunner;
    ncnn::Net net;
    bool ncnnInitialized = false;
    int ncnnRendererDeviceIndex = -1;
    bool ncnnModelLoaded = false;
    bool rifeModelAttachedToRenderer = false;
    bool rKeyPressed = false;
    bool rifeRealtimeInterpolationEnabled = false;
    bool rifeInferenceRequestWaitingForFramePair = false;
#endif

    void initWindow();

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    void initVulkan();

    void mainLoop();

    void cleanupSwapChain();

    void cleanup();

#if HAS_NCNN
    int findNcnnDeviceIndexForRenderer() const;

    void initNcnn();

    void shutdownNcnn();

    bool loadNcnnModel(const std::string& paramPath, const std::string& binPath);

    void tryLoadDefaultNcnnModel();
#endif

    void recreateSwapChain();

    void createInstance();

    void setupDebugMessenger();

    void createSurface();

    void pickPhysicalDevice();

    void createLogicalDevice();

    void createSwapChain();

    void createImageViews();

    VkFormat findDepthFormat();

    void createGBufferAttachments();

    void createRenderPass();

    void createDescriptorSetLayout();

    void createLightingDescriptorSetLayout();

    void createGraphicsPipeline();

    void createLightingPipeline();

    void createFramebuffers();

    void createDepthResources();

    void createCommandPool();

    void createVertexBuffer();

    void createIndexBuffer();


    void createUniformBuffers();

    void createDescriptorPool();

    void createDescriptorSets();

    void createLightingDescriptorPool();

    void createLightingDescriptorSets();

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                       VkBuffer& buffer, VkDeviceMemory& bufferMemory);

#if defined(_WIN32)
    bool createExportableFrameBuffer(VkDeviceSize size,
                                     VkBuffer& buffer,
                                     VkDeviceMemory& bufferMemory,
                                     HANDLE& externalHandle);
#endif

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    bool tryFindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& memoryTypeIndex);

    void createFrameProcessingResources();

    void cleanupFrameProcessingResources();

    uint32_t findAvailableRifeCaptureSlot() const;

    bool captureSwapchainImageForRife(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t captureSlot);

    void displayRifeFrameOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    void processCapturedFrameForSlot(uint32_t frameSlot);

    void createCommandBuffers();

    uint32_t recordCommandBuffer(VkCommandBuffer commandBuffer,
                                 uint32_t imageIndex,
                                 PresentationCommandMode mode);

    void createSyncObjects();

#if HAS_NCNN
    void waitForAsyncRifeInference();

    void pollAsyncRifeInference();

    bool submitAsyncRifeInferenceIfReady();
#endif

    void updateUniformBuffer(uint32_t currentImage);

    void drawFrame();

    void updateCameraFrontFromAngles();

    void processInput(float deltaTime);

    void processMouseLook();

    void loadModel(const std::string& path);

    void appendRotatingCubeGeometry();

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format,
        VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image, VkDeviceMemory& imageMemory);

    VkImageView createImageView(VkImage image, VkFormat format, uint32_t mipLevels);

    VkCommandBuffer beginSingleTimeCommands();

    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    void transitionImageLayout(VkImage image, VkFormat format,
        VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);

    void copyBufferToImage(VkBuffer buffer, VkImage image,
        uint32_t width, uint32_t height);

    void loadMaterialTextures();

    void generateMipmaps(VkImage image, VkFormat imageFormat,
        int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

    void createFallbackTexture();

    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    bool isDeviceSuitable(VkPhysicalDevice device);

    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    static std::vector<char> readFile(const std::string& filename);
};
