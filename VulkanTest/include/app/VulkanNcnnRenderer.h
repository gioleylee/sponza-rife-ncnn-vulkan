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
#include <glm/gtc/constants.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "rife.h"

#if !defined(HAS_NCNN)
#if __has_include(<ncnn/net.h>) && __has_include(<ncnn/gpu.h>) && __has_include(<ncnn/layer.h>)
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/allocator.h>
#include <ncnn/layer.h>
#include "rife_ops.h"
#define HAS_NCNN 1
#else
#define HAS_NCNN 0
#endif
#elif HAS_NCNN
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/allocator.h>
#include <ncnn/layer.h>
#include "rife_ops.h"
#endif

#if HAS_NCNN && defined(HAS_WARP_VK_SHADER)
#define HAS_NCNN_WARP_VK HAS_WARP_VK_SHADER
#elif __has_include("warp.comp.hex.h") && __has_include("warp_pack4.comp.hex.h") && __has_include("warp_pack8.comp.hex.h")
#define HAS_NCNN_WARP_VK 1
#else
#define HAS_NCNN_WARP_VK 0
#endif

#include "NcnnFrameInterpolator.h"
#include "CpuProfiler.h"
#include "VulkanFramePacer.h"

#include "AppTypes.h"
#include "validation_layers.h"

class VulkanNcnnRenderer {
public:
    void run();

private:
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXTFn = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXTFn = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXTFn = nullptr;
    PFN_vkQueueBeginDebugUtilsLabelEXT vkQueueBeginDebugUtilsLabelEXTFn = nullptr;
    PFN_vkQueueEndDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXTFn = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
    uint32_t presentQueueFamilyIndex = UINT32_MAX;
    uint32_t computeQueueFamilyIndex = UINT32_MAX;
    uint32_t graphicsQueueIndex = 0;
    uint32_t presentQueueIndex = 0;
    uint32_t computeQueueIndex = 0;
    std::mutex vulkanQueueMutex;
    std::mutex swapchainOperationMutex;
    std::mutex ncnnComputeQueueMutex;
    VkSemaphore nativeFrameTimelineSemaphore = VK_NULL_HANDLE;
    VkSemaphore interpolationTimelineSemaphore = VK_NULL_HANDLE;
    VkCommandPool ncnnInteropCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer ncnnInteropCommandBuffer = VK_NULL_HANDLE;
    ProfileGpuContext tracyGraphicsContext = nullptr;
    ProfileGpuContext tracyComputeContext = nullptr;
    uint64_t nextNativeFrameTimelineValue = 1;
    std::atomic<uint64_t> nextInterpolationTimelineValue = 1;
    uint64_t pendingGraphicsInterpolationWaitValue = 0;
    VulkanFramePacer framePacer;
    uint64_t swapchainGeneration = 1;
    bool framePacerStarted = false;
#if defined(_WIN32)
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHRFn = nullptr;
#endif

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass imguiRenderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> imguiFramebuffers;
    std::vector<VkFramebuffer> offscreenFramebuffers;

    std::vector<VkImage> gNormalImages;
    std::vector<VkDeviceMemory> gNormalImageMemories;
    std::vector<VkImageView> gNormalImageViews;

    std::vector<VkImage> gAlbedoImages;
    std::vector<VkDeviceMemory> gAlbedoImageMemories;
    std::vector<VkImageView> gAlbedoImageViews;

    std::vector<VkImage> gPositionImages;
    std::vector<VkDeviceMemory> gPositionImageMemories;
    std::vector<VkImageView> gPositionImageViews;

    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthImageMemories;
    std::vector<VkImageView> depthImageViews;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline skinnedGraphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout lightingDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout lightingPipelineLayout = VK_NULL_HANDLE;
    VkPipeline lightingPipeline = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;
    VkBuffer skinnedVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory skinnedVertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer skinnedIndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory skinnedIndexBufferMemory = VK_NULL_HANDLE;

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
    std::vector<VkBuffer> cesiumUniformBuffers;
    std::vector<VkDeviceMemory> cesiumUniformBuffersMemory;
    std::vector<void*> cesiumUniformBuffersMapped;
    std::vector<VkBuffer> skinUniformBuffers;
    std::vector<VkDeviceMemory> skinUniformBuffersMemory;
    std::vector<void*> skinUniformBuffersMapped;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkDescriptorPool lightingDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> lightingDescriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    std::vector<OffscreenFrame> offscreenFrames;
    std::vector<NcnnOutputBuffer> ncnnOutputBuffers;
    std::vector<FrameInterpolationTarget> ncnnInterpolationTargets;
    std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> pendingCaptureSlotByFrame = { UINT32_MAX, UINT32_MAX };
    NcnnPresentationState ncnnPresentationState;
    uint64_t capturedFrameCount = 0;
    uint64_t nextDebugFrameId = 0;
    std::string debugLastPresentedFrameLabel = "none";
    int64_t debugLastPresentedTimelineStep = -1;
    std::chrono::steady_clock::time_point benchmarkNextPresentTime{};
    std::chrono::steady_clock::time_point benchmarkNextRealFrameTime{};
    std::chrono::steady_clock::time_point fpsStatsWindowStart{};
    PresentedFrameKind pendingPresentedFrameKind = PresentedFrameKind::None;
    uint32_t fpsPresentedFrameCount = 0;
    uint32_t fpsRealFrameCount = 0;
    uint32_t fpsInterpolatedFrameCount = 0;
    float displayedPresentedFps = 0.0f;
    float displayedRealFps = 0.0f;
    float displayedInterpolatedFps = 0.0f;
    double previousFrameCaptureProcessMs = 0.0;
    double lastFrameCaptureProcessMs = 0.0;
    double lastFramePairCaptureProcessMs = 0.0;
    VkDeviceSize ncnnDisplayBufferSize = 0;

    bool framebufferResized = false;

    std::vector<Vertex> modelVertices;
    std::vector<uint32_t> modelIndices;
    std::vector<Material> materials;
    std::vector<Submesh> submeshes;
    SkinnedModel cesiumMan;
    uint32_t rotatingCubeIndexOffset = 0;
    uint32_t rotatingCubeIndexCount = 0;
    glm::vec3 rotatingCubePosition = glm::vec3(0.0f);
    float rotatingCubeYawRadians = 0.0f;
    float rotatingCubePitchRadians = 0.0f;

    glm::vec3 cameraPos = glm::vec3(-3.5f, 1.0f, 0.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

    float cameraYaw = -1.5f;
    float cameraPitch = 0.5f;
    float cameraSpeed = 2.5f;

    float modelScale = 0.01f;
    float elapsedTimeSeconds = 0.0f;
    float frameDeltaSeconds = 0.0f;

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
    bool showInterpolationDebugPanel = false;
    bool markInterpolatedFrames = false;
    bool yKeyPressed = false;
    bool showFpsCounter = true;
    bool unlimitedFramerate = false;
    bool benchmarkModeEnabled = false;
    bool uKeyPressed = false;
    bool imguiVisible = false;
    bool rightMousePressed = false;
    bool imguiInitialized = false;

    bool autoPanEnabled = false;
    bool tKeyPressed = false;
    bool oneKeyPressed = false;
    bool twoKeyPressed = false;
    float autoPanSpeedDegreesPerSecond = 8.0f;

#if HAS_NCNN
    NcnnFrameInterpolator ncnnFrameInterpolator;
    ncnn::Net net;
    bool ncnnInitialized = false;
    int ncnnRendererDeviceIndex = -1;
    bool ncnnCanRunWithoutQueueMutex = false;
    bool ncnnModelLoaded = false;
    bool ncnnModelAttachedToRenderer = false;
    bool rKeyPressed = false;
#endif

    void initWindow();

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    void initVulkan();

    void initializeCoreVulkan();

    void initializeSwapchainResources();

    void initializeRenderResources();

    void initializeSceneResources();

    void initializeDescriptorResources();

    void initializeCommandResources();

    void initializeImGuiResources();

    void initializeSyncResources();

    void initializeOptionalNcnn();

    void initializeFramePacer();

    void shutdownFramePacer();

    void initializeTracyGpuContexts();

    void shutdownTracyGpuContexts();

    void mainLoop();

    void cleanup();

#include "Swapchain.h"

    void createInstance();

    void setupDebugMessenger();

    void loadDebugUtilsFunctions();

    void loadNvtxFunctions();

    void unloadNvtxFunctions();

    void pushNvtxRange(const std::string& name);

    void popNvtxRange();

    void markNvtxInstant(const std::string& name);

    void setDebugObjectName(VkObjectType objectType, uint64_t objectHandle, const std::string& name);

    void beginDebugLabel(VkCommandBuffer commandBuffer, const std::string& name, const glm::vec4& color);

    void endDebugLabel(VkCommandBuffer commandBuffer);

    void beginQueueDebugLabel(VkQueue queue, const std::string& name, const glm::vec4& color);

    void endQueueDebugLabel(VkQueue queue);

    void createSurface();

    void pickPhysicalDevice();

    void createLogicalDevice();

    VkFormat findDepthFormat();

#include "DescriptorResources.h"

    void createCommandPool();

    void createVertexBuffer();

    void createIndexBuffer();

    void createSkinnedVertexBuffer();

    void createSkinnedIndexBuffer();

    void createUniformBuffers();

#include "VulkanHelpers.h"

    void createImGuiRenderPass();

    void createImGuiFramebuffers();

    void cleanupImGuiFramebuffers();

    void cleanupImGui();

    void beginImGuiFrame();

    void renderImGuiOverlay(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    void copyOffscreenImageToSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t offscreenSlot);

    void resetFrameInterpolationDebugState();

    std::string debugRealFrameLabel(uint64_t frameId) const;

    std::string debugInterpolatedFrameLabel(uint64_t previousFrameId) const;

    bool isDebugRealFramePresented(uint64_t frameId) const;

    uint32_t findEarliestUnpresentedRealFrameSlot() const;

    uint32_t findReadyInterpolatedOutputForPreviousFrame(uint64_t previousFrameId) const;

    const char* interpolationTargetStateName(InterpolationTargetState state) const;

    uint32_t findInterpolationTargetIndex(uint64_t previousFrameId) const;

    uint32_t findInterpolationTargetIndexForOutput(uint32_t outputIndex) const;

    uint32_t findOffscreenSlotForDebugFrame(uint64_t frameId) const;

    void createInterpolationTargetIfNeeded(uint32_t previousSourceIndex, uint32_t currentSourceIndex);

    uint32_t createWaitingInterpolationTargetIfNeeded(uint64_t previousFrameId, uint32_t previousSourceIndex);

    void updateWaitingInterpolationTargets();

    void dropInterpolationTarget(uint32_t targetIndex, const std::string& reason);

    void releaseObsoleteNcnnOutputBuffers();

    void markDebugRealFramePresented(uint32_t sourceIndex);

    void markDebugInterpolatedFramePresented(uint64_t previousFrameId);

    void processCapturedFrameForSlot(uint32_t frameSlot);

    void createCommandBuffers();

    uint32_t recordCommandBuffer(VkCommandBuffer commandBuffer,
                                 uint32_t imageIndex,
                                 PresentationCommandMode mode);

    void createSyncObjects();

#include "NcnnIntegration.h"

    void updateUniformBuffer(uint32_t currentImage);

    void updateSkinAnimation(float deltaTime, uint32_t currentImage);

    bool acquireFrame(uint32_t& imageIndex);

    bool updateFrameState(uint32_t frameSlot);

    void setBenchmarkModeEnabled(bool enabled);

    uint32_t recordMainRenderCommands(uint32_t frameSlot,
                                      uint32_t imageIndex,
                                      PresentationCommandMode mode);

    void submitGraphicsWork(uint32_t frameSlot, uint32_t imageIndex, uint32_t capturedNcnnSlot);

    void waitForNativeFramesOnNcnnQueue(uint64_t timelineValue);

    void signalInterpolationOnNcnnQueue(uint64_t timelineValue);

    void handlePresentation(uint32_t imageIndex);

    VkResult presentPreparedFrame(const VulkanPresentJob& job);

    bool processFramePacerCompletions();

    void recordPresentedFrameStats(PresentedFrameKind frameKind);

    void advanceFrameIndex();

    void drawFrame();

    void updateCameraFrontFromAngles();

    void processInput(float deltaTime);

    void processMouseLook();

#include "ModelLoader.h"

#include "TextureLoader.h"

    bool isDeviceSuitable(VkPhysicalDevice device);

    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

#include "RenderResources.h"
};
