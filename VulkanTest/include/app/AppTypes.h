#pragma once

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <optional>
#include <string>
#include <vector>

inline constexpr uint32_t WIDTH = 1280;
inline constexpr uint32_t HEIGHT = 720;

inline constexpr int MAX_FRAMES_IN_FLIGHT = 2;
inline constexpr int NCNN_INITIAL_INFERENCE_SCALE_DIVISOR = 1;
inline constexpr int NCNN_MIN_INFERENCE_SCALE_DIVISOR = 1;
inline constexpr int NCNN_MAX_INFERENCE_SCALE_DIVISOR = 1;
inline constexpr double NCNN_TARGET_INFERENCE_MS = 10.0;
inline constexpr double NCNN_FAST_INFERENCE_MS = 6.0;
inline constexpr uint32_t OFFSCREEN_FRAME_HISTORY_COUNT = 6;
inline constexpr uint32_t NCNN_OUTPUT_BUFFER_COUNT = 3;
// Keep the async queue shallow until the NCNN/RIFE wrapper is proven safe at
// higher concurrency.
inline constexpr uint32_t MAX_NCNN_IN_FLIGHT = 2;

inline const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(_WIN32)
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
#endif
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> transferFamily;

    bool isComplete() {

        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;
    glm::vec3 normal;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, normal);

        return attributeDescriptions;
    }
};

struct SkinnedVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::uvec4 joints;
    glm::vec4 weights;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(SkinnedVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 5> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(SkinnedVertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(SkinnedVertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(SkinnedVertex, texCoord);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_UINT;
        attributeDescriptions[3].offset = offsetof(SkinnedVertex, joints);

        attributeDescriptions[4].binding = 0;
        attributeDescriptions[4].location = 4;
        attributeDescriptions[4].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[4].offset = offsetof(SkinnedVertex, weights);

        return attributeDescriptions;
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

inline constexpr uint32_t MAX_SKIN_JOINTS = 64;

struct SkinUBO {
    alignas(16) glm::mat4 jointMatrices[MAX_SKIN_JOINTS];
};

struct LightingPushConstants {
    alignas(16) glm::vec4 lightPos;
    alignas(16) glm::vec4 lightColor;
    alignas(4) float showNormals;
    alignas(4) float showAlbedo;
    alignas(4) float showPosition;
    alignas(4) float showSpecular;
    alignas(16) glm::vec4 cameraPos;
};

// A real rendered frame lives entirely on the GPU. imageView preserves SRGB
// rendering and presentation behavior; ncnnInputImageView reads the same image
// through a compatible UNORM view so NCNN sees the stored normalized color
// values without an image-to-buffer copy or implicit SRGB decoding.
struct OffscreenFrame {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkImageView ncnnInputImageView = VK_NULL_HANDLE;
    VkBuffer gpuBuffer = VK_NULL_HANDLE;
    VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint64_t debugFrameId = UINT64_MAX;
    bool debugPresented = false;
};

struct NcnnOutputBuffer {
    VkBuffer gpuBuffer = VK_NULL_HANDLE;
    VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    bool ready = false;
    bool inUseByInference = false;
    bool inUseByGraphics = false;
    uint32_t graphicsFrameSlot = UINT32_MAX;
    uint64_t sequence = 0;
    uint64_t debugPreviousFrameId = UINT64_MAX;
    uint64_t debugCurrentFrameId = UINT64_MAX;
};

struct AsyncNcnnResult {
    int processRet = -1;
    double inferenceMs = 0.0;
    double queueWaitMs = 0.0;
    double rifeProcessMs = 0.0;
    int inputW = 0;
    int inputH = 0;
    int inferenceW = 0;
    int inferenceH = 0;
    uint32_t outputIndex = UINT32_MAX;
    uint32_t currentSourceIndex = UINT32_MAX;
    uint32_t previousSourceIndex = UINT32_MAX;
    uint64_t previousFrameId = UINT64_MAX;
    uint64_t currentFrameId = UINT64_MAX;
    uint32_t interpolationTargetIndex = UINT32_MAX;
};

enum class InterpolationTargetState {
    Pending,
    Running,
    Ready,
    Presenting,
    Released,
    Dropped
};

struct FrameInterpolationTarget {
    uint64_t previousFrameId = UINT64_MAX;
    uint64_t currentFrameId = UINT64_MAX;
    uint32_t previousSourceIndex = UINT32_MAX;
    uint32_t currentSourceIndex = UINT32_MAX;
    uint32_t outputIndex = UINT32_MAX;
    InterpolationTargetState state = InterpolationTargetState::Pending;
    bool waitingForFutureSource = false;
    std::string reason;
#if HAS_NCNN
    std::future<AsyncNcnnResult> future;
#endif
};

struct NcnnPresentationState {
    bool hasNcnnGpuFramePair = false;
    uint32_t currentNcnnGpuFrameIndex = UINT32_MAX;
    uint32_t previousNcnnGpuFrameIndex = UINT32_MAX;
    bool hasNcnnDisplayFrame = false;
    uint64_t nextNcnnOutputSequence = 1;
    uint32_t ncnnPendingInterpolatedOutputIndex = UINT32_MAX;
    uint32_t ncnnPendingSourceDisplayIndex = UINT32_MAX;
    uint32_t ncnnHeldSourceDisplayIndex = UINT32_MAX;
    uint32_t ncnnLastPresentedSourceIndex = UINT32_MAX;
    bool ncnnRenderAheadPending = false;
    uint32_t ncnnRunningJobCount = 0;
    int ncnnInferenceScaleDivisor = NCNN_INITIAL_INFERENCE_SCALE_DIVISOR;
    uint64_t ncnnCompletedInferenceCount = 0;
    bool ncnnRealtimeInterpolationEnabled = false;
    bool ncnnInferenceRequestWaitingForFramePair = false;
};

enum class PresentationCommandMode {
    RenderFrame,
    DisplayInterpolatedFrame,
    DisplayCapturedSourceFrame,
    DisplayHeldSourceFrame
};

struct Material {
    std::string diffusePath;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct Submesh {
    uint32_t indexOffset;
    uint32_t indexCount;
    uint32_t materialIndex;
};

struct SkinnedNode {
    int parent = -1;
    std::vector<int> children;
    glm::vec3 baseTranslation = glm::vec3(0.0f);
    glm::quat baseRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 baseScale = glm::vec3(1.0f);
    glm::mat4 baseMatrix = glm::mat4(1.0f);
    bool hasMatrix = false;
};

struct AnimationSampler {
    std::vector<float> times;
    std::vector<glm::vec4> values;
};

struct AnimationChannel {
    enum class Path {
        Translation,
        Rotation,
        Scale
    };

    int targetNode = -1;
    int samplerIndex = -1;
    Path path = Path::Translation;
};

struct SkinnedModel {
    std::vector<SkinnedVertex> vertices;
    std::vector<uint16_t> indices;
    std::vector<SkinnedNode> nodes;
    std::vector<int> jointNodeIndices;
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<AnimationSampler> animationSamplers;
    std::vector<AnimationChannel> animationChannels;
    std::vector<glm::mat4> localTransforms;
    std::vector<glm::mat4> globalTransforms;
    float animationTime = 0.0f;
    float animationDuration = 0.0f;
    Material material;
};
