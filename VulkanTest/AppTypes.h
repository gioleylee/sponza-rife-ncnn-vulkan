#pragma once

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

inline constexpr uint32_t WIDTH = 1280;
inline constexpr uint32_t HEIGHT = 720;

inline constexpr int MAX_FRAMES_IN_FLIGHT = 2;
inline constexpr int RIFE_INITIAL_INFERENCE_SCALE_DIVISOR = 4;
inline constexpr int RIFE_MIN_INFERENCE_SCALE_DIVISOR = 2;
inline constexpr int RIFE_MAX_INFERENCE_SCALE_DIVISOR = 6;
inline constexpr double RIFE_TARGET_INFERENCE_MS = 10.0;
inline constexpr double RIFE_FAST_INFERENCE_MS = 6.0;
inline constexpr uint32_t RIFE_CAPTURE_BUFFER_COUNT = 4;
inline constexpr uint32_t RIFE_OUTPUT_BUFFER_COUNT = 3;

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

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
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

struct FrameCaptureBuffer {
    VkBuffer gpuBuffer = VK_NULL_HANDLE;
    VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct RifeOutputBuffer {
    VkBuffer gpuBuffer = VK_NULL_HANDLE;
    VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    bool ready = false;
    bool inUseByInference = false;
    bool inUseByGraphics = false;
    uint32_t graphicsFrameSlot = UINT32_MAX;
    uint64_t sequence = 0;
};

struct AsyncRifeResult {
    int processRet = -1;
    double inferenceMs = 0.0;
    int inputW = 0;
    int inputH = 0;
    int inferenceW = 0;
    int inferenceH = 0;
    uint32_t outputIndex = UINT32_MAX;
};

enum class PresentationCommandMode {
    RenderFrame,
    DisplayInterpolatedFrame
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
