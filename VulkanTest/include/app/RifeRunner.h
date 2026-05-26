#pragma once

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

#if !defined(HAS_NCNN)
#if __has_include(<ncnn/net.h>) && __has_include(<ncnn/gpu.h>) && __has_include(<ncnn/layer.h>)
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/allocator.h>
#include <ncnn/layer.h>
#define HAS_NCNN 1
#else
#define HAS_NCNN 0
#endif
#endif

#if HAS_NCNN
#include "rife.h"

class RifeRunner {
public:
    bool initialize(ncnn::Net& targetNet,
                    const std::string& paramPath,
                    const std::string& binPath,
                    VkDevice rendererDevice,
                    int ncnnDeviceIndex);

    bool isReady() const;

    int processGpuRgbaFrames(VkBuffer prevBuffer,
                             VkDeviceMemory prevMemory,
                             VkDeviceSize prevSize,
                             VkBuffer currBuffer,
                             VkDeviceMemory currMemory,
                             VkDeviceSize currSize,
                             VkBuffer outBuffer,
                             VkDeviceMemory outMemory,
                             VkDeviceSize outSize,
                             int width,
                             int height,
                             int inferenceWidth,
                             int inferenceHeight,
                             float timestep) const;

    bool wasVulkanDeviceSet() const;
    void reset();

private:
    std::unique_ptr<RIFE> rifeEngine;
    std::unique_ptr<ncnn::VulkanDevice> rendererVkdev;
    std::string loadedParamPath;
    std::string loadedBinPath;
    bool warpLayerRegistered = false;
    bool setVulkanDeviceCalled = false;
};
#endif
