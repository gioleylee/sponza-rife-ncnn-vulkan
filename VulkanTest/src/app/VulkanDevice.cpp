// Owns Vulkan instance/device setup and queue-family selection.
#include "VulkanNcnnRenderer.h"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

void VulkanNcnnRenderer::createInstance() {
    if (validation::Enabled && !validation::checkLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanNcnnRenderer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = validation::getRequiredInstanceExtensions();
    uint32_t instanceExtensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> availableInstanceExtensions(instanceExtensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, availableInstanceExtensions.data());

    const bool debugUtilsAvailable = std::any_of(
        availableInstanceExtensions.begin(),
        availableInstanceExtensions.end(),
        [](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        });
    const bool debugUtilsRequested = std::any_of(
        extensions.begin(),
        extensions.end(),
        [](const char* extension) {
            return std::strcmp(extension, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        });
    if (debugUtilsAvailable && !debugUtilsRequested) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (validation::Enabled) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validation::Layers.size());
        createInfo.ppEnabledLayerNames = validation::Layers.data();

        validation::populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void VulkanNcnnRenderer::setupDebugMessenger() {
    if (!validation::Enabled) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    validation::populateDebugMessengerCreateInfo(createInfo);

    if (validation::createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void VulkanNcnnRenderer::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void VulkanNcnnRenderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice firstSuitableDevice = VK_NULL_HANDLE;
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        const bool suitable = isDeviceSuitable(device);

        if (!suitable) {
            continue;
        }

        if (firstSuitableDevice == VK_NULL_HANDLE) {
            firstSuitableDevice = device;
        }

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        physicalDevice = firstSuitableDevice;
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    VkPhysicalDeviceProperties selectedProperties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &selectedProperties);
}

void VulkanNcnnRenderer::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<std::vector<float>> queuePriorities(queueFamilyProperties.size());
    for (uint32_t queueFamily = 0; queueFamily < queueFamilyProperties.size(); ++queueFamily) {
        const uint32_t queueCount = queueFamilyProperties[queueFamily].queueCount;
        if (queueCount == 0) {
            continue;
        }

        queuePriorities[queueFamily].assign(queueCount, 1.0f);

        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = queueCount;
        queueCreateInfo.pQueuePriorities = queuePriorities[queueFamily].data();
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
    if (!deviceFeatures.samplerAnisotropy) {
        throw std::runtime_error("sampler anisotropy not supported");
    }

    deviceFeatures.samplerAnisotropy = VK_TRUE;

    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

    auto hasDeviceExtension = [&](const char* name) {
        return std::any_of(availableExtensions.begin(), availableExtensions.end(), [&](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
    };

    std::vector<const char*> enabledDeviceExtensions = deviceExtensions;
    auto enableOptionalDeviceExtension = [&](const char* name) {
        const bool alreadyEnabled = std::any_of(enabledDeviceExtensions.begin(), enabledDeviceExtensions.end(), [&](const char* enabled) {
            return std::strcmp(enabled, name) == 0;
        });
        if (!alreadyEnabled && hasDeviceExtension(name)) {
            enabledDeviceExtensions.push_back(name);
        }
    };

#if HAS_NCNN
    enableOptionalDeviceExtension("VK_KHR_8bit_storage");
    enableOptionalDeviceExtension("VK_KHR_16bit_storage");
    enableOptionalDeviceExtension("VK_KHR_bind_memory2");
    enableOptionalDeviceExtension("VK_KHR_dedicated_allocation");
    enableOptionalDeviceExtension("VK_KHR_descriptor_update_template");
    enableOptionalDeviceExtension("VK_KHR_get_memory_requirements2");
    enableOptionalDeviceExtension("VK_KHR_maintenance1");
    enableOptionalDeviceExtension("VK_KHR_maintenance3");
    enableOptionalDeviceExtension("VK_KHR_push_descriptor");
    enableOptionalDeviceExtension("VK_KHR_shader_float16_int8");
    enableOptionalDeviceExtension("VK_KHR_storage_buffer_storage_class");
    enableOptionalDeviceExtension("VK_EXT_descriptor_indexing");
#endif

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    VkPhysicalDeviceFeatures2 enabledFeatures2{};
    enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabledFeatures2.features = deviceFeatures;

    void* featureChain = nullptr;
#if HAS_NCNN
    VkPhysicalDevice8BitStorageFeatures enabled8BitStorageFeatures{};
    enabled8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDevice16BitStorageFeatures enabled16BitStorageFeatures{};
    enabled16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDeviceFloat16Int8FeaturesKHR enabledFloat16Int8Features{};
    enabledFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
    VkPhysicalDeviceVulkanMemoryModelFeatures enabledVulkanMemoryModelFeatures{};
    enabledVulkanMemoryModelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabledCooperativeMatrixFeatures{};
    enabledCooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceSubgroupSizeControlFeatures enabledSubgroupSizeControlFeatures{};
    enabledSubgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProperties{};
    subgroupSizeControlProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;

    VkPhysicalDeviceFeatures2 queriedFeatures2{};
    queriedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    queriedFeatures2.pNext = &enabled8BitStorageFeatures;
    enabled8BitStorageFeatures.pNext = &enabled16BitStorageFeatures;
    enabled16BitStorageFeatures.pNext = &enabledFloat16Int8Features;
    enabledFloat16Int8Features.pNext = &enabledVulkanMemoryModelFeatures;
    enabledVulkanMemoryModelFeatures.pNext = &enabledCooperativeMatrixFeatures;
    enabledCooperativeMatrixFeatures.pNext = &enabledSubgroupSizeControlFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &queriedFeatures2);

    VkPhysicalDeviceProperties2 queriedProperties2{};
    queriedProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    queriedProperties2.pNext = &subgroupSizeControlProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &queriedProperties2);

    const bool canEnableSubgroupSizeControl =
        hasDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) &&
        enabledSubgroupSizeControlFeatures.subgroupSizeControl == VK_TRUE &&
        (subgroupSizeControlProperties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;

    if (canEnableSubgroupSizeControl) {
        enableOptionalDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        enabledSubgroupSizeControlFeatures.computeFullSubgroups = VK_FALSE;
        enabledSubgroupSizeControlFeatures.pNext = featureChain;
        featureChain = &enabledSubgroupSizeControlFeatures;
    }
    else {
        enabledSubgroupSizeControlFeatures.subgroupSizeControl = VK_FALSE;
        enabledSubgroupSizeControlFeatures.computeFullSubgroups = VK_FALSE;
    }

    if (enabledVulkanMemoryModelFeatures.vulkanMemoryModel == VK_TRUE) {
        enableOptionalDeviceExtension(VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME);
        enabledVulkanMemoryModelFeatures.vulkanMemoryModelDeviceScope =
            enabledVulkanMemoryModelFeatures.vulkanMemoryModelDeviceScope == VK_TRUE ? VK_TRUE : VK_FALSE;
        enabledVulkanMemoryModelFeatures.vulkanMemoryModelAvailabilityVisibilityChains =
            enabledVulkanMemoryModelFeatures.vulkanMemoryModelAvailabilityVisibilityChains == VK_TRUE ? VK_TRUE : VK_FALSE;
        enabledVulkanMemoryModelFeatures.pNext = featureChain;
        featureChain = &enabledVulkanMemoryModelFeatures;
    }
    else {
        std::cout << "[Vulkan] selected GPU does not support vulkanMemoryModel; "
                  << "NCNN shaders requiring it may fail validation" << std::endl;
    }

    if (enabledCooperativeMatrixFeatures.cooperativeMatrix == VK_TRUE) {
        enableOptionalDeviceExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        enabledCooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess =
            enabledCooperativeMatrixFeatures.cooperativeMatrixRobustBufferAccess == VK_TRUE ? VK_TRUE : VK_FALSE;
        enabledCooperativeMatrixFeatures.pNext = featureChain;
        featureChain = &enabledCooperativeMatrixFeatures;
    }
    else {
        std::cout << "[Vulkan] selected GPU does not support cooperativeMatrix; "
                  << "NCNN shaders requiring it may fail validation" << std::endl;
    }

    enabledFloat16Int8Features.pNext = featureChain;
    featureChain = &enabledFloat16Int8Features;
    enabled16BitStorageFeatures.pNext = featureChain;
    featureChain = &enabled16BitStorageFeatures;
    enabled8BitStorageFeatures.pNext = featureChain;
    featureChain = &enabled8BitStorageFeatures;
#endif
    enabledFeatures2.pNext = featureChain;
    createInfo.pNext = &enabledFeatures2;
    createInfo.pEnabledFeatures = nullptr;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();

    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
    loadDebugUtilsFunctions();

    graphicsQueueFamilyIndex = indices.graphicsFamily.value();
    presentQueueFamilyIndex = indices.presentFamily.value();
    computeQueueFamilyIndex = indices.computeFamily.value();

    computeQueueIndex = 0;

    const auto preferNonComputeQueueIndex = [&](uint32_t queueFamilyIndex) {
        if (queueFamilyIndex == computeQueueFamilyIndex &&
            queueFamilyProperties[queueFamilyIndex].queueCount > 1) {
            return 1u;
        }

        return 0u;
    };

    graphicsQueueIndex = preferNonComputeQueueIndex(graphicsQueueFamilyIndex);
    presentQueueIndex = preferNonComputeQueueIndex(presentQueueFamilyIndex);

    vkGetDeviceQueue(device, graphicsQueueFamilyIndex, graphicsQueueIndex, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueFamilyIndex, presentQueueIndex, &presentQueue);
    vkGetDeviceQueue(device, computeQueueFamilyIndex, computeQueueIndex, &computeQueue);
    setDebugObjectName(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<uint64_t>(graphicsQueue), "Graphics Queue");
    setDebugObjectName(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<uint64_t>(presentQueue), "Present Queue");
    setDebugObjectName(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<uint64_t>(computeQueue), "Compute Queue");

    if (computeQueue == graphicsQueue || computeQueue == presentQueue) {
        std::cout << "[Vulkan] warning: compute queue aliases a renderer queue; "
                  << "this device exposes no distinct queue handle for the selected families" << std::endl;
    }
    else {
        std::cout << "[Vulkan] using distinct compute queue handle"
                  << " (family=" << computeQueueFamilyIndex
                  << ", index=" << computeQueueIndex << ")" << std::endl;
    }

#if defined(_WIN32)
    vkGetMemoryWin32HandleKHRFn =
        reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (!vkGetMemoryWin32HandleKHRFn) {
        throw std::runtime_error("VK_KHR_external_memory_win32 was enabled but vkGetMemoryWin32HandleKHR is unavailable");
    }
#endif
}

bool VulkanNcnnRenderer::isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool VulkanNcnnRenderer::checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices VulkanNcnnRenderer::findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        if ((queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) && !indices.computeFamily.has_value()) {
            indices.computeFamily = i;
        }
        if ((queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.computeFamily = i;
        }

        if ((queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) && !indices.transferFamily.has_value()) {
            indices.transferFamily = i;
        }
        if ((queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.transferFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        i++;
    }

    if (!indices.transferFamily.has_value()) {
        indices.transferFamily = indices.computeFamily;
    }

    return indices;
}

VkFormat VulkanNcnnRenderer::findDepthFormat() {
    return VK_FORMAT_D32_SFLOAT;
}
