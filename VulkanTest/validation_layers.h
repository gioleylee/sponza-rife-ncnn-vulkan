#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace validation {

extern const std::vector<const char*> Layers;

#ifdef NDEBUG
constexpr bool Enabled = false;
#else
constexpr bool Enabled = true;
#endif

bool checkLayerSupport();
std::vector<const char*> getRequiredInstanceExtensions();
void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

VkResult createDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* createInfo,
    const VkAllocationCallbacks* allocator,
    VkDebugUtilsMessengerEXT* debugMessenger);

void destroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* allocator);

} // namespace validation
