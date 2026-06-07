void cleanupSwapChain();

void recreateSwapChain();

void createSwapChain();

void createImageViews();

void createFramebuffers();

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
