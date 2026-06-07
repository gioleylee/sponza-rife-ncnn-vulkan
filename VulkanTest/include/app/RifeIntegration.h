#if HAS_NCNN
int findNcnnDeviceIndexForRenderer() const;

void initNcnn();

void shutdownNcnn();

bool loadNcnnModel(const std::string& paramPath, const std::string& binPath);

void tryLoadDefaultNcnnModel();

void applyNcnnVulkanOptions();
#endif

#if defined(_WIN32)
bool createExportableFrameBuffer(VkDeviceSize size,
                                 VkBuffer& buffer,
                                 VkDeviceMemory& bufferMemory,
                                 HANDLE& externalHandle);
#endif

void createFrameProcessingResources();

void cleanupFrameProcessingResources();

uint32_t findAvailableOffscreenFrameSlot() const;

void copyRifeBufferToSwapchain(VkCommandBuffer commandBuffer,
                               uint32_t imageIndex,
                               VkBuffer sourceBuffer,
                               VkAccessFlags sourceAccessMask,
                               VkPipelineStageFlags sourceStageMask);

void displayRifeFrameOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex);

void displayRifeSourceBufferOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t sourceIndex);

void displayCapturedRifeSourceOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex);

#if HAS_NCNN
void waitForAsyncRifeInference();

void pollAsyncRifeInference();

bool submitAsyncRifeInferenceIfReady();
#endif
