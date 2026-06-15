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

void initializeFrameProcessingImageLayouts();

void cleanupFrameProcessingResources();

uint32_t findAvailableOffscreenFrameSlot() const;

void copyNcnnBufferToSwapchain(VkCommandBuffer commandBuffer,
                               uint32_t imageIndex,
                               VkBuffer sourceBuffer,
                               VkAccessFlags sourceAccessMask,
                               VkPipelineStageFlags sourceStageMask);

void displayNcnnFrameOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex);

void displayNcnnSourceBufferOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex, uint32_t sourceIndex);

void displayCapturedNcnnSourceOnSwapchain(VkCommandBuffer commandBuffer, uint32_t imageIndex);

#if HAS_NCNN
void waitForAsyncNcnnInference();

void setNcnnRealtimeInterpolationEnabled(bool enabled);

void pollAsyncNcnnInference();

bool submitAsyncNcnnInferenceIfReady();
#endif
