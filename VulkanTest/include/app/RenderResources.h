void createRenderPass();

void createDepthResources();

void createGBufferAttachments();

void createFramebuffers();

void createGraphicsPipeline();

void createLightingPipeline();

void cleanupFramebuffers();

void cleanupDepthResources();

void cleanupGBufferAttachments();

void cleanupRenderPipelines();

VkShaderModule createShaderModule(const std::vector<char>& code);

static std::vector<char> readFile(const std::string& filename);
