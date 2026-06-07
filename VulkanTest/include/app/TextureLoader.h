VkImageView createImageView(VkImage image, VkFormat format, uint32_t mipLevels);

void loadMaterialTextures();

void generateMipmaps(VkImage image, VkFormat imageFormat,
    int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

void createFallbackTexture();
