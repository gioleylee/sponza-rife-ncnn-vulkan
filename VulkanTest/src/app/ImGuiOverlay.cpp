// Owns the Dear ImGui overlay lifecycle and swapchain compositing pass.
#include "VulkanNcnnRenderer.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <stdexcept>

namespace {

void checkImGuiVkResult(VkResult result) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Dear ImGui Vulkan backend call failed");
    }
}

}

void VulkanNcnnRenderer::initializeImGuiResources() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    createImGuiRenderPass();
    createImGuiFramebuffers();

    if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        throw std::runtime_error("failed to initialize Dear ImGui GLFW backend");
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = physicalDevice;
    initInfo.Device = device;
    initInfo.QueueFamily = graphicsQueueFamilyIndex;
    initInfo.Queue = graphicsQueue;
    initInfo.DescriptorPoolSize = 64;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = static_cast<uint32_t>(swapChainImages.size());
    initInfo.PipelineInfoMain.RenderPass = imguiRenderPass;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CheckVkResultFn = checkImGuiVkResult;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("failed to initialize Dear ImGui Vulkan backend");
    }

    imguiInitialized = true;

    // Prime the backend-owned font texture before NCNN async work can contend
    // for the graphics queue. Later frames only record draw commands.
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Render();
    vkDeviceWaitIdle(device);
}

void VulkanNcnnRenderer::createImGuiRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &imguiRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create Dear ImGui render pass");
    }
}

void VulkanNcnnRenderer::createImGuiFramebuffers() {
    if (imguiRenderPass == VK_NULL_HANDLE || swapChainImageViews.empty()) {
        return;
    }

    imguiFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
        VkImageView attachments[] = { swapChainImageViews[i] };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = imguiRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &imguiFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create Dear ImGui framebuffer");
        }
    }
}

void VulkanNcnnRenderer::cleanupImGuiFramebuffers() {
    for (auto framebuffer : imguiFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    imguiFramebuffers.clear();
}

void VulkanNcnnRenderer::cleanupImGui() {
    if (imguiInitialized) {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imguiInitialized = false;
    }

    cleanupImGuiFramebuffers();

    if (imguiRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, imguiRenderPass, nullptr);
        imguiRenderPass = VK_NULL_HANDLE;
    }
}

void VulkanNcnnRenderer::beginImGuiFrame() {
    if (!imguiInitialized) {
        return;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (showFpsCounter) {
        ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.35f);
        ImGui::Begin("Presented FPS",
            nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav);
        ImGui::Text("Presented: %.1f FPS", displayedPresentedFps);
        ImGui::Text("Real: %.1f FPS", displayedRealFps);
        ImGui::Text("Interpolated: %.1f FPS", displayedInterpolatedFps);
        ImGui::End();
    }

    if (imguiVisible) {
        ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Renderer Features", &imguiVisible, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::TextUnformatted("Overlay");
        ImGui::Checkbox("FPS counter", &showFpsCounter);
        ImGui::Separator();

        ImGui::TextUnformatted("Debug Views");
        ImGui::Checkbox("Normal buffer (N)", &showNormals);
        ImGui::Checkbox("Albedo buffer (B)", &showAlbedo);
        ImGui::Checkbox("Position buffer (V)", &showPosition);
        ImGui::Checkbox("Specular debug (M)", &showSpecular);
        ImGui::Checkbox("Interpolation debug panel", &showInterpolationDebugPanel);

        ImGui::Separator();
        ImGui::TextUnformatted("Frame Interpolation");
#if HAS_NCNN
        bool realtimeInterpolation = ncnnPresentationState.ncnnRealtimeInterpolationEnabled;
        if (ImGui::Checkbox("Realtime RIFE interpolation (R)", &realtimeInterpolation)) {
            setNcnnRealtimeInterpolationEnabled(realtimeInterpolation);
        }
        if (!ncnnModelAttachedToRenderer) {
            ImGui::TextDisabled("RIFE model is not attached to the Vulkan renderer");
        }
#else
        ImGui::TextDisabled("NCNN/RIFE support is not compiled in");
#endif
        ImGui::Checkbox("Green interpolated-frame marker (Y)", &markInterpolatedFrames);
        bool benchmarkMode = benchmarkModeEnabled;
        if (ImGui::Checkbox("Benchmark 30 FPS source lock (U)", &benchmarkMode)) {
            setBenchmarkModeEnabled(benchmarkMode);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Camera");
        ImGui::Checkbox("Auto-pan camera (T)", &autoPanEnabled);
        ImGui::SliderFloat("Auto-pan speed", &autoPanSpeedDegreesPerSecond, 0.25f, 120.0f, "%.2f deg/s");

        ImGui::End();
    }
    ImGui::Render();
}

void VulkanNcnnRenderer::renderImGuiOverlay(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (!imguiInitialized ||
        imageIndex >= imguiFramebuffers.size()) {
        return;
    }

    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->CmdListsCount == 0) {
        return;
    }

    VkImageMemoryBarrier toColorAttachment{};
    toColorAttachment.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toColorAttachment.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toColorAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColorAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColorAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColorAttachment.image = swapChainImages[imageIndex];
    toColorAttachment.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toColorAttachment.subresourceRange.levelCount = 1;
    toColorAttachment.subresourceRange.layerCount = 1;
    toColorAttachment.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toColorAttachment);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = imguiRenderPass;
    renderPassInfo.framebuffer = imguiFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
    vkCmdEndRenderPass(commandBuffer);
}
