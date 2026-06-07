# main.cpp Refactor Plan

Goal: split `VulkanTest/src/main.cpp` into smaller implementation files without changing behavior. Keep `VulkanRifeRendererApp` as the owning class and move only function definitions plus their required includes.

## Proposed Target Files

### `VulkanTest/src/main.cpp`
Keep only process entry and startup error handling:
- `main`

Include only what `main` needs after extraction, likely `cstdlib`, `exception`, `iostream`, and `VulkanRifeRendererApp.h`.

### `VulkanTest/src/app/VulkanRifeRendererApp.cpp`
Application lifecycle and high-level orchestration:
- `VulkanRifeRendererApp::run`
- `VulkanRifeRendererApp::initVulkan`
- `VulkanRifeRendererApp::mainLoop`
- `VulkanRifeRendererApp::cleanup`

This file should coordinate existing subsystem functions but avoid owning low-level Vulkan object creation.

### `VulkanTest/src/rife/RifeIntegration.cpp`
NCNN/RIFE model setup and shutdown:
- `VulkanRifeRendererApp::findNcnnDeviceIndexForRenderer`
- `VulkanRifeRendererApp::applyNcnnVulkanOptions`
- `VulkanRifeRendererApp::initNcnn`
- `VulkanRifeRendererApp::shutdownNcnn`
- `VulkanRifeRendererApp::loadNcnnModel`
- `VulkanRifeRendererApp::tryLoadDefaultNcnnModel`

Keep `#if HAS_NCNN` guards with these definitions. This separates model/device selection from frame processing.

### `VulkanTest/src/app/VulkanDescriptors.cpp`
Descriptor pool and descriptor set allocation/update:
- `VulkanRifeRendererApp::createDescriptorPool`
- `VulkanRifeRendererApp::createDescriptorSets`
- `VulkanRifeRendererApp::createLightingDescriptorPool`
- `VulkanRifeRendererApp::createLightingDescriptorSets`

Extract after textures and fallback texture behavior are stable, because descriptor updates depend on material image views and samplers.

### `VulkanTest/src/app/FrameProcessing.cpp`
Offscreen frame capture, RIFE presentation buffers, async inference coordination, and per-frame draw flow:
- `VulkanRifeRendererApp::createFrameProcessingResources`
- `VulkanRifeRendererApp::cleanupFrameProcessingResources`
- `VulkanRifeRendererApp::findAvailableOffscreenFrameSlot`
- `VulkanRifeRendererApp::copyOffscreenImageToSwapchain`
- `VulkanRifeRendererApp::copyRifeBufferToSwapchain`
- `VulkanRifeRendererApp::displayRifeFrameOnSwapchain`
- `VulkanRifeRendererApp::displayRifeSourceBufferOnSwapchain`
- `VulkanRifeRendererApp::displayCapturedRifeSourceOnSwapchain`
- `VulkanRifeRendererApp::processCapturedFrameForSlot`
- `VulkanRifeRendererApp::waitForAsyncRifeInference`
- `VulkanRifeRendererApp::pollAsyncRifeInference`
- `VulkanRifeRendererApp::submitAsyncRifeInferenceIfReady`
- `VulkanRifeRendererApp::drawFrame`

This is the riskiest split because it crosses rendering synchronization and RIFE state. Move as one patch only after simpler splits build cleanly.

### `VulkanTest/src/app/VulkanCommands.cpp`
Command pool, command buffer allocation/recording, sync objects, and one-shot command helpers:
- `VulkanRifeRendererApp::createCommandPool`
- `VulkanRifeRendererApp::createCommandBuffers`
- `VulkanRifeRendererApp::recordCommandBuffer`
- `VulkanRifeRendererApp::createSyncObjects`
- `VulkanRifeRendererApp::beginSingleTimeCommands`
- `VulkanRifeRendererApp::endSingleTimeCommands`

Move this before `FrameProcessing.cpp` if needed, because frame processing and texture upload both use command helpers.

### `VulkanTest/src/app/SceneResources.cpp`
Scene loading, generated cube geometry, texture upload, and image helpers:
- `VulkanRifeRendererApp::loadModel`
- `VulkanRifeRendererApp::appendRotatingCubeGeometry`
- `VulkanRifeRendererApp::createImage`
- `VulkanRifeRendererApp::transitionImageLayout`
- `VulkanRifeRendererApp::copyBufferToImage`
- `VulkanRifeRendererApp::loadMaterialTextures`
- `VulkanRifeRendererApp::generateMipmaps`
- `VulkanRifeRendererApp::createFallbackTexture`

This file will own `STB_IMAGE_IMPLEMENTATION`; remove it from `main.cpp` when this extraction happens.

### `VulkanTest/src/app/UniformUpdates.cpp`
Per-frame CPU-side uniform updates:
- `VulkanRifeRendererApp::updateUniformBuffer`

This can be extracted alone, but it is also reasonable to keep with draw code if minimizing files is preferred.

## Safest Extraction Order

1. Create `VulkanRifeRendererApp.cpp` and move only `run`, `initVulkan`, `mainLoop`, and `cleanup`; build.
2. Move NCNN/RIFE setup functions to `src/rife/RifeIntegration.cpp`; build with `HAS_NCNN` enabled and disabled if practical.
3. Move descriptor functions to `VulkanDescriptors.cpp`; build.
4. Move scene and texture resource functions to `SceneResources.cpp`, including the single `STB_IMAGE_IMPLEMENTATION` definition; build.
5. Move command helpers and command recording to `VulkanCommands.cpp`; build.
6. Move `updateUniformBuffer` to `UniformUpdates.cpp`; build.
7. Move frame processing and `drawFrame` to `FrameProcessing.cpp`; build and run with Vulkan validation layers.
8. After all splits are stable, reduce `main.cpp` includes to the minimum needed by `main`.

## Safety Notes

- Do one target file per patch where possible.
- Do not combine frame processing/RIFE extraction with unrelated rendering cleanup.
- Keep project metadata updates (`.vcxproj` and `.filters`) in the same patch as each new `.cpp` file.
- After every extraction, build `Debug|x64` and scan for Vulkan validation messages if any Vulkan resource lifetime, synchronization, or layout code moved.
