#include "RifeRunner.h"

#if HAS_NCNN

#include "../../VulkanTest/rife_src/rife_ops.h"

#include <filesystem>
#include <iostream>
#include <utility>

namespace {

ncnn::Layer* CreateWarpLayer(void*) {
    return new Warp;
}

} // namespace

bool RifeRunner::initialize(ncnn::Net& targetNet,
                            const std::string& paramPath,
                            const std::string& binPath,
                            VkDevice rendererDevice,
                            int ncnnDeviceIndex) {
    if (!std::filesystem::exists(paramPath)) {
        std::cerr << "[RIFE] param file not found: " << paramPath << std::endl;
        return false;
    }
    if (!std::filesystem::exists(binPath)) {
        std::cerr << "[RIFE] bin file not found: " << binPath << std::endl;
        return false;
    }

    targetNet.clear();
    rifeEngine.reset();
    rendererVkdev.reset();
    setVulkanDeviceCalled = false;

    if (!targetNet.opt.use_vulkan_compute) {
        std::cerr << "[RIFE] GPU inference requested but ncnn Vulkan compute is unavailable" << std::endl;
        loadedParamPath.clear();
        loadedBinPath.clear();
        return false;
    }

    if (rendererDevice == VK_NULL_HANDLE || ncnnDeviceIndex < 0) {
        std::cerr << "[RIFE] renderer VkDevice is not available for ncnn interop" << std::endl;
        loadedParamPath.clear();
        loadedBinPath.clear();
        return false;
    }

    rendererVkdev = std::make_unique<ncnn::VulkanDevice>(ncnnDeviceIndex, rendererDevice, false);
    targetNet.set_vulkan_device(rendererVkdev.get());
    setVulkanDeviceCalled = true;

    if (!warpLayerRegistered) {
        if (targetNet.register_custom_layer("rife.Warp", CreateWarpLayer) != 0) {
            std::cerr << "[RIFE] failed to register custom layer: rife.Warp" << std::endl;
            loadedParamPath.clear();
            loadedBinPath.clear();
            return false;
        }
        warpLayerRegistered = true;
    }

    const int paramRet = targetNet.load_param(paramPath.c_str());
    if (paramRet != 0) {
        std::cerr << "[RIFE] failed to load param: " << paramPath << " (code=" << paramRet << ")" << std::endl;
        loadedParamPath.clear();
        loadedBinPath.clear();
        return false;
    }

    const int binRet = targetNet.load_model(binPath.c_str());
    if (binRet != 0) {
        std::cerr << "[RIFE] failed to load model: " << binPath << " (code=" << binRet << ")" << std::endl;
        loadedParamPath.clear();
        loadedBinPath.clear();
        return false;
    }

    auto engine = std::make_unique<RIFE>(rendererVkdev.get(), false, false, false, 1, false, true);
    const std::filesystem::path modelDir = std::filesystem::path(paramPath).parent_path();
#if _WIN32
    const int engineLoadRet = engine->load(modelDir.wstring());
#else
    const int engineLoadRet = engine->load(modelDir.string());
#endif
    if (engineLoadRet != 0) {
        std::cerr << "[RIFE] failed to initialize Vulkan RIFE engine from: "
                  << modelDir.string() << " (code=" << engineLoadRet << ")" << std::endl;
        loadedParamPath.clear();
        loadedBinPath.clear();
        return false;
    }

    rifeEngine = std::move(engine);
    loadedParamPath = paramPath;
    loadedBinPath = binPath;
    std::cout << "[RIFE] model loaded for shared renderer VkDevice GPU path: "
              << loadedParamPath << " + " << loadedBinPath << std::endl;
    return true;
}

bool RifeRunner::isReady() const {
    return rifeEngine && !loadedParamPath.empty() && !loadedBinPath.empty();
}

int RifeRunner::processBgrFrames(const std::vector<uint8_t>& prevBgr,
                                 const std::vector<uint8_t>& currBgr,
                                 int width,
                                 int height,
                                 float timestep,
                                 std::vector<uint8_t>& outBgr) const {
    if (!isReady()) {
        return -1;
    }

    const size_t expectedSize = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    if (prevBgr.size() != expectedSize || currBgr.size() != expectedSize) {
        return -2;
    }

    outBgr.assign(expectedSize, 0);
    ncnn::Mat prev(width, height, const_cast<uint8_t*>(prevBgr.data()), (size_t)3u, 1);
    ncnn::Mat curr(width, height, const_cast<uint8_t*>(currBgr.data()), (size_t)3u, 1);
    ncnn::Mat out(width, height, outBgr.data(), (size_t)3u, 1);
    return rifeEngine->process(prev, curr, timestep, out);
}

int RifeRunner::processGpuRgbaFrames(VkBuffer prevBuffer,
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
                                     float timestep) const {
    if (!isReady()) {
        return -1;
    }

    const VkDeviceSize expectedSize =
        static_cast<VkDeviceSize>(width) *
        static_cast<VkDeviceSize>(height) * 4;
    if (prevBuffer == VK_NULL_HANDLE || currBuffer == VK_NULL_HANDLE || outBuffer == VK_NULL_HANDLE ||
        prevSize < expectedSize || currSize < expectedSize || outSize < expectedSize) {
        return -2;
    }

    auto wrapExternalBuffer = [](VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize size) {
        ncnn::VkBufferMemory wrapped{};
        wrapped.buffer = buffer;
        wrapped.offset = 0;
        wrapped.capacity = static_cast<size_t>(size);
        wrapped.memory = memory;
        wrapped.mapped_ptr = nullptr;
        wrapped.access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        wrapped.stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
        wrapped.refcount = 0;
        return wrapped;
    };

    ncnn::VkBufferMemory prevWrapped = wrapExternalBuffer(prevBuffer, prevMemory, prevSize);
    ncnn::VkBufferMemory currWrapped = wrapExternalBuffer(currBuffer, currMemory, currSize);
    ncnn::VkBufferMemory outWrapped = wrapExternalBuffer(outBuffer, outMemory, outSize);
    outWrapped.access_flags = 0;
    outWrapped.stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    ncnn::VkMat prev(width, height, &prevWrapped, (size_t)4u, 1, nullptr);
    ncnn::VkMat curr(width, height, &currWrapped, (size_t)4u, 1, nullptr);
    ncnn::VkMat out(width, height, &outWrapped, (size_t)4u, 1, nullptr);

    return rifeEngine->process_v4_gpu(prev, curr, width, height, inferenceWidth, inferenceHeight, timestep, out);
}

bool RifeRunner::warmupInference(int width, int height, int runs) const {
    if (!isReady() || width <= 0 || height <= 0 || runs <= 0) {
        return false;
    }

    const size_t expectedSize = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    std::vector<uint8_t> prevBgr(expectedSize, 0);
    std::vector<uint8_t> currBgr(expectedSize, 0);
    std::vector<uint8_t> outBgr;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
            prevBgr[idx + 0] = static_cast<uint8_t>((x * 3 + y * 5) & 255);
            prevBgr[idx + 1] = static_cast<uint8_t>((x * 7 + y * 11) & 255);
            prevBgr[idx + 2] = static_cast<uint8_t>((x * 13 + y * 17) & 255);
            currBgr[idx + 0] = static_cast<uint8_t>((x * 19 + y * 23 + 31) & 255);
            currBgr[idx + 1] = static_cast<uint8_t>((x * 29 + y * 37 + 47) & 255);
            currBgr[idx + 2] = static_cast<uint8_t>((x * 41 + y * 43 + 59) & 255);
        }
    }

    for (int i = 0; i < runs; ++i) {
        const int ret = processBgrFrames(prevBgr, currBgr, width, height, 0.5f, outBgr);
        if (ret != 0 || outBgr.size() != expectedSize) {
            std::cerr << "[RIFE] warmup inference failed (code=" << ret << ")" << std::endl;
            return false;
        }

        for (size_t j = 0; j < currBgr.size(); ++j) {
            currBgr[j] = static_cast<uint8_t>(currBgr[j] + 1);
        }
    }

    std::cout << "[RIFE] warmup inference completed (runs=" << runs
              << ", input=" << width << "x" << height << ")" << std::endl;
    return true;
}

bool RifeRunner::wasVulkanDeviceSet() const {
    return setVulkanDeviceCalled;
}

void RifeRunner::reset() {
    rifeEngine.reset();
    rendererVkdev.reset();
    loadedParamPath.clear();
    loadedBinPath.clear();
}

#endif
