#pragma once

#if defined(ENABLE_TRACY)
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <common/TracySystem.hpp>
using ProfileGpuContext = TracyVkCtx;

#define PROFILE_ZONE(name) ZoneScopedN(name)
#define PROFILE_THREAD(name) tracy::SetThreadName(name)
#define PROFILE_FRAME_MARK() FrameMark
#define PROFILE_PLOT(name, ...) TracyPlot(name, (__VA_ARGS__))
#define PROFILE_MESSAGE(text) TracyMessage((text).data(), (text).size())
#define PROFILE_GPU_CONTEXT(physicalDevice, device, queue, commandBuffer) \
    TracyVkContext(physicalDevice, device, queue, commandBuffer)
#define PROFILE_GPU_CONTEXT_NAME(context, name) TracyVkContextName(context, name, sizeof(name) - 1)
#define PROFILE_GPU_DESTROY(context) TracyVkDestroy(context)
#define PROFILE_GPU_ZONE(context, commandBuffer, name) TracyVkZone(context, commandBuffer, name)
#define PROFILE_GPU_COLLECT(context, commandBuffer) TracyVkCollect(context, commandBuffer)
#else
using ProfileGpuContext = void*;
#define PROFILE_ZONE(name) ((void)0)
#define PROFILE_THREAD(name) ((void)0)
#define PROFILE_FRAME_MARK() ((void)0)
#define PROFILE_PLOT(name, ...) ((void)0)
#define PROFILE_MESSAGE(text) ((void)0)
#define PROFILE_GPU_CONTEXT(physicalDevice, device, queue, commandBuffer) nullptr
#define PROFILE_GPU_CONTEXT_NAME(context, name) ((void)0)
#define PROFILE_GPU_DESTROY(context) ((void)0)
#define PROFILE_GPU_ZONE(context, commandBuffer, name) ((void)0)
#define PROFILE_GPU_COLLECT(context, commandBuffer) ((void)0)
#endif
