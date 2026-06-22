#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "AppTypes.h"

struct VulkanPresentJob {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    uint32_t imageIndex = 0;
    uint64_t swapchainGeneration = 0;
    int64_t timelineStep = -1;
    PresentedFrameKind frameKind = PresentedFrameKind::None;
    std::string label;
    std::chrono::steady_clock::time_point readyTime{};
};

struct VulkanPresentCompletion {
    VkResult result = VK_SUCCESS;
    uint64_t swapchainGeneration = 0;
    int64_t timelineStep = -1;
    PresentedFrameKind frameKind = PresentedFrameKind::None;
    std::string label;
};

class VulkanFramePacer {
public:
    using Clock = std::chrono::steady_clock;
    using PresentCallback = std::function<VkResult(const VulkanPresentJob&)>;

    VulkanFramePacer() = default;
    ~VulkanFramePacer();

    VulkanFramePacer(const VulkanFramePacer&) = delete;
    VulkanFramePacer& operator=(const VulkanFramePacer&) = delete;

    void start(PresentCallback callback);
    void stop();

    void setInterpolationEnabled(bool enabled);
    void enqueue(VulkanPresentJob job);
    bool tryPopCompletion(VulkanPresentCompletion& completion);

    // Swapchain recreation invalidates every queued VkSwapchainKHR and semaphore.
    // pause() cancels jobs that have not reached vkQueuePresentKHR and waits for
    // an active present call to return before the renderer destroys those objects.
    void pause();
    void resume(uint64_t swapchainGeneration);

private:
    struct TimedPresentJob {
        VulkanPresentJob job;
        Clock::time_point targetTime{};
    };

    static constexpr size_t NativeFrameWindow = 12;

    void interpolationLoop();
    void processInterpolationJob(VulkanPresentJob job);
    void presentLoop();
    void queueTimedJob(VulkanPresentJob job, Clock::time_point targetTime);
    void queueNativeOnly(VulkanPresentJob job, bool repeatedNative);
    void queueGeneratedAndNative(VulkanPresentJob generated, VulkanPresentJob native);
    void updateNativeFrameAverage(const VulkanPresentJob& native);
    Clock::duration nativeFrameInterval() const;
    Clock::duration halfFrameInterval() const;
    Clock::time_point nextTarget(Clock::duration interval);
    void clearQueuedJobsLocked();

    PresentCallback presentCallback;
    std::thread interpolationThread;
    std::thread presentThread;

    std::mutex interpolationMutex;
    std::condition_variable interpolationCondition;
    std::condition_variable interpolationIdleCondition;
    std::deque<VulkanPresentJob> interpolationQueue;
    bool interpolationWorkActive = false;

    std::mutex presentMutex;
    std::condition_variable presentCondition;
    std::condition_variable presentIdleCondition;
    std::deque<TimedPresentJob> presentQueue;
    bool presentCallActive = false;

    std::mutex completionMutex;
    std::deque<VulkanPresentCompletion> completions;

    std::atomic<bool> running = false;
    std::atomic<bool> paused = false;
    std::atomic<bool> interpolationEnabled = false;
    std::atomic<bool> resetInterpolationTimeline = false;
    std::atomic<uint64_t> activeSwapchainGeneration = 0;

    std::optional<VulkanPresentJob> pendingGenerated;
    std::deque<Clock::duration> nativeFrameSamples;
    Clock::time_point previousNativeReadyTime{};
    int64_t previousNativeTimelineStep = -1;
    Clock::time_point lastScheduledPresentTime{};
};
