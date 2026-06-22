#include "VulkanFramePacer.h"
#include "CpuProfiler.h"

#include <algorithm>
#include <utility>

namespace {

using Clock = VulkanFramePacer::Clock;

constexpr auto DefaultNativeFrameInterval = std::chrono::microseconds(33333);
constexpr auto MinimumNativeFrameInterval = std::chrono::milliseconds(8);
constexpr auto MaximumNativeFrameInterval = std::chrono::milliseconds(100);

bool isUnset(Clock::time_point timePoint) {
    return timePoint.time_since_epoch() == Clock::duration::zero();
}

}

VulkanFramePacer::~VulkanFramePacer() {
    stop();
}

void VulkanFramePacer::start(PresentCallback callback) {
    if (running.exchange(true)) {
        return;
    }

    presentCallback = std::move(callback);
    paused.store(false);
    interpolationThread = std::thread(&VulkanFramePacer::interpolationLoop, this);
    presentThread = std::thread(&VulkanFramePacer::presentLoop, this);
}

void VulkanFramePacer::stop() {
    if (!running.exchange(false)) {
        return;
    }

    paused.store(true);
    interpolationCondition.notify_all();
    presentCondition.notify_all();

    if (interpolationThread.joinable()) {
        interpolationThread.join();
    }
    if (presentThread.joinable()) {
        presentThread.join();
    }

    std::scoped_lock queueLock(interpolationMutex, presentMutex);
    clearQueuedJobsLocked();
    pendingGenerated.reset();
    presentCallback = {};
}

void VulkanFramePacer::setInterpolationEnabled(bool enabled) {
    const bool wasEnabled = interpolationEnabled.exchange(enabled);
    if (!wasEnabled && enabled) {
        resetInterpolationTimeline.store(true);
    }
    if (wasEnabled && !enabled && running.load() && !paused.load()) {
        VulkanPresentJob controlJob{};
        controlJob.swapchainGeneration = activeSwapchainGeneration.load();
        {
            std::lock_guard<std::mutex> lock(interpolationMutex);
            interpolationQueue.push_back(std::move(controlJob));
        }
        interpolationCondition.notify_one();
    }
}

void VulkanFramePacer::enqueue(VulkanPresentJob job) {
    if (!running.load() || paused.load()) {
        return;
    }

    job.readyTime = Clock::now();
    {
        std::lock_guard<std::mutex> lock(interpolationMutex);
        interpolationQueue.push_back(std::move(job));
        const auto nativeCount = std::count_if(
            interpolationQueue.begin(), interpolationQueue.end(), [](const VulkanPresentJob& queued) {
                return queued.frameKind == PresentedFrameKind::Real;
            });
        PROFILE_PLOT("Native Frame Queue Size", static_cast<int64_t>(nativeCount));
    }
    interpolationCondition.notify_one();
}

bool VulkanFramePacer::tryPopCompletion(VulkanPresentCompletion& completion) {
    std::lock_guard<std::mutex> lock(completionMutex);
    if (completions.empty()) {
        return false;
    }

    completion = std::move(completions.front());
    completions.pop_front();
    return true;
}

void VulkanFramePacer::pause() {
    if (!running.load()) {
        return;
    }

    paused.store(true);
    interpolationCondition.notify_all();
    presentCondition.notify_all();

    {
        std::unique_lock<std::mutex> interpolationLock(interpolationMutex);
        {
            PROFILE_ZONE("Wait Interpolation Thread Idle");
            interpolationQueue.clear();
            interpolationIdleCondition.wait(interpolationLock, [this]() {
                return !interpolationWorkActive;
            });
        }
        pendingGenerated.reset();
    }

    {
        std::lock_guard<std::mutex> presentLock(presentMutex);
        presentQueue.clear();
    }

    std::unique_lock<std::mutex> presentLock(presentMutex);
    {
        PROFILE_ZONE("Wait Present Thread Idle");
        presentIdleCondition.wait(presentLock, [this]() {
            return !presentCallActive;
        });
    }
}

void VulkanFramePacer::resume(uint64_t swapchainGeneration) {
    {
        std::scoped_lock queueLock(interpolationMutex, presentMutex);
        clearQueuedJobsLocked();
        pendingGenerated.reset();
        nativeFrameSamples.clear();
        previousNativeReadyTime = {};
        previousNativeTimelineStep = -1;
        lastScheduledPresentTime = {};
    }

    activeSwapchainGeneration.store(swapchainGeneration);
    paused.store(false);
    interpolationCondition.notify_all();
    presentCondition.notify_all();
}

void VulkanFramePacer::interpolationLoop() {
    PROFILE_THREAD("InterpolationThread");
    while (running.load()) {
        VulkanPresentJob job;
        {
            std::unique_lock<std::mutex> lock(interpolationMutex);
            {
                PROFILE_ZONE("Wait Native Frame Pair");
                interpolationCondition.wait(lock, [this]() {
                    return !running.load() || (!paused.load() && !interpolationQueue.empty());
                });
            }

            if (!running.load()) {
                break;
            }
            if (paused.load() || interpolationQueue.empty()) {
                continue;
            }

            job = std::move(interpolationQueue.front());
            interpolationQueue.pop_front();
            interpolationWorkActive = true;
        }

        processInterpolationJob(std::move(job));

        {
            std::lock_guard<std::mutex> lock(interpolationMutex);
            interpolationWorkActive = false;
        }
        interpolationIdleCondition.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(interpolationMutex);
        interpolationWorkActive = false;
    }
    interpolationIdleCondition.notify_all();
}

void VulkanFramePacer::processInterpolationJob(VulkanPresentJob job) {
    if (job.swapchainGeneration != activeSwapchainGeneration.load()) {
        return;
    }

    if (resetInterpolationTimeline.exchange(false)) {
        // The renderer resets its logical frame IDs when interpolation is
        // toggled. Keep the learned cadence but restart duplicate detection.
        previousNativeReadyTime = {};
        previousNativeTimelineStep = -1;
        lastScheduledPresentTime = {};
    }

    if (!interpolationEnabled.load()) {
        if (pendingGenerated) {
            queueTimedJob(std::move(*pendingGenerated), Clock::now());
            pendingGenerated.reset();
        }
        if (job.swapchain != VK_NULL_HANDLE) {
            if (job.frameKind == PresentedFrameKind::Real) {
                // Learn the renderer's native cadence before interpolation is
                // enabled. Once enabled, generated-frame latency must not feed
                // back into this average or the pacer will progressively slow
                // the renderer to the interpolation completion rate.
                updateNativeFrameAverage(job);
            }
            queueTimedJob(std::move(job), Clock::now());
        }
        return;
    }

    if (job.frameKind == PresentedFrameKind::Interpolated) {
        // The renderer has completed both source frames and the interpolation
        // output. Hold N+0.5 until native N+1 is prepared so the two jobs can
        // be assigned adjacent half-frame timestamps as one ordered pair.
        pendingGenerated = std::move(job);
        return;
    }

    if (job.frameKind != PresentedFrameKind::Real) {
        queueTimedJob(std::move(job), nextTarget(halfFrameInterval()));
        return;
    }

    const bool repeatedNative =
        previousNativeTimelineStep >= 0 && job.timelineStep <= previousNativeTimelineStep;
    if (pendingGenerated &&
        pendingGenerated->timelineStep >= 0 &&
        job.timelineStep == pendingGenerated->timelineStep + 1) {
        queueGeneratedAndNative(std::move(*pendingGenerated), std::move(job));
        pendingGenerated.reset();
    }
    else if (repeatedNative) {
        // Render-ahead may prepare another copy of the held N frame while N+0.5
        // is already waiting. Present that fallback copy without throwing away
        // N+0.5; the next unique native frame will complete the ordered pair.
        queueNativeOnly(std::move(job), true);
    }
    else if (pendingGenerated) {
        // A timeline discontinuity should not strand an acquired swapchain
        // image. Preserve generated-before-native order, then resynchronize.
        queueGeneratedAndNative(std::move(*pendingGenerated), std::move(job));
        pendingGenerated.reset();
    }
    else {
        // No usable interpolation output exists for this native frame.
        // Present the native frame at the next native cadence point.
        pendingGenerated.reset();
        queueNativeOnly(std::move(job), repeatedNative);
    }
}

void VulkanFramePacer::presentLoop() {
    PROFILE_THREAD("PresentThread");
    while (running.load()) {
        TimedPresentJob timedJob;
        double pacingWaitMs = 0.0;
        {
            std::unique_lock<std::mutex> lock(presentMutex);
            {
                PROFILE_ZONE("Wait Present Queue");
                presentCondition.wait(lock, [this]() {
                    return !running.load() || (!paused.load() && !presentQueue.empty());
                });
            }

            if (!running.load()) {
                break;
            }
            if (paused.load() || presentQueue.empty()) {
                continue;
            }

            const auto targetTime = presentQueue.front().targetTime;
            const auto pacingWaitStart = Clock::now();
            {
                PROFILE_ZONE("Pacing Wait");
                presentCondition.wait_until(lock, targetTime, [this, targetTime]() {
                    return !running.load() || paused.load() || presentQueue.empty() ||
                        presentQueue.front().targetTime != targetTime;
                });
            }
            pacingWaitMs = std::chrono::duration<double, std::milli>(
                Clock::now() - pacingWaitStart).count();
            PROFILE_PLOT("Pacing Wait (ms)", pacingWaitMs);

            if (!running.load()) {
                break;
            }
            if (paused.load() || presentQueue.empty() ||
                presentQueue.front().targetTime > Clock::now()) {
                continue;
            }

            {
                PROFILE_ZONE("Select Present Job");
                timedJob = std::move(presentQueue.front());
                presentQueue.pop_front();
            }
            PROFILE_PLOT("Present Queue Size", static_cast<int64_t>(presentQueue.size()));
            presentCallActive = true;
        }

        VkResult result = VK_ERROR_DEVICE_LOST;
        const auto presentStart = Clock::now();
        if (timedJob.job.swapchainGeneration == activeSwapchainGeneration.load() &&
            presentCallback) {
            result = presentCallback(timedJob.job);
        }
        const auto presentEnd = Clock::now();
        PROFILE_PLOT("End-to-End Present Job Latency (ms)",
            std::chrono::duration<double, std::milli>(
                presentEnd - timedJob.job.readyTime).count());
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
            const char kind =
                timedJob.job.frameKind == PresentedFrameKind::Interpolated ? 'G' : 'N';
            const uint64_t id =
                timedJob.job.frameKind == PresentedFrameKind::Interpolated
                    ? timedJob.job.previousNativeFrameId
                    : timedJob.job.currentNativeFrameId;
            const std::string message =
                "Presented seq=" + std::to_string(timedJob.job.presentSequenceId) +
                " " + kind + std::to_string(id) +
                (kind == 'G' ? ".5" : "");
            PROFILE_MESSAGE(message);
        }
        if (!isUnset(previousActualPresentTime)) {
            const double presentIntervalMs = std::chrono::duration<double, std::milli>(
                presentStart - previousActualPresentTime).count();
            PROFILE_PLOT("Present Interval (ms)", presentIntervalMs);
        }
        previousActualPresentTime = presentStart;

        {
            std::lock_guard<std::mutex> lock(completionMutex);
            completions.push_back({
                result,
                timedJob.job.swapchainGeneration,
                timedJob.job.timelineStep,
                timedJob.job.frameKind,
                std::move(timedJob.job.label)
            });
        }

        {
            std::lock_guard<std::mutex> lock(presentMutex);
            presentCallActive = false;
        }
        presentIdleCondition.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(presentMutex);
        presentCallActive = false;
    }
    presentIdleCondition.notify_all();
}

void VulkanFramePacer::queueTimedJob(VulkanPresentJob job, Clock::time_point targetTime) {
    if (!running.load() || paused.load()) {
        return;
    }

    job.presentSequenceId = nextPresentSequenceId.fetch_add(1);
    const char kind = job.frameKind == PresentedFrameKind::Interpolated ? 'G' : 'N';
    const uint64_t id =
        job.frameKind == PresentedFrameKind::Interpolated
            ? job.previousNativeFrameId
            : job.currentNativeFrameId;
    const std::string message =
        "Enqueued seq=" + std::to_string(job.presentSequenceId) +
        " " + kind + std::to_string(id) +
        (kind == 'G' ? ".5" : "");
    PROFILE_MESSAGE(message);

    {
        std::lock_guard<std::mutex> lock(presentMutex);
        presentQueue.push_back({ std::move(job), targetTime });
        PROFILE_PLOT("Present Queue Size", static_cast<int64_t>(presentQueue.size()));
    }
    presentCondition.notify_one();
}

void VulkanFramePacer::queueNativeOnly(VulkanPresentJob job, bool repeatedNative) {
    PROFILE_ZONE("Enqueue Native Present Job");
    const auto interval = repeatedNative ? halfFrameInterval() : nativeFrameInterval();
    queueTimedJob(std::move(job), nextTarget(interval));
}

void VulkanFramePacer::queueGeneratedAndNative(VulkanPresentJob generated, VulkanPresentJob native) {
    PROFILE_ZONE("Enqueue Generated + Native Present Jobs");
    const auto halfInterval = halfFrameInterval();
    const auto generatedTarget = nextTarget(halfInterval);
    const auto nativeTarget = nextTarget(halfInterval);

    // Generated N+0.5 is always inserted before native N+1. Both timestamps are
    // derived from the same moving-average interval to avoid a short/long pair.
    queueTimedJob(std::move(generated), generatedTarget);
    queueTimedJob(std::move(native), nativeTarget);
}

void VulkanFramePacer::updateNativeFrameAverage(const VulkanPresentJob& native) {
    if (!isUnset(previousNativeReadyTime)) {
        auto sample = native.readyTime - previousNativeReadyTime;
        sample = std::clamp(sample,
            std::chrono::duration_cast<Clock::duration>(MinimumNativeFrameInterval),
            std::chrono::duration_cast<Clock::duration>(MaximumNativeFrameInterval));
        nativeFrameSamples.push_back(sample);
        if (nativeFrameSamples.size() > NativeFrameWindow) {
            nativeFrameSamples.pop_front();
        }
    }

    previousNativeReadyTime = native.readyTime;
    previousNativeTimelineStep = native.timelineStep;
}

VulkanFramePacer::Clock::duration VulkanFramePacer::nativeFrameInterval() const {
    if (nativeFrameSamples.empty()) {
        return std::chrono::duration_cast<Clock::duration>(DefaultNativeFrameInterval);
    }

    Clock::duration total{};
    for (const auto sample : nativeFrameSamples) {
        total += sample;
    }
    return total / static_cast<int64_t>(nativeFrameSamples.size());
}

VulkanFramePacer::Clock::duration VulkanFramePacer::halfFrameInterval() const {
    return nativeFrameInterval() / 2;
}

VulkanFramePacer::Clock::time_point VulkanFramePacer::nextTarget(Clock::duration interval) {
    const auto now = Clock::now();
    if (isUnset(lastScheduledPresentTime) || lastScheduledPresentTime + interval < now) {
        lastScheduledPresentTime = now;
    }
    else {
        lastScheduledPresentTime += interval;
    }
    return lastScheduledPresentTime;
}

void VulkanFramePacer::clearQueuedJobsLocked() {
    interpolationQueue.clear();
    presentQueue.clear();
}
