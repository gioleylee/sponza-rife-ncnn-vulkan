// Owns window creation, resize callbacks, keyboard input, and mouse-look camera updates.
#include "WindowInput.h"

void HelloTriangleApplication::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanTestSponza", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void HelloTriangleApplication::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void HelloTriangleApplication::updateCameraFrontFromAngles() {
    float radYaw = glm::radians(cameraYaw);
    float radPitch = glm::radians(cameraPitch);
    glm::vec3 front;
    front.x = cos(radYaw) * cos(radPitch);
    front.y = sin(radPitch);
    front.z = sin(radYaw) * cos(radPitch);
    cameraFront = glm::normalize(front);
}

void HelloTriangleApplication::processInput(float deltaTime) {
    float velocity = cameraSpeed * deltaTime;
    bool cameraOrientationChanged = false;

    glm::vec3 forward = glm::normalize(glm::vec3(cameraFront.x, 0.0f, cameraFront.z));
    glm::vec3 right = glm::normalize(glm::cross(forward, cameraUp));

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        cameraPos += forward * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        cameraPos -= forward * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        cameraPos -= right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        cameraPos += right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        cameraPos.y += velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        cameraPos.y -= velocity;
    }

    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) {
        if (!nKeyPressed) {
            showNormals = !showNormals;
            nKeyPressed = true;
        }
    } else {
        nKeyPressed = false;
    }

    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
        if (!bKeyPressed) {
            showAlbedo = !showAlbedo;
            bKeyPressed = true;
        }
    } else {
        bKeyPressed = false;
    }

    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
        if (!vKeyPressed) {
            showPosition = !showPosition;
            vKeyPressed = true;
        }
    } else {
        vKeyPressed = false;
    }

    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
        if (!mKeyPressed) {
            showSpecular = !showSpecular;
            mKeyPressed = true;
        }
    } else {
        mKeyPressed = false;
    }

#if HAS_NCNN
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        if (!rKeyPressed) {
            rifeRealtimeInterpolationEnabled = !rifeRealtimeInterpolationEnabled;
            waitForAsyncRifeInference();
            rifeInferenceRequestWaitingForFramePair = false;
            hasRifeGpuFramePair = false;
            hasRifeDisplayFrame = false;
            for (auto& output : rifeOutputBuffers) {
                output.ready = false;
                output.inUseByInference = false;
                output.inUseByGraphics = false;
                output.graphicsFrameSlot = UINT32_MAX;
                output.sequence = 0;
            }
            nextRifeOutputSequence = 1;
            rifeInferenceScaleDivisor = RIFE_INITIAL_INFERENCE_SCALE_DIVISOR;
            rifeCompletedInferenceCount = 0;
            capturedFrameCount = 0;
            currentRifeGpuFrameIndex = UINT32_MAX;
            previousRifeGpuFrameIndex = UINT32_MAX;
            previousFrameCaptureProcessMs = 0.0;
            lastFrameCaptureProcessMs = 0.0;
            lastFramePairCaptureProcessMs = 0.0;
            pendingCaptureSlotByFrame.fill(UINT32_MAX);
            std::cout << "[RIFE] realtime interpolation "
                      << (rifeRealtimeInterpolationEnabled ? "enabled" : "disabled")
                      << std::endl;
            rKeyPressed = true;
        }
    } else {
        rKeyPressed = false;
    }
#endif

    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        if (!tKeyPressed) {
            autoPanEnabled = !autoPanEnabled;
            std::cout << "[CAMERA] auto pan " << (autoPanEnabled ? "enabled" : "disabled")
                      << " (speed=" << autoPanSpeedDegreesPerSecond << " deg/s)" << std::endl;
            tKeyPressed = true;
        }
    } else {
        tKeyPressed = false;
    }

    if (autoPanEnabled) {
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            if (!oneKeyPressed) {
                autoPanSpeedDegreesPerSecond = std::min(120.0f, autoPanSpeedDegreesPerSecond * 1.25f);
                std::cout << "[CAMERA] auto pan speed increased to "
                          << autoPanSpeedDegreesPerSecond << " deg/s" << std::endl;
                oneKeyPressed = true;
            }
        } else {
            oneKeyPressed = false;
        }

        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            if (!twoKeyPressed) {
                autoPanSpeedDegreesPerSecond = std::max(0.25f, autoPanSpeedDegreesPerSecond / 1.25f);
                std::cout << "[CAMERA] auto pan speed decreased to "
                          << autoPanSpeedDegreesPerSecond << " deg/s" << std::endl;
                twoKeyPressed = true;
            }
        } else {
            twoKeyPressed = false;
        }

        cameraYaw += autoPanSpeedDegreesPerSecond * deltaTime;
        cameraOrientationChanged = true;
    } else {
        oneKeyPressed = false;
        twoKeyPressed = false;
    }

    if (cameraOrientationChanged) {
        updateCameraFrontFromAngles();
    }
}

void HelloTriangleApplication::processMouseLook() {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) {
        firstMouse = true;
        return;
    }

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (firstMouse) {
        lastMouseX = xpos;
        lastMouseY = ypos;
        firstMouse = false;
    }

    double xoffset = xpos - lastMouseX;
    double yoffset = lastMouseY - ypos;
    lastMouseX = xpos;
    lastMouseY = ypos;

    float sensitivity = mouseSensitivity;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    cameraYaw += static_cast<float>(xoffset);
    cameraPitch += static_cast<float>(yoffset);

    if (cameraPitch > 89.0f)  cameraPitch = 89.0f;
    if (cameraPitch < -89.0f) cameraPitch = -89.0f;

    updateCameraFrontFromAngles();
}

