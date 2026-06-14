// Owns window creation, resize callbacks, keyboard input, and mouse-look camera updates.
#include "WindowInput.h"

#include <GLFW/glfw3.h>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

void VulkanNcnnRenderer::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanNcnnRenderer", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VulkanNcnnRenderer::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanNcnnRenderer*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void VulkanNcnnRenderer::updateCameraFrontFromAngles() {
    float radYaw = glm::radians(cameraYaw);
    float radPitch = glm::radians(cameraPitch);
    glm::vec3 front;
    front.x = cos(radYaw) * cos(radPitch);
    front.y = sin(radPitch);
    front.z = sin(radYaw) * cos(radPitch);
    cameraFront = glm::normalize(front);
}

void VulkanNcnnRenderer::processInput(float deltaTime) {
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

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        if (!rightMousePressed) {
            imguiVisible = !imguiVisible;
            rightMousePressed = true;
        }
    } else {
        rightMousePressed = false;
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

    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
        if (!yKeyPressed) {
            markInterpolatedFrames = !markInterpolatedFrames;
            std::cout << "[NCNN] interpolated frame marker "
                      << (markInterpolatedFrames ? "enabled" : "disabled")
                      << std::endl;
            yKeyPressed = true;
        }
    } else {
        yKeyPressed = false;
    }

#if HAS_NCNN
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        if (!rKeyPressed) {
            setNcnnRealtimeInterpolationEnabled(!ncnnPresentationState.ncnnRealtimeInterpolationEnabled);
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

void VulkanNcnnRenderer::processMouseLook() {
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

