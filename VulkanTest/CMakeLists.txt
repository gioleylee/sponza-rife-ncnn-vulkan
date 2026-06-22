find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL Release)
find_package(ncnn CONFIG REQUIRED)
find_path(GLM_INCLUDE_DIR glm/glm.hpp REQUIRED)


set(IMGUI_DIR "" CACHE PATH "Path to Dear ImGui sources")
if(NOT EXISTS "${IMGUI_DIR}/imgui.cpp")
    message(FATAL_ERROR "Set IMGUI_DIR to the Dear ImGui source directory")
endif()

set(RENDERER_SOURCES
    src/main.cpp
    src/app/DescriptorResources.cpp
    src/app/ImGuiOverlay.cpp
    src/app/ModelLoader.cpp
    src/app/NcnnIntegration.cpp
    src/app/RenderResources.cpp
    src/app/Swapchain.cpp
    src/app/TextureLoader.cpp
    src/app/VulkanBuffers.cpp
    src/app/VulkanDevice.cpp
    src/app/VulkanFramePacer.cpp
    src/app/VulkanHelpers.cpp
    src/app/VulkanNcnnRenderer.cpp
    src/ncnn/NcnnFrameInterpolator.cpp
    src/platform/WindowInput.cpp
    src/platform/validation_layers.cpp
    third_party/imgui_impl_vulkan.cpp
    third_party/rife/rife.cpp
    third_party/rife/warp.cpp
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
)

if(ENABLE_TRACY)
    list(APPEND RENDERER_SOURCES third_party/tracy/public/TracyClient.cpp)
endif()

add_executable(VulkanTest ${RENDERER_SOURCES})
target_compile_features(VulkanTest PRIVATE cxx_std_20)
target_compile_definitions(VulkanTest PRIVATE VK_USE_PLATFORM_WIN32_KHR)
target_include_directories(VulkanTest PRIVATE
    include/app
    include/platform
    third_party
    third_party/rife
    third_party/nvtx/c/include
    third_party/tracy/public
    "${IMGUI_DIR}"
    "${GLM_INCLUDE_DIR}"
)
target_link_libraries(VulkanTest PRIVATE Vulkan::Vulkan glfw assimp::assimp ncnn)

if(ENABLE_TRACY)
    target_compile_definitions(VulkanTest PRIVATE ENABLE_TRACY TRACY_ENABLE)
endif()

set_target_properties(VulkanTest PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
