cmake_minimum_required(VERSION 3.24)
project(vulkan_ncnn_rendering LANGUAGES CXX)

option(ENABLE_TRACY "Enable Tracy CPU and Vulkan GPU profiling" OFF)

add_subdirectory(VulkanTest)
