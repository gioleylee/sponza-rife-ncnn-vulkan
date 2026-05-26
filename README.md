# Vulkan + RIFE Sponza Renderer with NCNN

## Overview

This project is a Vulkan-based Sponza renderer with real-time frame interpolation using RIFE and NCNN.

---

### Required Tools

- Visual Studio
  - Install the **Desktop development with C++** workload
- Windows SDK
- Vulkan SDK
  - Make sure the `VULKAN_SDK` environment variable is set

---

Visual Studio Settings
C/C++ > General > Additional Include Directories
 - NCNN include
 - Vulkan include
 - GLM include
 - GLFW include
 - Assimp include
 - $(ProjectDir)rife_src

Linker > General > Additional Library Directories
 - GLFW lib
 - Vulkan lib
 - NCNN lib
 - Assimp lib

Linker > Input > Additional Dependencies
 - vulkan-1.lib
 - glfw3.lib
 - ncnnd.lib
 - glslangd.lib
 - SPIRV-Toolsd.lib
 - SPIRV-Tools-optd.li
 - assimp-vc143-mt.lib

Build Events > Post-Build Event
 - Copy Assimp DLL to output directory.