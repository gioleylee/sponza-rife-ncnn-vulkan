# Vulkan + RIFE Sponza Renderer with NCNN

## Overview

This project is a Vulkan-based Sponza renderer with real-time frame interpolation using RIFE and NCNN.

---

### Controls
- Key Inputs:
  - "R": Toggle Frame Interpolation (ON/OFF)
  - "T": Auto-rotate Camera (ON/OFF)
  - "1": Speeds up auto-rotate (if enabled)
  - "2": Slows down auto-rotate (if enabled)

### Required Tools

- Visual Studio
  - **Desktop development with C++**
- Windows SDK
- Vulkan SDK
  - `VULKAN_SDK` set environment variable
- NCNN
  - Setup given in ncnn_setup.md

---

Visual Studio Settings
C/C++ > General > Additional Include Directories
 - NCNN include
 - Vulkan include
 - GLM include
 - GLFW include
 - Assimp include
 - $(ProjectDir)include\app
 - $(ProjectDir)include\platform
 - $(ProjectDir)third_party\rife

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
 - SPIRV-Tools-optd.lib
 - assimp-vc143-mt.lib

RIFE model files are loaded from `VulkanTest/assets/models/rife-v4`.
