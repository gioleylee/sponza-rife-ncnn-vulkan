# Vulkan + RIFE Sponza Renderer w/ NCNN

## 📌 Overview
This project implements a real-time neural rendering pipeline using Vulkan and RIFE, featuring the Sponza scene as a benchmark environment and utilizing NCNN for Vulkan + RIFE.

## ⚙️ Setup
In Visual Studio, open the .slnx folder and build

Install/build Assimp locally and update these Visual Studio settings:

  1. C/C++ > General > Additional Include Directories
     Add their Assimp include folder.
  2. Linker > General > Additional Library Directories
     Add their Assimp lib folder.
  3. Linker > Input > Additional Dependencies
     Keep assimp-vc143-mt.lib, or replace it with the exact .lib name they have.
  4. Build Events > Post-Build Event
     Update the DLL copy path to wherever their assimp-vc143-mt.dll is installed.

  After changing those settings, reload the project and do a clean rebuild.

Install NCNN, setup is shown in ncnn_setup.md

Required Tools

  - Visual Studio with C++ desktop workload.
  - Windows SDK.
  - Vulkan SDK installed, with VULKAN_SDK environment variable set.
  - A Vulkan-capable GPU driver/runtime.

  Required Libraries

  - GLFW 3
      - Include path: ...\glfw-3.4.bin.WIN64\include
      - Lib path: ...\glfw-3.4.bin.WIN64\lib-vc2022
      - Linker input: glfw3.lib
  - GLM
      - Header-only.
      - Include path should point to the folder containing glm\glm.hpp.
  - Assimp
      - Include path: folder containing assimp\Importer.hpp
      - Lib path: folder containing the Assimp .lib
      - Linker input currently expects: assimp-vc143-mt.lib
      - Runtime DLL must be copied beside the .exe: assimp-vc143-mt.dll
      
  - NCNN built with Vulkan support
      - Include path: NCNN install include
      - Lib path: NCNN install lib
      - Current linker inputs expect:
          - ncnnd.lib
          - glslangd.lib
          - SPIRV-Toolsd.lib
          - SPIRV-Tools-optd.lib
  - Vulkan SDK libraries
      - Include path: $(VULKAN_SDK)\Include
      - Lib path: $(VULKAN_SDK)\Lib
      - Linker input: vulkan-1.lib

Visual Studio Settings Users Must Update

  - C/C++ > General > Additional Include Directories
      - NCNN include
      - Vulkan include
      - GLM include
      - GLFW include
      - Assimp include
      - $(ProjectDir)rife_src
  - Linker > General > Additional Library Directories
      - GLFW lib
      - Vulkan lib
      - NCNN lib
      - Assimp lib
  - Linker > Input > Additional Dependencies
      - vulkan-1.lib
      - glfw3.lib
      - ncnnd.lib
      - glslangd.lib
      - SPIRV-Toolsd.lib
      - SPIRV-Tools-optd.lib
      - assimp-vc143-mt.lib
  - Build Events > Post-Build Event
      - Copy Assimp DLL to output directory.

