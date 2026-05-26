# Vulkan Sponza Renderer

## 📌 Overview
This project implements a real-time rendering pipeline using Vulkan, featuring the Sponza scene as a benchmark environment.

## 🚀 Features
- Vulkan-based real-time renderer
- Sponza scene rendering (complex geometry & lighting)
- GPU memory management using Vulkan buffers and images
- 
## 🧠 Motivation
The goal of this project is to build a rendering pipeline and understand the mechanisms behind Vulkan graphics rendering

## 🏗️ Tech Stack
- **Graphics API:** Vulkan
- **Language:** C++
- **Windowing:** GLFW
- **Build System:** CMake / Visual Studio

## 🖼️ Demo
(Add screenshots or GIFs here)

## ⚙️ Setup & Installation

### Prerequisites
- Vulkan SDK installed
- C++ compiler (MSVC recommended)
- GLFW

### Build Instructions

#### Using CMake (recommended)
Dependencies are resolved through CMake packages (`Vulkan`, `glfw3`, `glm`, `assimp`).

Set one of the following to the folder containing `stb_image.h`:
- `STB_INCLUDE_DIR` (CMake variable), or
- `STB_DIR` (environment variable)

Optional ncnn support:
- Set `NCNN_DIR` to your ncnn install root (contains `include` and `lib`).

```bash
cmake -S . -B build -DSTB_INCLUDE_DIR="C:/path/to/stb"
cmake --build build --config Release
```

In Visual Studio, open the repository folder directly to use CMake project mode.
"# sponza-rife-ncnn-vulkan" 
