# Tracy CPU profiling

Tracy v0.13.1 client sources are vendored under `third_party/tracy`.

Profiling is disabled by default. Build an instrumented executable with:

```powershell
msbuild VulkanTest.slnx /p:Configuration=Debug /p:Platform=x64 /p:EnableTracy=true
```

Run a Tracy v0.13.1 profiler client, then launch `x64/Debug/VulkanTest.exe`
with `VulkanTest/` as its working directory.

Use `/p:EnableTracy=false` or omit the property for a normal build. In that
configuration `TracyClient.cpp` is excluded and all profiling macros are no-ops.

For CMake builds:

```powershell
cmake -S .. -B ../out/tracy -DENABLE_TRACY=ON -DIMGUI_DIR=C:/path/to/imgui
cmake --build ../out/tracy --config Debug
```

The existing Visual Studio project remains the preferred build for the current
machine-specific NCNN and dependency layout.
