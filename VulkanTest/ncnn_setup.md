# ncnn setup

Set the environment variable before opening/building the project:

- `NCNN_DIR` = path to your ncnn install root (contains `include` and `lib`)

Example:

- `NCNN_DIR=C:\libs\ncnn\install`

Then ensure these paths/libraries are available in Visual Studio project settings:

- Include: `$(NCNN_DIR)\include`
- Library dir: `$(NCNN_DIR)\lib`
- Linker input: `ncnn.lib`
