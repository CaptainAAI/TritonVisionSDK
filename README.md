# TritonVision

A small C++ library (Windows DLL) for object detection through an
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) over gRPC.
An application calls 5 functions; gRPC, Protobuf and the Triton client are built into
`TritonVision.dll`, so the application never compiles or links them.

```cpp
#include "TritonVision.h"

if (!Connect("192.168.1.156:8001", "model1"))        // once
    std::cerr << LastError() << "\n";

std::vector<Detection> dets = Detect(frame);          // per image (cv::Mat, BGR)
if (dets.empty() && !LastError().empty())
    std::cerr << LastError() << "\n";                 // the call failed
Draw(frame, dets);                                    // boxes + labels
```

## Contents

- [API](#api)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Property sheets (.props)](#property-sheets-props)
- [Building the DLL](#building-the-dll)
- [Using the DLL in another project](#using-the-dll-in-another-project)
- [Class names](#class-names)
- [Supported model outputs](#supported-model-outputs)
- [Regenerating the gRPC code](#regenerating-the-grpc-code)
- [Troubleshooting](#troubleshooting)
- [Third-party code](#third-party-code)

## API

Declared in [`SDKsourceCode/include/TritonVision.h`](SDKsourceCode/include/TritonVision.h).

| Function | Description |
| --- | --- |
| `bool Connect(const std::string& url, const std::string& model)` | Connects to Triton and reads the model's input/output metadata. `url` is `"IP:port"` of the **gRPC** endpoint (default port 8001). Returns `false` on failure. |
| `std::vector<Detection> Detect(const cv::Mat& frame, float conf = 0.25f, float nms = 0.45f)` | Runs detection on one BGR image of any size. Boxes are returned in original image pixels. |
| `void Draw(cv::Mat& frame, const std::vector<Detection>& dets)` | Draws each box and `label score` onto the image (in place). |
| `std::string LastError()` | Error message of the last `Connect`/`Detect`; empty string = success. |
| `double LastInferMs()` | Duration of the last inference request in milliseconds (network + server). |

```cpp
struct Detection {
    cv::Rect    box;       // x, y, width, height in pixels of the input image
    float       score;     // confidence, 0.0 - 1.0
    int         class_id;  // class index returned by the model
    std::string label;     // class name, or the class index as text
};
```

Behaviour to know:

- An empty result from `Detect` means either *no objects* or *error*. Check `LastError()`.
- `Detect` blocks until the server answers. In a GUI, call it from a worker thread to keep the UI responsive.
- The library holds **one global connection** (one server, one model). Use it from one thread at a time.
  Calling `Connect` again replaces the connection.
- The server address can come from anywhere (variable, config file, text box):

  ```cpp
  std::string ip = "192.168.1.156";
  int port = 8001;
  Connect(ip + ":" + std::to_string(port), "model1");
  ```

## Repository layout

```
SDKsourceCode.sln                 Visual Studio 2022 solution
UseTritonVision.props             property sheet for projects that USE the DLL
SDKsourceCode/
├─ SDKsourceCode.vcxproj          DLL project (output: TritonVision.dll)
├─ TritonVision.Build.props       include paths, defines, libraries, dependency locations
├─ include/TritonVision.h         public API
├─ src/TritonVision.cpp           implementation (pre-process, inference, post-process)
├─ triton/                        Triton gRPC client (from NVIDIA)
├─ generated/                     C++ code generated from proto/ by protoc + grpc_cpp_plugin
└─ proto/                         Triton service definitions (.proto)
```

Build output goes to `bin/x64/<Debug|Release>/` and intermediate files to `obj/`. Both are ignored by git.

## Requirements

| Item | Version used |
| --- | --- |
| Windows | 10/11, **x64** |
| Visual Studio | 2022 with *Desktop development with C++* (toolset v143) |
| gRPC | 1.82.0, built with MSVC, **Release, /MD**, static libraries |
| Protobuf | 35.0 (C++ runtime 7.35.0), from the same gRPC build |
| Abseil | 20250512, from the same gRPC build |
| OpenCV | 5.0.0, prebuilt Windows package (`opencv_world500`, `x64/vc16`) |

The dependencies are **not** part of this repository.

> The files in `generated/` only compile against the Protobuf version that produced them
> (they check `PROTOBUF_VERSION == 7035000`). With a different gRPC/Protobuf build,
> [regenerate them](#regenerating-the-grpc-code) first.

### Where the build looks for gRPC and OpenCV

`TritonVision.Build.props` resolves each dependency in this order:

| | gRPC | OpenCV |
| --- | --- | --- |
| 1. MSBuild property | `/p:GrpcRoot=<folder>` | `/p:OpenCVRoot=<folder>` |
| 2. Environment variable | `TRITONVISION_GRPC_ROOT` | `TRITONVISION_OPENCV_ROOT` |
| 3. Default (relative to the project folder) | `..\..\..\library\grpc-msvc\` | `..\..\..\library\opencv\build\` |

- The **gRPC folder** must contain `include\grpcpp\grpcpp.h`, `lib\grpc++.lib` and `bin\protoc.exe`.
- The **OpenCV folder** is the package's `build` folder: it contains `include\opencv2\` and `x64\vc16\lib\opencv_world500.lib`.

If a folder is not found, the build stops with a message naming the path it tried.

## Property sheets (.props)

A `.props` file (MSBuild property sheet) holds Visual Studio project settings: include
directories, library directories, the list of `.lib` files to link, defines, runtime library.
A project imports it once instead of setting each of these by hand in **Project → Properties**,
for every configuration. When a setting changes, only the `.props` file is edited.

This repository has two:

| File | Used by | What it sets |
| --- | --- | --- |
| [`SDKsourceCode/TritonVision.Build.props`](SDKsourceCode/TritonVision.Build.props) | The DLL project itself (already imported by `SDKsourceCode.vcxproj`) | gRPC and OpenCV locations, include paths, `TRITONVISION_EXPORTS`, `/MD`, C++17, and the ~110 gRPC / Protobuf / Abseil libraries |
| [`UseTritonVision.props`](UseTritonVision.props) | Projects that **use** the DLL | `TritonVision.h` and OpenCV include paths, `TritonVision.lib` + `opencv_world500.lib`, `/MD`, `_DEBUG` undefined, C++17, and copying the DLLs next to the `.exe` |

A property sheet is imported in one of two ways:

- **Visual Studio:** **View → Other Windows → Property Manager**, right-click a configuration
  (e.g. **Debug | x64**) → **Add Existing Property Sheet…** → select the `.props` file.
- **Project file:** one line in the `.vcxproj`, inside the `PropertySheets` import group:
  ```xml
  <Import Project="..\TritonVisionSDK\UseTritonVision.props" />
  ```

Settings in a `.props` file use `%(…)` (for example `%(AdditionalDependencies)`), so they are added
to the project's own settings instead of replacing them.

## Building the DLL

### Visual Studio

1. (Optional) Set the environment variables, then restart Visual Studio:
   ```bash
   setx TRITONVISION_GRPC_ROOT "C:\deps\grpc-msvc"
   ```
   ```bash
   setx TRITONVISION_OPENCV_ROOT "C:\deps\opencv\build"
   ```
2. Open `SDKsourceCode.sln`.
3. Select **x64** and **Debug** or **Release**, then **Build → Build Solution**.

### Command line

From a *Developer Command Prompt for VS 2022*, in the folder containing `SDKsourceCode.sln`:

```bash
msbuild SDKsourceCode.sln /p:Configuration=Release /p:Platform=x64 /p:GrpcRoot=C:\deps\grpc-msvc /p:OpenCVRoot=C:\deps\opencv\build
```

### Output

```
bin/x64/Release/TritonVision.dll   the library (gRPC included, about 10 MB)
bin/x64/Release/TritonVision.lib   import library for projects that use the DLL
bin/x64/Release/TritonVision.pdb   debug symbols
```

The DLL depends on `opencv_world500.dll` and the Visual C++ runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`).

### Build rules and why

| Rule | Reason |
| --- | --- |
| x64 only | gRPC and OpenCV are x64 builds. |
| `/MD` in **every** configuration, no `_DEBUG` | gRPC exists only as a Release build (`_ITERATOR_DEBUG_LEVEL=0`). Linking it with the debug runtime (`/MDd`) fails with `LNK2038`. The Debug configuration still has no optimisation and full debug symbols. |
| No `/clr` | gRPC and Abseil headers use `<mutex>`, `<thread>` and `<atomic>`, which cannot be compiled as managed C++/CLI code. Keeping them inside a native DLL means C++/CLI applications can still use the library: they only include `TritonVision.h`. |

## Using the DLL in another project

The calling project needs `TritonVision.h`, the OpenCV headers, `TritonVision.lib` and
`opencv_world500.lib`, and must use the same C++ runtime as the DLL, because
`std::string`, `std::vector` and `cv::Mat` cross the DLL boundary.

### Option A: `UseTritonVision.props` (recommended)

1. Build this repository's `SDKsourceCode.sln` in **Release | x64** (produces `bin/x64/Release/TritonVision.lib` and `.dll`).
2. In your project, add [`UseTritonVision.props`](UseTritonVision.props) to **Debug | x64** and **Release | x64**
   ([how](#property-sheets-props)).
3. Set the platform to **x64**, `#include "TritonVision.h"`, and build.

`TritonVision.dll` and `opencv_world500.dll` are copied next to your `.exe` after each build.
The build stops with a clear message if the platform is not x64, the DLL has not been built yet,
or OpenCV is not found. In Debug, warning D9025 (`overriding '/D_DEBUG' with '/U_DEBUG'`) is expected.

### Option B: manual settings

| Project property (x64, all configurations) | Value |
| --- | --- |
| C/C++ → General → Additional Include Directories | `<repo>\SDKsourceCode\include`; `<OpenCV>\include` |
| C/C++ → Code Generation → Runtime Library | **Multi-threaded DLL (/MD)**, also in Debug |
| C/C++ → Preprocessor → Undefine Preprocessor Definitions | `_DEBUG` |
| C/C++ → Language → C++ Language Standard | ISO C++17 or newer |
| Linker → General → Additional Library Directories | `<repo>\bin\x64\Release`; `<OpenCV>\x64\vc16\lib` |
| Linker → Input → Additional Dependencies | `TritonVision.lib;opencv_world500.lib` |

At run time, place `TritonVision.dll` and `opencv_world500.dll` next to the application's `.exe`
(for example with a post-build event: `xcopy /y /d "<path>\*.dll" "$(OutDir)"`).

Do not define `TRITONVISION_EXPORTS` in the calling project; it switches the header to export mode.

### Minimal example

```cpp
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "TritonVision.h"

int main()
{
    if (!Connect("192.168.1.156:8001", "model1")) {
        std::cerr << "connect failed: " << LastError() << "\n";
        return 1;
    }

    cv::Mat img = cv::imread("test.jpg");
    std::vector<Detection> dets = Detect(img);
    if (dets.empty() && !LastError().empty()) {
        std::cerr << "detect failed: " << LastError() << "\n";
        return 1;
    }

    for (const Detection& d : dets)
        std::cout << d.label << " " << d.score << " " << d.box << "\n";

    Draw(img, dets);
    cv::imwrite("result.jpg", img);
    std::cout << dets.size() << " objects, infer " << LastInferMs() << " ms\n";
    return 0;
}
```

## Class names

`kClassNames` in `src/TritonVision.cpp` is **empty**, so `Detection::label` is the class index as text
(`"0"`, `"1"`, ...). The application maps `class_id` to its own names before drawing:

```cpp
std::vector<std::string> names = { "sensor", "particle" };   // in the model's training order

for (Detection& d : dets)
    if (d.class_id >= 0 && d.class_id < (int)names.size())
        d.label = names[d.class_id];

Draw(frame, dets);
```

To build fixed names into the DLL instead, fill `kClassNames` in `src/TritonVision.cpp` and rebuild.

## Supported model outputs

`Detect` sends one `FP32` input tensor shaped `[1, 3, H, W]` (RGB, values 0..1, letterboxed with grey 114);
`H` and `W` are read from the model metadata (default 640×640). The first output tensor is decoded in
one of two YOLO-style layouts:

| Output shape | Layout | NMS |
| --- | --- | --- |
| `[1, N, 6]` | per row: `x1, y1, x2, y2, score, class` | already done by the model |
| `[1, 4 + nc, N]` | per candidate: `cx, cy, w, h`, then one score per class (column-wise) | done in `Detect` (`cv::dnn::NMSBoxes`, per class) |

Other output formats need changes in `src/TritonVision.cpp`.

## Regenerating the gRPC code

Needed after changing the gRPC/Protobuf version or updating the `.proto` files.
From the `SDKsourceCode` project folder, with `GRPC` set to the gRPC install folder:

```bash
"%GRPC%\bin\protoc.exe" --proto_path=proto --cpp_out=generated --grpc_out=generated --plugin=protoc-gen-grpc="%GRPC%\bin\grpc_cpp_plugin.exe" proto/grpc_service.proto proto/health.proto proto/model_config.proto
```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `gRPC not found at '...'` / `OpenCV not found at '...'` | Dependency folder not found | Set the environment variable or `/p:` property ([details](#where-the-build-looks-for-grpc-and-opencv)) |
| `Protobuf C++ gencode is built with an incompatible version of Protobuf C++ headers/runtime` | `generated/` does not match the installed Protobuf | [Regenerate the gRPC code](#regenerating-the-grpc-code) |
| `LNK2038: mismatch detected for '_ITERATOR_DEBUG_LEVEL'` | `/MDd` or `_DEBUG` in this or the calling project | Use `/MD` and undefine `_DEBUG` |
| `LNK1112` / missing libraries | Platform is Win32 | Build x64 |
| `LNK1104: cannot open file 'TritonVision.dll'` | The DLL is in use by a running program | Close the program or debugger, then rebuild |
| `TritonVision.lib not found in '...'` (from `UseTritonVision.props`) | The DLL has not been built | Build `SDKsourceCode.sln` in Release \| x64 |
| `warning D9025: overriding '/D_DEBUG' with '/U_DEBUG'` | `UseTritonVision.props` undefines `_DEBUG` on purpose | Expected; ignore |
| Crash or garbled strings in `Detect` / `LastError` in the calling app | The app uses the debug runtime | Same runtime as the DLL: `/MD`, no `_DEBUG` |
| `Connect` fails: `failed to connect to all addresses` / `UNAVAILABLE` | Wrong IP, HTTP port 8000 instead of gRPC 8001, server down, firewall | Check the address and that the server is reachable |
| `Connect` fails with a model error | Model name wrong or model not READY | Use the exact name from the Triton model repository |
| C++/CLI (WinForms) app crashes at start-up with `0xC0000374` | Not caused by this DLL; happens when gRPC is compiled **into** a C++/CLI exe whose entry point is `main` | Use this DLL instead, or set Linker → Advanced → Entry Point to `mainCRTStartup` |

## Third-party code

| Component | Location | License |
| --- | --- | --- |
| Triton client (`grpc_client`, `common`, `ipc`) and `.proto` files | `triton/`, `proto/` | BSD 3-Clause, © NVIDIA Corporation (notice kept in each file) |
| Generated gRPC/Protobuf code | `generated/` | Generated from `proto/` |
| gRPC, Protobuf, Abseil, BoringSSL, RE2, c-ares, zlib | linked statically into the DLL, not in this repository | Apache 2.0, BSD 3-Clause, Apache 2.0, OpenSSL/ISC, BSD 3-Clause, MIT, zlib |
| OpenCV | linked dynamically, not in this repository | Apache 2.0 |

Distributing `TritonVision.dll` means distributing the statically linked libraries above;
include their license texts with the binary.
