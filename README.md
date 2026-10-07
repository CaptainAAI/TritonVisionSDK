# TritonVision

A small C++ library (Windows DLL) for object detection through an
[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) over gRPC.
An application calls a handful of functions; gRPC, Protobuf and the Triton client are built into
`TritonVision.dll`, so the application never compiles or links them.

Two ways to detect, usable side by side:

- `Connect` + `Detect`: the image is pre-processed **on the PC** and sent as a float tensor (~4.9 MB at 640×640).
- `ConnectImage` + `DetectImage`: only the **encoded image** (JPEG/PNG, typically 50–300 KB) is sent;
  pre-processing runs **on the server's GPU** with NVIDIA DALI (e.g. on the Jetson). Needs the server models from
  [Server-side pre-processing](#server-side-pre-processing).

```cpp
#include "TritonVision.h"

if (!Connect("140.129.7.180:8001", "model1"))        // once
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
- [Sending images to DetectImage (JPEG/PNG)](#sending-images-to-detectimage-jpegpng)
- [Server-side pre-processing](#server-side-pre-processing)
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
| `std::string LastError()` | Error message of the last `Connect`/`Detect`/`ConnectImage`/`DetectImage`; empty string = success. |
| `double LastInferMs()` | Duration of the last inference request in milliseconds (network + server). For `DetectImage` this includes the server's pre-processing. Work on the PC (pre-processing in `Detect`, JPEG encoding, post-processing) is not included; see [what LastInferMs measures](#what-lastinferms-measures). |
| `bool ConnectImage(const std::string& url, const std::string& ensemble)` | Connects to Triton for [server-side pre-processing](#server-side-pre-processing). `ensemble` is the ensemble model name, e.g. `"model1_image"`. Returns `false` on failure, also when the model is not a pre-processing ensemble. |
| `std::vector<Detection> DetectImage(const cv::Mat& frame, float conf = 0.25f, float nms = 0.45f, int jpegQuality = 90)` | Like `Detect`, but JPEG-encodes the BGR image and lets the server pre-process it. |
| `std::vector<Detection> DetectImage(const std::vector<uint8_t>& encoded, float conf = 0.25f, float nms = 0.45f)` | Same, for bytes that are already a JPEG/PNG file (sent as is, no decoding on the PC). Other data is rejected before sending, with `LastError()` = `not a JPEG or PNG image`. |

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
  `ConnectImage` has its own separate connection, so `Detect` and `DetectImage` can be used in the same program.
- Boxes from `Detect` and `DetectImage(frame)` on the same image differ by a few pixels at most (JPEG compression);
  pass the file's bytes to `DetectImage(bytes)`, or use `jpegQuality = 100`, to avoid that
  ([details](#sending-images-to-detectimage-jpegpng)).
- The server address can come from anywhere (variable, config file, text box):

  ```cpp
  std::string ip = "140.129.7.180";
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
server/model_repository/          Triton models for DetectImage (copy to the server)
├─ preprocess/                    DALI (GPU): decode + letterbox + RGB + /255 + CHW
└─ model1_image/ … model3_image/  ensembles: preprocess -> model1 … model3
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
    if (!Connect("140.129.7.180:8001", "model1")) {
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

Other output formats need changes in `src/TritonVision.cpp`. `DetectImage` decodes the ensemble's
detection output with the same code (`Postprocess` in `src/TritonVision.cpp`).

## Sending images to DetectImage (JPEG/PNG)

`DetectImage` sends an **encoded image file** (JPEG or PNG) to the server, not pixels. There are two ways
to give it one:

| You have | Use | What happens on the PC |
| --- | --- | --- |
| A `.jpg`/`.png` file, or JPEG bytes from a camera (e.g. MJPEG) | `DetectImage(bytes)` | Nothing: the bytes are sent as they are |
| A `cv::Mat` (from `cv::imread`, `cv::VideoCapture`, a camera SDK, …) | `DetectImage(frame)` | The image is JPEG-encoded (`cv::imencode`, quality `jpegQuality`, default 90), then sent |
| A `.bmp`, `.tif` or other format | `cv::imread` it, then `DetectImage(frame)` | As above |

### Sending a file as it is (recommended for image files)

The file is not decoded or re-compressed on the PC, so the server sees exactly the pixels in the file.
This gives the same detections as `Detect` and is the fastest option.

```cpp
#include <fstream>
#include <iterator>

std::ifstream f("image.jpg", std::ios::binary);
std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

std::vector<Detection> dets = DetectImage(bytes);
if (dets.empty() && !LastError().empty())
    std::cerr << LastError() << "\n";

// To draw the result, decode the same bytes on the PC (only for drawing):
cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
Draw(img, dets);
```

### Sending a cv::Mat

```cpp
cv::Mat frame;                      // BGR, CV_8UC3, any size - same input as Detect()
cap >> frame;
std::vector<Detection> dets = DetectImage(frame);          // JPEG quality 90
std::vector<Detection> best = DetectImage(frame, 0.25f, 0.45f, 100);   // JPEG quality 100
```

JPEG is lossy: the pixels the server decodes are slightly different from `frame`, so scores can move a
little. Detections whose score is close to `conf` can appear or disappear compared to `Detect`.
The quality setting trades request size against that difference:

| `jpegQuality` | Request size | Difference to `Detect` |
| --- | --- | --- |
| 100 | largest JPEG (still far smaller than the ~4.9 MB float tensor of `Detect`) | smallest |
| 90 (default) | e.g. ~1 MB for a 4K frame | boxes within a few pixels; borderline detections may change |
| lower | smaller | larger |

Measured on 41 particle images (1280×720 JPEG files, ~90–115 KB each, `model1`, `conf = 0.25`):

| Call | Detections found of the 38 from `Detect` | Box difference |
| --- | --- | --- |
| `DetectImage(fileBytes)` | 38 / 38 | ≤ 1 px |
| `DetectImage(frame)`, quality 90 | 36 / 38 | ≤ 1 px |

The 2 missing detections were borderline: score 0.257 and 0.260 with `Detect`, 0.243 after re-encoding.
Re-encoding an image that was already a JPEG compresses it twice; send the file bytes instead when you have them.

### What is accepted

`DetectImage(bytes)` checks the first bytes of the data before sending anything:

| Starts with | Format | Result |
| --- | --- | --- |
| `FF D8 FF` | JPEG | sent |
| `89 50 4E 47` (`.PNG`) | PNG | sent |
| anything else (BMP, TIFF, raw pixels, text, empty) | | not sent; `LastError()` = `not a JPEG or PNG image` (or `empty image`) |

The check exists because the server's DALI model cannot skip data it fails to decode: one request that is
not an image breaks it for every client until it is reloaded
([details](#things-to-know-about-dali-on-the-jetson)). Only the start of the data is checked; a truncated
JPEG was tested on the server and did not break it. Clients that call Triton directly, without this library,
do not have this check.

`DetectImage(frame)` always sends a valid JPEG it made itself, so the check never rejects it.

Not tested so far (if in doubt, load the image with `cv::imread` and use `DetectImage(frame)`):

- JPEGs with an EXIF orientation tag (e.g. photos from a phone held upright). `cv::imread` rotates them by
  default before `Detect`; whether the server rotates them the same way has not been checked.
- CMYK JPEGs, 16-bit PNGs and PNGs with transparency.
- Grayscale input (`CV_8UC1`) to `DetectImage(frame)`.

### What LastInferMs measures

`LastInferMs()` times only the request to the server: sending, the server's work, and the answer coming back.
Work done on the PC before or after the request is not included.

```
Detect:       PC pre-processing ─▶ [ send tensor ─▶ model ─▶ answer ] ─▶ PC post-processing
DetectImage:  (PC JPEG encode) ─▶ [ send JPEG ─▶ DALI pre-processing ─▶ model ─▶ answer ] ─▶ PC post-processing
                                  └──────────── LastInferMs() ────────────┘
```

So for `DetectImage` the pre-processing **is** included (it runs on the server), for `Detect` it is not
(it runs on the PC before the request). Post-processing (NMS, mapping boxes back) always runs on the PC and is
never included. To time a whole call, measure it yourself:

```cpp
auto t0 = std::chrono::steady_clock::now();
auto dets = DetectImage(bytes);
double totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
```

The first request after the Jetson has been idle is slower (~50–60 ms instead of ~23–30 ms) while the GPU
clocks ramp up. Send one warm-up request after `ConnectImage`, or run `sudo jetson_clocks` on the Jetson to
keep the clocks at maximum (until the next reboot; uses more power).

## Server-side pre-processing

With `DetectImage` the PC sends only the encoded image. A Triton **ensemble** on the server
decodes and pre-processes it on the GPU with [NVIDIA DALI](https://github.com/NVIDIA/DALI)
and passes the tensor to the detection model:

```
PC                                   Triton server (Jetson)
DetectImage(frame)                   model1_image (ensemble)
  JPEG encode ── IMAGE (UINT8) ───▶    preprocess (DALI, GPU): decode, letterbox, RGB, /255, CHW
                                       model1 (YOLO, TensorRT)
  Postprocess ◀── output0 + LETTERBOX ─┘
  (NMS, boxes back to the image)
```

The server files are in [`server/model_repository/`](server/model_repository):

| Model | Files | What it does |
| --- | --- | --- |
| `preprocess` | `config.pbtxt`, `1/dali.py` ([code](#server-code)) | DALI pipeline: GPU decode, resize keeping the aspect ratio, centre on a 640×640 grey (114) canvas, RGB, 0..1, CHW. Outputs `images` `[3, 640, 640]` and `LETTERBOX`. Batches up to 8 requests. |
| `model1_image`, `model2_image`, `model3_image` | `config.pbtxt`, empty `1/` | Ensembles `preprocess` → `model1` / `model2` / `model3` |

The letterbox is the same as in `Detect`, so both functions give the same detections.
Measured on the Jetson with frames from `testing.mp4`:

| | `Detect` | `DetectImage` |
| --- | --- | --- |
| Data sent per 4K frame | ~4.9 MB (float tensor) | ~1 MB (JPEG, quality 90) |
| Server time per request | ~27 ms (model only) | ~35 ms at 1080p, ~44 ms at 4K (DALI + model) |
| Detections | reference | same objects, boxes within 1–3 px |

Sending less data matters most when the server is on Wi-Fi; DALI adds ~8 ms (1080p) to ~17 ms (4K) on the server.

Measured from a Windows PC on 41 particle images (1280×720 JPEG, `model1`), whole call incl. post-processing:

| Network | `Detect(img)` | `DetectImage(fileBytes)` | `DetectImage(img)` |
| --- | --- | --- | --- |
| Wired, `140.129.7.180` | ~70 ms | ~31 ms | ~34 ms |
| Wi-Fi, `192.168.1.156` | ~500–675 ms | ~108–145 ms | ~130–150 ms |

`DetectImage(fileBytes)` found all 38 detections of `Detect`, boxes within 1 px. `DetectImage(img)` re-encodes
an image that was already a JPEG and lost 2 borderline detections (score 0.257 → 0.243 with `conf = 0.25`);
send the file bytes, or use `jpegQuality = 100`, when that matters.

### Ensemble contract

`ConnectImage` checks this and reads the tensor names from the server, so only `LETTERBOX` is a fixed name:

| Tensor | Type / shape | Content |
| --- | --- | --- |
| input (any name, here `IMAGE`) | `UINT8` `[1, -1]` (or `[-1]`) | bytes of a JPEG/PNG file |
| detection output (any name, here `output0`) | `FP32`, a [supported layout](#supported-model-outputs) | the YOLO model's output, in letterboxed coordinates |
| `LETTERBOX` | `FP32` `[1, 5]` (or `[5]`) | `scale, pad_x, pad_y, width, height` of the letterbox the server applied |

### Server code

The complete server side is three small files. Model repository layout:

```
model_repository/
├─ model1/                 existing YOLO model (TensorRT), unchanged
├─ preprocess/
│  ├─ config.pbtxt
│  └─ 1/dali.py
└─ model1_image/
   ├─ config.pbtxt
   └─ 1/                   empty folder (required)
```

**`preprocess/1/dali.py`**: the DALI pipeline. Triton's DALI backend serializes it when the model loads.

```python
# DALI pre-processing for the YOLO models. Triton's DALI backend serializes this
# pipeline when the model loads (@autoserialize), so there is no model.dali file.
#
# Input : ENCODED   - bytes of a JPEG/PNG file, UINT8 [N]
# Output: images    - letterboxed RGB image, FP32 [3, 640, 640], 0..1, CHW
#         LETTERBOX - [scale, pad_x, pad_y, width, height], FP32 [5]; the client
#                     uses it to map boxes back to the original image
#
# Same letterbox as TritonVision Detect(): resize keeping the aspect ratio,
# centre on a 640x640 grey (114) canvas.
#
# Note: DALI cannot skip a bad sample. Bytes that are not an image make the
# pipeline fail and every later request fails too, until the model is reloaded.

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.triton import autoserialize

W, H = 640, 640


@autoserialize
@dali.pipeline_def(batch_size=8, num_threads=4, device_id=0)
def pipe():
    encoded = fn.external_source(device="cpu", name="ENCODED", dtype=types.UINT8, ndim=1)

    # Letterbox geometry, computed on the CPU from the image header.
    shape = fn.cast(fn.peek_image_shape(encoded), dtype=types.FLOAT)    # [h, w, c]
    h, w = shape[0], shape[1]
    scale = dali.math.min(W / w, H / h)
    nw = dali.math.floor(w * scale + 0.5)
    nh = dali.math.floor(h * scale + 0.5)
    px = dali.math.floor((W - nw) / 2)
    py = dali.math.floor((H - nh) / 2)

    # Decode on the GPU. affine=False: thread pinning needs NVML, which cannot
    # query the Jetson iGPU, and every decode then fails with "nvml error (6)".
    img = fn.decoders.image(encoded, device="mixed", output_type=types.RGB, affine=False)
    img = fn.resize(img, resize_x=nw, resize_y=nh, interp_type=types.INTERP_LINEAR, antialias=False)
    img = fn.slice(img, start=fn.cast(fn.stack(-py, -px), dtype=types.INT32), shape=[H, W],
                   axis_names="HW", out_of_bounds_policy="pad", fill_values=114)
    img = fn.crop_mirror_normalize(img, dtype=types.FLOAT, output_layout="CHW",
                                   mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])

    letterbox = fn.stack(scale, px, py, w, h)
    return img, letterbox
```

How each step matches `Detect` on the PC:

| `dali.py` | `Detect` in `src/TritonVision.cpp` |
| --- | --- |
| `peek_image_shape` → `scale`, `nw`, `nh`, `px`, `py` | `scale = min(W/cols, H/rows)`, `std::round`, padding `(W - nw) / 2` |
| `decoders.image(..., output_type=RGB)` | `cv::cvtColor(BGR2RGB)` |
| `resize(..., INTERP_LINEAR)` | `cv::resize` (bilinear) |
| `slice(..., out_of_bounds_policy="pad", fill_values=114)` | grey 114 canvas + `copyTo` |
| `crop_mirror_normalize(std=255, layout CHW)` | `convertTo(1/255)` + `cv::split` into planes |

**`preprocess/config.pbtxt`**

```protobuf
name: "preprocess"
backend: "dali"
max_batch_size: 8

input [
  {
    name: "ENCODED"
    data_type: TYPE_UINT8
    dims: [ -1 ]          # bytes of a JPEG/PNG file
  }
]

# Same order as the pipeline's return values in 1/dali.py.
output [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  },
  {
    name: "LETTERBOX"
    data_type: TYPE_FP32
    dims: [ 5 ]           # scale, pad_x, pad_y, width, height
  }
]

instance_group [{ kind: KIND_GPU }]
```

**`model1_image/config.pbtxt`**: the ensemble the client connects to. For `model2_image` / `model3_image`
only `name`, the comment and the second step's `model_name` change.

```protobuf
# Encoded image in, detections out: preprocess (DALI) -> model1.
# Used by TritonVision ConnectImage() / DetectImage().
name: "model1_image"
platform: "ensemble"
max_batch_size: 0         # model1 has no batching; the batch dim of 1 is explicit

input [
  {
    name: "IMAGE"
    data_type: TYPE_UINT8
    dims: [ 1, -1 ]       # bytes of a JPEG/PNG file
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 1, 6, 8400 ]
  },
  {
    name: "LETTERBOX"
    data_type: TYPE_FP32
    dims: [ 1, 5 ]        # scale, pad_x, pad_y, width, height
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocess"
      model_version: -1
      input_map { key: "ENCODED" value: "IMAGE" }
      output_map { key: "images" value: "preprocessed" }
      output_map { key: "LETTERBOX" value: "LETTERBOX" }
    },
    {
      model_name: "model1"
      model_version: -1
      input_map { key: "images" value: "preprocessed" }
      output_map { key: "output0" value: "output0" }
    }
  ]
}
```

The `model1` it chains to (existing, for reference): `max_batch_size: 0`, input `images` FP32
`[1, 3, 640, 640]`, output `output0` FP32 `[1, 6, 8400]` (4 box values + 2 classes, 8400 candidates).

### Checking the ensemble on the server

Before testing from the PC, send one image directly on the Jetson (needs `pip install tritonclient[grpc]`):

```python
import numpy as np
import tritonclient.grpc as g

c = g.InferenceServerClient("localhost:8001")
data = np.fromfile("test.jpg", dtype=np.uint8)[None]          # [1, N] bytes of a JPEG file
inp = g.InferInput("IMAGE", list(data.shape), "UINT8")
inp.set_data_from_numpy(data)

r = c.infer("model1_image", [inp])
print(r.as_numpy("output0").shape)      # (1, 6, 8400)
print(r.as_numpy("LETTERBOX"))          # e.g. [[0.5, 0, 140, 1280, 720]] for a 1280x720 image
```

Only send real image files here: a non-image breaks `preprocess` until it is reloaded.

### Complete PC example

Runs every JPEG/PNG in a folder through the server-side pre-processing and saves the drawn results:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include "TritonVision.h"

namespace fs = std::filesystem;

int main()
{
    const fs::path inputDir  = "D:\\images";
    const fs::path outputDir = "inferenceresult";

    if (!ConnectImage("140.129.7.180:8001", "model1_image")) { std::cerr << LastError() << "\n"; return 1; }
    fs::create_directories(outputDir);

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        // Read the file as it is: sent to the server without decoding or re-encoding on the PC
        std::ifstream f(entry.path(), std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        std::vector<Detection> dets = DetectImage(bytes);
        if (dets.empty() && !LastError().empty()) {
            std::cerr << entry.path().filename().string() << ": " << LastError() << "\n";
            continue;
        }

        // Decode on the PC only to draw the result
        cv::Mat img = cv::imdecode(bytes, cv::IMREAD_COLOR);
        Draw(img, dets);
        cv::imwrite((outputDir / entry.path().filename()).string(), img);

        std::cout << entry.path().filename().string() << " -> " << dets.size()
                  << " objects, " << LastInferMs() << " ms\n";
    }
    return 0;
}
```

### Installing on the server

Requires a Triton container that includes the DALI backend (`backend: "dali"`); the one on the
Jetson does. The pipeline is serialized from `dali.py` when the model loads, so there is no
`model.dali` file to build.

1. Create `preprocess/` and `model1_image/` … `model3_image/` in the Triton model repository
   (on the Jetson: `~/nvidiatriton/model_repository/`), next to `model1` … `model3`, with the
   files from [Server code](#server-code). Keep the empty `1/` folders: Triton needs a version
   folder for every model.
2. For a different detection model, copy one `model*_image` folder and change `name`, the
   `model_name` of the second step, and the dims of `output0`. The ensembles assume the
   model has `max_batch_size: 0`, input `images` `[1, 3, 640, 640]` and output `output0`.
3. For a model input other than 640×640, change `W, H` in `dali.py` and the `images` dims
   in `preprocess/config.pbtxt`.
4. Restart Triton and check that all models are `READY`:
   ```bash
   sudo docker restart triton
   ```
   ```bash
   curl -s -X POST localhost:8000/v2/repository/index
   ```

Then on the PC:

```cpp
if (!ConnectImage("140.129.7.180:8001", "model1_image"))     // once
    std::cerr << LastError() << "\n";

std::vector<Detection> dets = DetectImage(frame);             // per image (cv::Mat, BGR)
// or: DetectImage(bytesOfJpgFile);
```

### Things to know about DALI on the Jetson

- **`affine=False` is required.** By default the DALI GPU decoder asks NVML which CPU cores
  to pin its threads to. NVML cannot answer that for the Jetson's built-in GPU, and every
  decode fails with `nvml error (6)`. `dali.py` sets `affine=False` on the decoder.
- **A bad image breaks the model until it is reloaded.** DALI cannot skip a sample it fails
  to decode, so after one request that is not an image every later request fails too.
  `DetectImage` therefore rejects anything that is not a JPEG or PNG before sending it.
  Other clients sending to the server do not have that check. To recover, restart the
  container, or move `preprocess/` out of the repository and back.
- **Harmless log lines at load time:** `Could not create nvtiff decoder` (no GPU TIFF decoding
  on Jetson; only TIFF images are affected) and `AttributeError: 'NoneType' object has no
  attribute 'loader'` (followed by `DALI pipeline ... loaded successfully`).
- **`__pycache__`:** the container runs as root and creates a root-owned `preprocess/1/__pycache__/`.
  It is harmless; deleting it needs `sudo`.

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
| `ConnectImage` fails: `... is not a pre-processing ensemble` | Name of the plain model (e.g. `model1`) given instead of the ensemble | Use the ensemble name (`model1_image`) |
| `ConnectImage` fails with a model error / ensemble not READY | Ensemble names or dims don't match the detection model, or `preprocess` failed to load | See the Triton log; [install steps](#installing-on-the-server) 2–3 |
| Every `DetectImage` fails, although it worked before | A non-image request (e.g. from another client) broke the DALI `preprocess` model | Restart the container, or move `preprocess/` out of the repository and back ([details](#things-to-know-about-dali-on-the-jetson)) |
| `DetectImage` fails with `nvml error (6)` | `affine=False` missing on the DALI decoder | Keep `affine=False` in `dali.py` ([details](#things-to-know-about-dali-on-the-jetson)) |
| `DetectImage` fails: `not a JPEG or PNG image` | The bytes are not a JPEG/PNG image | Pass a valid image file's bytes, or use the `cv::Mat` overload |
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
