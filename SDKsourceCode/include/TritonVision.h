// TritonVision.h
//
// Public API of TritonVision: object detection through an NVIDIA Triton
// Inference Server over gRPC.
//
// Typical use:
//
//     if (!Connect("192.168.1.156:8001", "model1"))      // once
//         std::cerr << LastError();
//
//     std::vector<Detection> dets = Detect(frame);        // per image
//     if (dets.empty() && !LastError().empty())
//         std::cerr << LastError();                       // the call failed
//     Draw(frame, dets);                                  // boxes + labels
//
// Notes:
// - The library keeps one global connection: one server and one model at a
//   time. Call it from one thread at a time.
// - std::string, std::vector and cv::Mat cross the DLL boundary, so the
//   calling project must use the same C++ runtime as the DLL: x64, /MD
//   (Multi-threaded DLL), _DEBUG not defined.

#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

// TRITONVISION_EXPORTS is defined only by the DLL project (TritonVision.vcxproj).
// Projects that use the DLL do not define it, so the functions are imported.
#ifdef TRITONVISION_EXPORTS
#define TV_API __declspec(dllexport)
#else
#define TV_API __declspec(dllimport)
#endif

// One detected object.
struct Detection {
    cv::Rect    box;       // x, y, width, height in pixels of the image passed to Detect()
    float       score;     // confidence, 0.0 - 1.0
    int         class_id;  // class index returned by the model
    std::string label;     // class name, or the class index as text when no names are set
};

// Connects to the Triton server and reads the model's input/output metadata.
//   url   - "IP:port" of the server's gRPC endpoint, e.g. "192.168.1.156:8001"
//           (8001 is Triton's default gRPC port; the HTTP port 8000 does not work)
//   model - model name exactly as it appears in the Triton model repository
// Returns true when the server answered and the model is ready.
// On failure returns false; LastError() holds the reason.
// Calling it again replaces the previous connection.
TV_API bool Connect(const std::string& url, const std::string& model);

// Runs object detection on one image.
//   frame - BGR image (CV_8UC3) of any size; resizing, padding and colour
//           conversion are done internally
//   conf  - minimum confidence to keep a detection
//   nms   - IoU threshold for non-maximum suppression (removes duplicate boxes)
// Returns the detections, with boxes mapped back to the original image.
// An empty result means either "no objects" or "error": check LastError()
// (empty string = success). Blocks until the server answers.
TV_API std::vector<Detection> Detect(const cv::Mat& frame, float conf = 0.25f, float nms = 0.45f);

// Draws each detection's box and "label score" text onto the image.
// The image is modified in place; draw on a clone to keep the original.
TV_API void Draw(cv::Mat& frame, const std::vector<Detection>& dets);

// Error message of the last Connect(), Detect(), ConnectImage() or DetectImage()
// call; empty when it succeeded.
TV_API std::string LastError();

// Duration of the last inference request (network round trip + server time),
// in milliseconds. Pre- and post-processing are not included.
TV_API double LastInferMs();

// ===== Server-side pre-processing (send the image only) =====
//
//     if (!ConnectImage("192.168.1.156:8001", "model1_image"))   // once
//         std::cerr << LastError();
//     std::vector<Detection> dets = DetectImage(frame);           // per image
//
// The image is sent as JPEG/PNG bytes; decoding, letterbox, colour conversion
// and scaling run on the server (Triton ensemble "preprocess" -> YOLO model).
// Post-processing (confidence filter, NMS, box mapping) still runs here.
// Independent of Connect()/Detect(): both can be used side by side.

// Connects to the Triton server and reads the ensemble's metadata.
//   url      - "IP:port" of the server's gRPC endpoint, as in Connect()
//   ensemble - ensemble model name, e.g. "model1_image". It must take one
//              UINT8 input (the encoded image) and return the detection
//              tensor plus "LETTERBOX" = [scale, pad_x, pad_y, width, height].
// Returns true when the server answered and the ensemble is ready.
// On failure returns false; LastError() holds the reason.
TV_API bool ConnectImage(const std::string& url, const std::string& ensemble);

// Runs object detection on one image, pre-processed on the server.
//   frame       - BGR image (CV_8UC3) of any size; it is JPEG-encoded before sending
//   jpegQuality - JPEG quality 0 - 100 (higher = bigger request, closer to the original)
// Results and errors as in Detect(). LastInferMs() includes server pre-processing.
TV_API std::vector<Detection> DetectImage(const cv::Mat& frame, float conf = 0.25f, float nms = 0.45f, int jpegQuality = 90);

// Same, for an image that is already encoded (bytes of a .jpg/.png file):
// sent as is, with no decoding or re-encoding on this side.
TV_API std::vector<Detection> DetectImage(const std::vector<uint8_t>& encoded, float conf = 0.25f, float nms = 0.45f);
