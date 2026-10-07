// TritonVision.cpp
//
// Implementation of the TritonVision API (see TritonVision.h).
// Pipeline per image: letterbox pre-processing -> Triton gRPC inference ->
// YOLO-style post-processing (confidence filter, NMS, box mapping).

#include "TritonVision.h"

#include <chrono>
#include <memory>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include "grpc_client.h"

namespace tc = triton::client;

// ===== Library state (visible only inside this file) =====
static std::unique_ptr<tc::InferenceServerGrpcClient> g_client;   // gRPC connection
static std::string g_model, g_in, g_out, g_error;                 // model name, input/output tensor names, last error
static int    g_W = 640, g_H = 640;                               // model input size, read from the model metadata
static double g_infer_ms = 0;                                     // duration of the last inference request

// Class names, indexed by class id, in the order the model was trained with.
// Left empty on purpose: labels are then the class id as text ("0", "1", ...),
// and the calling application maps class_id to its own names.
static const std::vector<std::string> kClassNames = {};

// ===== State of ConnectImage() / DetectImage() (separate from Connect()) =====
static std::unique_ptr<tc::InferenceServerGrpcClient> g_img_client;
static std::string g_img_model, g_img_in, g_img_out;    // ensemble name, input/detection tensor names
static bool g_img_batched = false;                       // input shape [1, -1] (max_batch_size > 0) instead of [-1]
static const char* const kLetterbox = "LETTERBOX";       // ensemble output: scale, pad_x, pad_y, width, height

// Turns the raw model output (letterboxed model coordinates) into detections in
// the original image (cols x rows). scale/px/py describe the letterbox that was applied.
static std::vector<Detection> Postprocess(const float* d, const std::vector<int64_t>& shape,
    float scale, int px, int py, int cols, int rows, float conf, float nms)
{
    // Maps a box from model (letterboxed) coordinates back to the original frame.
    auto toFrame = [&](float x1, float y1, float x2, float y2) {
        x1 = std::clamp((x1 - px) / scale, 0.f, (float)cols - 1);
        y1 = std::clamp((y1 - py) / scale, 0.f, (float)rows - 1);
        x2 = std::clamp((x2 - px) / scale, 0.f, (float)cols - 1);
        y2 = std::clamp((y2 - py) / scale, 0.f, (float)rows - 1);
        return cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        };
    // Class name for an id, or the id as text when there is no name for it.
    auto label = [](int id) { return id >= 0 && id < (int)kClassNames.size() ? kClassNames[id] : std::to_string(id); };

    std::vector<Detection> dets;

    // Output format A, end-to-end models: [1, N, 6] = x1, y1, x2, y2, score, class.
    // NMS is already done by the model.
    if (shape.size() == 3 && shape[2] == 6) {
        for (int i = 0; i < (int)shape[1]; ++i) {
            const float* r = d + i * 6;
            if (r[4] < conf) continue;
            dets.push_back({ toFrame(r[0], r[1], r[2], r[3]), r[4], (int)r[5], label((int)r[5]) });
        }
        return dets;
    }

    // Output format B, one-to-many models: [1, 4 + nc, N] = cx, cy, w, h, then one
    // score per class, stored column-wise (value k of candidate i is d[k * n + i]).
    const int nc = (int)shape[1] - 4, n = (int)shape[2];
    std::vector<cv::Rect> boxes, nms_boxes;
    std::vector<float> scores;
    std::vector<int> ids;
    for (int i = 0; i < n; ++i) {
        int best = 0; float s = 0.f;                                 // best class for this candidate
        for (int c = 0; c < nc; ++c)
            if (d[(4 + c) * n + i] > s) { s = d[(4 + c) * n + i]; best = c; }
        if (s < conf) continue;

        float cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
        cv::Rect b = toFrame(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
        boxes.push_back(b);
        // Shift each class far apart so NMS only suppresses boxes of the same class.
        nms_boxes.push_back(b + cv::Point(best * 7680, 0));
        scores.push_back(s);
        ids.push_back(best);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(nms_boxes, scores, conf, nms, keep);
    for (int k : keep) dets.push_back({ boxes[k], scores[k], ids[k], label(ids[k]) });
    return dets;
}

bool Connect(const std::string& url, const std::string& model)
{
    tc::Error err = tc::InferenceServerGrpcClient::Create(&g_client, url, false);
    if (!err.IsOk()) { g_error = err.Message(); return false; }

    // The metadata tells us the input/output tensor names and the input size.
    inference::ModelMetadataResponse md;
    err = g_client->ModelMetadata(&md, model);
    if (!err.IsOk()) { g_error = err.Message(); g_client.reset(); return false; }

    g_model = model;
    g_in = md.inputs(0).name();
    g_out = md.outputs(0).name();
    if (md.inputs(0).shape_size() == 4) {                             // NCHW: [1, 3, H, W]
        if (md.inputs(0).shape(2) > 0) g_H = (int)md.inputs(0).shape(2);
        if (md.inputs(0).shape(3) > 0) g_W = (int)md.inputs(0).shape(3);
    }
    return true;
}

std::vector<Detection> Detect(const cv::Mat& frame, float conf, float nms)
{
    g_error.clear();
    if (!g_client) { g_error = "not connected: call Connect() first"; return {}; }
    if (frame.empty()) { g_error = "empty frame"; return {}; }

    // ----- 1. Pre-process: letterbox + BGR->RGB + scale to 0..1 + HWC->CHW -----
    // Resize while keeping the aspect ratio, then pad to the model size with grey (114).
    float scale = std::min(g_W / (float)frame.cols, g_H / (float)frame.rows);
    int nw = (int)std::round(frame.cols * scale), nh = (int)std::round(frame.rows * scale);
    int px = (g_W - nw) / 2, py = (g_H - nh) / 2;                    // padding left / top

    cv::Mat resized, canvas(g_H, g_W, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, resized, cv::Size(nw, nh));
    resized.copyTo(canvas(cv::Rect(px, py, nw, nh)));

    cv::Mat f32;
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    // Split the interleaved channels straight into one planar float buffer (CHW).
    std::vector<float> blob(3 * g_W * g_H);
    std::vector<cv::Mat> ch;
    for (int c = 0; c < 3; ++c) ch.emplace_back(g_H, g_W, CV_32F, blob.data() + c * g_W * g_H);
    cv::split(f32, ch);

    // ----- 2. Inference request -----
    tc::InferInput* in_raw;
    tc::InferInput::Create(&in_raw, g_in, { 1, 3, g_H, g_W }, "FP32");
    std::unique_ptr<tc::InferInput> input(in_raw);
    input->AppendRaw(reinterpret_cast<const uint8_t*>(blob.data()), blob.size() * sizeof(float));

    tc::InferRequestedOutput* out_raw;
    tc::InferRequestedOutput::Create(&out_raw, g_out);
    std::unique_ptr<tc::InferRequestedOutput> output(out_raw);

    auto t0 = std::chrono::steady_clock::now();
    tc::InferResult* res_raw;
    tc::Error err = g_client->Infer(&res_raw, tc::InferOptions(g_model), { input.get() }, { output.get() });
    g_infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!err.IsOk()) { g_error = err.Message(); return {}; }

    std::unique_ptr<tc::InferResult> res(res_raw);
    std::vector<int64_t> shape;
    const uint8_t* buf; size_t size;
    res->Shape(g_out, &shape);
    res->RawData(g_out, &buf, &size);
    const float* d = reinterpret_cast<const float*>(buf);

    // ----- 3. Post-process -----
    return Postprocess(d, shape, scale, px, py, frame.cols, frame.rows, conf, nms);
}

bool ConnectImage(const std::string& url, const std::string& ensemble)
{
    tc::Error err = tc::InferenceServerGrpcClient::Create(&g_img_client, url, false);
    if (!err.IsOk()) { g_error = err.Message(); return false; }

    inference::ModelMetadataResponse md;
    err = g_img_client->ModelMetadata(&md, ensemble);
    if (!err.IsOk()) { g_error = err.Message(); g_img_client.reset(); return false; }

    // Expect one input (encoded image) and two outputs: detections + LETTERBOX.
    g_img_out.clear();
    for (const auto& o : md.outputs())
        if (o.name() != kLetterbox) g_img_out = o.name();
    if (md.inputs_size() != 1 || md.outputs_size() != 2 || g_img_out.empty()) {
        g_error = ensemble + " is not a pre-processing ensemble (needs 1 input and outputs <detections> + LETTERBOX)";
        g_img_client.reset(); return false;
    }

    g_img_model = ensemble;
    g_img_in = md.inputs(0).name();
    g_img_batched = md.inputs(0).shape_size() == 2;
    return true;
}

std::vector<Detection> DetectImage(const std::vector<uint8_t>& encoded, float conf, float nms)
{
    g_error.clear();
    if (!g_img_client) { g_error = "not connected: call ConnectImage() first"; return {}; }
    if (encoded.empty()) { g_error = "empty image"; return {}; }

    // ----- 1. Inference request: the encoded bytes, pre-processed on the server -----
    std::vector<int64_t> in_shape = { (int64_t)encoded.size() };
    if (g_img_batched) in_shape.insert(in_shape.begin(), 1);

    tc::InferInput* in_raw;
    tc::InferInput::Create(&in_raw, g_img_in, in_shape, "UINT8");
    std::unique_ptr<tc::InferInput> input(in_raw);
    input->AppendRaw(encoded.data(), encoded.size());

    tc::InferRequestedOutput *out_raw, *lb_raw;
    tc::InferRequestedOutput::Create(&out_raw, g_img_out);
    tc::InferRequestedOutput::Create(&lb_raw, kLetterbox);
    std::unique_ptr<tc::InferRequestedOutput> output(out_raw), letterbox(lb_raw);

    auto t0 = std::chrono::steady_clock::now();
    tc::InferResult* res_raw;
    tc::Error err = g_img_client->Infer(&res_raw, tc::InferOptions(g_img_model), { input.get() }, { output.get(), letterbox.get() });
    g_infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!err.IsOk()) { g_error = err.Message(); return {}; }

    std::unique_ptr<tc::InferResult> res(res_raw);
    if (!res->RequestStatus().IsOk()) { g_error = res->RequestStatus().Message(); return {}; }

    std::vector<int64_t> shape;
    const uint8_t* buf; size_t size;
    res->Shape(g_img_out, &shape);
    res->RawData(g_img_out, &buf, &size);
    const float* d = reinterpret_cast<const float*>(buf);

    // How the server letterboxed the image, needed to map boxes back.
    const uint8_t* lb_buf; size_t lb_size;
    res->RawData(kLetterbox, &lb_buf, &lb_size);
    if (lb_size < 5 * sizeof(float)) { g_error = "invalid LETTERBOX output"; return {}; }
    const float* lb = reinterpret_cast<const float*>(lb_buf);    // scale, pad_x, pad_y, width, height

    // ----- 2. Post-process -----
    return Postprocess(d, shape, lb[0], (int)lb[1], (int)lb[2], (int)lb[3], (int)lb[4], conf, nms);
}

std::vector<Detection> DetectImage(const cv::Mat& frame, float conf, float nms, int jpegQuality)
{
    if (frame.empty()) { g_error = "empty frame"; return {}; }
    std::vector<uint8_t> jpg;
    if (!cv::imencode(".jpg", frame, jpg, { cv::IMWRITE_JPEG_QUALITY, jpegQuality })) {
        g_error = "JPEG encoding failed"; return {};
    }
    return DetectImage(jpg, conf, nms);
}

void Draw(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        // A stable colour per class id.
        cv::Scalar col((d.class_id * 67) % 255, (d.class_id * 131) % 255, (d.class_id * 199) % 255);
        cv::rectangle(frame, d.box, col, 2);
        cv::putText(frame, d.label + " " + cv::format("%.2f", d.score),
            cv::Point(d.box.x, std::max(d.box.y - 5, 15)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 2);
    }
}

std::string LastError() { return g_error; }
double      LastInferMs() { return g_infer_ms; }
