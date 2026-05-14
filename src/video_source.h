#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>

// Video source configuration (primarily for cameras)
struct VideoSourceConfig {
    int width = 1280;
    int height = 720;
    int framerate = 30;
    std::string camera_id = "0";
};

// RTP H265 source configuration. Mirrors the GStreamer pipeline knobs in
// the ACTrackerLoratPi reference implementation.
struct RtpH265Config {
    int port = 5004;            // udpsrc port
    int latency_ms = 50;        // rtpjitterbuffer latency
    int payload_type = 96;      // RTP dynamic payload type
    int clock_rate = 90000;     // standard for H.265 video over RTP
    // If non-empty, overrides the auto-built pipeline. Must terminate in an
    // appsink that emits BGR cv::Mat-compatible frames.
    std::string custom_pipeline;
};

// Abstract base class for all video sources
class IVideoSource {
public:
    virtual ~IVideoSource() = default;

    // Open the video source
    virtual bool open() = 0;

    // Check if source is open
    virtual bool is_open() const = 0;

    // Close the video source
    virtual void close() = 0;

    // Read next frame - returns false on failure/end
    virtual bool read(cv::Mat& frame) = 0;

    // Get video properties
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;

    // Get total frame count (-1 for live sources like cameras)
    virtual int64_t frame_count() const = 0;

    // Check if this is a live source (camera) vs file
    virtual bool is_live() const = 0;

    // Get source description for logging
    virtual std::string description() const = 0;
};

// Video file source using cv::VideoCapture
class VideoSourceFile : public IVideoSource {
public:
    explicit VideoSourceFile(const std::string& filename);
    ~VideoSourceFile() override;

    bool open() override;
    bool is_open() const override;
    void close() override;
    bool read(cv::Mat& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;
    int64_t frame_count() const override;
    bool is_live() const override { return false; }
    std::string description() const override;

private:
    std::string filename_;
    cv::VideoCapture cap_;
    int width_ = 0;
    int height_ = 0;
    double fps_ = 0;
    int64_t total_frames_ = 0;
};

// GStreamer camera source using libcamerasrc pipeline
class GStreamerSource : public IVideoSource {
public:
    explicit GStreamerSource(const VideoSourceConfig& config);
    ~GStreamerSource() override;

    bool open() override;
    bool is_open() const override;
    void close() override;
    bool read(cv::Mat& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;
    int64_t frame_count() const override { return -1; }
    bool is_live() const override { return true; }
    std::string description() const override;

    // Set custom pipeline string (for advanced users)
    void set_pipeline(const std::string& pipeline);

    // Check if GStreamer backend is available
    static bool is_available();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    VideoSourceConfig config_;
    std::string custom_pipeline_;
};

// V4L2 source for USB UVC cameras. Uses cv::VideoCapture with the
// CAP_V4L2 backend, so it handles both YUYV and MJPEG cameras (OpenCV
// transcodes MJPEG → BGR internally).
class V4L2Source : public IVideoSource {
public:
    explicit V4L2Source(const VideoSourceConfig& config);
    ~V4L2Source() override;

    V4L2Source(const V4L2Source&) = delete;
    V4L2Source& operator=(const V4L2Source&) = delete;

    bool open() override;
    bool is_open() const override;
    void close() override;
    bool read(cv::Mat& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;
    int64_t frame_count() const override { return -1; }
    bool is_live() const override { return true; }
    std::string description() const override;

    // True iff at least one /dev/video* device exists.
    static bool is_available();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    VideoSourceConfig config_;
};

#ifdef HAVE_LIBCAMERA
// Native libcamera source for best performance
class LibcameraSource : public IVideoSource {
public:
    explicit LibcameraSource(const VideoSourceConfig& config);
    ~LibcameraSource() override;

    // Non-copyable
    LibcameraSource(const LibcameraSource&) = delete;
    LibcameraSource& operator=(const LibcameraSource&) = delete;

    bool open() override;
    bool is_open() const override;
    void close() override;
    bool read(cv::Mat& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;
    int64_t frame_count() const override { return -1; }
    bool is_live() const override { return true; }
    std::string description() const override;

    // Check if libcamera is available
    static bool is_available();

    // Get list of available cameras
    static std::vector<std::string> list_cameras();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    VideoSourceConfig config_;
};
#endif

// RTP H265 network source. Hardware-decodes via decodebin (v4l2slh265dec
// on Pi 5) and emits BGR frames through OpenCV's GStreamer backend.
class RtpH265Source : public IVideoSource {
public:
    explicit RtpH265Source(const RtpH265Config& config);
    ~RtpH265Source() override;

    RtpH265Source(const RtpH265Source&) = delete;
    RtpH265Source& operator=(const RtpH265Source&) = delete;

    bool open() override;
    bool is_open() const override;
    void close() override;
    bool read(cv::Mat& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;
    int64_t frame_count() const override { return -1; }
    bool is_live() const override { return true; }
    std::string description() const override;

    // True iff OpenCV was built with GStreamer support.
    static bool is_available();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    RtpH265Config config_;
};

// Factory function to create appropriate video source
// backend: "libcamera", "gstreamer", "auto" (tries libcamera first, then gstreamer)
std::unique_ptr<IVideoSource> create_video_source(
    const std::string& input,
    bool is_camera,
    const std::string& backend,
    const VideoSourceConfig& config
);

// Factory for an RTP H265 source. Returns nullptr if GStreamer is unavailable
// or the pipeline fails to start.
std::unique_ptr<IVideoSource> create_rtp_video_source(const RtpH265Config& config);

// List available cameras (uses available backend)
std::vector<std::string> list_available_cameras();
