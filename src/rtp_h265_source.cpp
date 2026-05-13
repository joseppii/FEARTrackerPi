#include "video_source.h"
#include <iostream>
#include <sstream>

// RtpH265Source implementation.
//
// Ported from ACTrackerLoratPi's GstHevcSource (raw GStreamer C API)
// to the IVideoSource pattern used in this project. We run the pipeline
// through OpenCV's cv::CAP_GSTREAMER backend instead of the raw appsink
// API, since cv::VideoCapture already handles GstSample mapping for us.
//
// Hardware decode: decodebin auto-plugs v4l2slh265dec on Raspberry Pi OS
// Bookworm; we don't get a runtime confirmation log (the raw GStreamer
// path did), but the HW decode path still runs when available.

struct RtpH265Source::Impl {
    cv::VideoCapture cap;
    bool opened = false;
    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0;
    std::string pipeline_used;
};

namespace {

std::string build_rtp_pipeline(const RtpH265Config& cfg) {
    std::ostringstream oss;
    oss << "udpsrc port=" << cfg.port
        << " caps=\"application/x-rtp"
        << ",media=video"
        << ",clock-rate=" << cfg.clock_rate
        << ",encoding-name=H265"
        << ",payload=" << cfg.payload_type
        << "\""
        << " ! rtpjitterbuffer latency=" << cfg.latency_ms
        << " ! rtph265depay"
        << " ! h265parse"
        << " ! decodebin"
        << " ! videoconvert"
        << " ! video/x-raw,format=BGR"
        << " ! appsink drop=1 sync=false max-buffers=2";
    return oss.str();
}

}  // namespace

RtpH265Source::RtpH265Source(const RtpH265Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
}

RtpH265Source::~RtpH265Source() {
    close();
}

bool RtpH265Source::open() {
    impl_->pipeline_used = config_.custom_pipeline.empty()
                               ? build_rtp_pipeline(config_)
                               : config_.custom_pipeline;

    std::cout << "RTP H265 pipeline: " << impl_->pipeline_used << std::endl;

    impl_->cap.open(impl_->pipeline_used, cv::CAP_GSTREAMER);

    if (!impl_->cap.isOpened()) {
        std::cerr << "RtpH265Source: failed to open pipeline (port "
                  << config_.port << ")" << std::endl;
        std::cerr << "Check that gstreamer1.0-plugins-{base,good,bad,libav} "
                     "are installed and the sender is streaming H265/RTP."
                  << std::endl;
        return false;
    }

    impl_->actual_width  = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_WIDTH));
    impl_->actual_height = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    impl_->actual_fps    = impl_->cap.get(cv::CAP_PROP_FPS);

    // Dimensions aren't known until the first SPS arrives; that's OK,
    // the first read() will populate them via OpenCV.
    impl_->opened = true;

    std::cout << "RtpH265Source: listening on UDP port " << config_.port
              << " (payload=" << config_.payload_type
              << ", latency=" << config_.latency_ms << "ms)" << std::endl;

    return true;
}

bool RtpH265Source::is_open() const {
    return impl_->opened && impl_->cap.isOpened();
}

void RtpH265Source::close() {
    if (impl_->cap.isOpened()) {
        impl_->cap.release();
    }
    impl_->opened = false;
}

bool RtpH265Source::read(cv::Mat& frame) {
    if (!is_open()) {
        return false;
    }
    if (!impl_->cap.read(frame) || frame.empty()) {
        return false;
    }
    if (impl_->actual_width == 0)  impl_->actual_width  = frame.cols;
    if (impl_->actual_height == 0) impl_->actual_height = frame.rows;
    return true;
}

int RtpH265Source::width() const {
    return impl_->actual_width;
}

int RtpH265Source::height() const {
    return impl_->actual_height;
}

double RtpH265Source::fps() const {
    return impl_->actual_fps;
}

std::string RtpH265Source::description() const {
    std::ostringstream oss;
    oss << "RTP H265 (udp:" << config_.port
        << " pt=" << config_.payload_type
        << " latency=" << config_.latency_ms << "ms)";
    return oss.str();
}

bool RtpH265Source::is_available() {
    const std::string info = cv::getBuildInformation();
    return info.find("GStreamer:                   YES") != std::string::npos;
}
