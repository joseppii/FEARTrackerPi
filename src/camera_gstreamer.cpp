#include "video_source.h"
#include <iostream>
#include <sstream>

// GStreamerSource implementation

struct GStreamerSource::Impl {
    cv::VideoCapture cap;
    bool opened = false;
    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0;
};

GStreamerSource::GStreamerSource(const VideoSourceConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
}

GStreamerSource::~GStreamerSource() {
    close();
}

bool GStreamerSource::open() {
    std::string pipeline;

    if (!custom_pipeline_.empty()) {
        // Use user-provided pipeline
        pipeline = custom_pipeline_;
    } else {
        // Build default libcamerasrc pipeline for Pi5 CSI camera
        // Format: libcamerasrc ! video/x-raw,width=W,height=H,framerate=F/1 ! videoconvert ! appsink
        std::ostringstream oss;

        // libcamerasrc with camera index
        oss << "libcamerasrc";
        if (!config_.camera_id.empty() && config_.camera_id != "0") {
            oss << " camera-name=" << config_.camera_id;
        }

        // Caps filter for resolution and framerate
        oss << " ! video/x-raw"
            << ",width=" << config_.width
            << ",height=" << config_.height
            << ",framerate=" << config_.framerate << "/1";

        // Convert to BGR for OpenCV
        oss << " ! videoconvert ! video/x-raw,format=BGR";

        // Output to appsink
        oss << " ! appsink drop=1";

        pipeline = oss.str();
    }

    std::cout << "GStreamer pipeline: " << pipeline << std::endl;

    // Open with GStreamer backend
    impl_->cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!impl_->cap.isOpened()) {
        std::cerr << "GStreamerSource: Failed to open pipeline" << std::endl;
        std::cerr << "Make sure gstreamer1.0-libcamera is installed" << std::endl;
        return false;
    }

    // Get actual properties (may differ from requested)
    impl_->actual_width = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_WIDTH));
    impl_->actual_height = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    impl_->actual_fps = impl_->cap.get(cv::CAP_PROP_FPS);

    // Fallback if properties not reported
    if (impl_->actual_width <= 0) impl_->actual_width = config_.width;
    if (impl_->actual_height <= 0) impl_->actual_height = config_.height;
    if (impl_->actual_fps <= 0) impl_->actual_fps = config_.framerate;

    impl_->opened = true;

    std::cout << "GStreamerSource: Opened camera at "
              << impl_->actual_width << "x" << impl_->actual_height
              << " @ " << impl_->actual_fps << " FPS" << std::endl;

    return true;
}

bool GStreamerSource::is_open() const {
    return impl_->opened && impl_->cap.isOpened();
}

void GStreamerSource::close() {
    if (impl_->cap.isOpened()) {
        impl_->cap.release();
    }
    impl_->opened = false;
}

bool GStreamerSource::read(cv::Mat& frame) {
    if (!is_open()) {
        return false;
    }
    return impl_->cap.read(frame) && !frame.empty();
}

int GStreamerSource::width() const {
    return impl_->actual_width;
}

int GStreamerSource::height() const {
    return impl_->actual_height;
}

double GStreamerSource::fps() const {
    return impl_->actual_fps;
}

std::string GStreamerSource::description() const {
    std::ostringstream oss;
    oss << "GStreamer Camera (camera_id=" << config_.camera_id << ")";
    return oss.str();
}

void GStreamerSource::set_pipeline(const std::string& pipeline) {
    custom_pipeline_ = pipeline;
}

bool GStreamerSource::is_available() {
    // Check if OpenCV was built with GStreamer support
    // Try to create a simple test pipeline
    cv::VideoCapture test;

    // Try opening a dummy pipeline - this will fail but tells us if GStreamer backend exists
    // A better approach: check build info
    std::string build_info = cv::getBuildInformation();
    return build_info.find("GStreamer") != std::string::npos &&
           build_info.find("GStreamer:                   YES") != std::string::npos;
}
