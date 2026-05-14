#include "video_source.h"
#include <iostream>
#include <sstream>
#include <sys/stat.h>

// V4L2 source: USB UVC cameras (and other plain V4L2 devices). Uses
// OpenCV's cv::CAP_V4L2 backend, which handles raw (YUYV) and MJPEG
// cameras transparently — for MJPEG cameras OpenCV inserts a JPEG
// decode step on its own.
//
// camera_id semantics: "/dev/videoN" used as-is; bare digit "N" is
// expanded to "/dev/videoN".

struct V4L2Source::Impl {
    cv::VideoCapture cap;
    bool opened = false;
    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0;
    std::string device_path;
};

namespace {
std::string resolve_device(const std::string& camera_id) {
    if (camera_id.empty()) return "/dev/video0";
    if (camera_id.front() == '/') return camera_id;
    return "/dev/video" + camera_id;
}
}

V4L2Source::V4L2Source(const VideoSourceConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    impl_->device_path = resolve_device(config_.camera_id);
}

V4L2Source::~V4L2Source() {
    close();
}

bool V4L2Source::open() {
    struct stat st{};
    if (stat(impl_->device_path.c_str(), &st) != 0) {
        std::cerr << "V4L2Source: device not found: " << impl_->device_path << std::endl;
        return false;
    }

    if (!impl_->cap.open(impl_->device_path, cv::CAP_V4L2)) {
        std::cerr << "V4L2Source: cv::VideoCapture failed to open "
                  << impl_->device_path << std::endl;
        return false;
    }

    // Many UVC cameras default to YUYV at small sizes. Requesting MJPEG
    // first lets us get higher resolutions / framerates from cameras
    // that only expose those modes over MJPEG.
    impl_->cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    impl_->cap.set(cv::CAP_PROP_FRAME_WIDTH,  config_.width);
    impl_->cap.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height);
    impl_->cap.set(cv::CAP_PROP_FPS,          config_.framerate);

    impl_->actual_width  = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_WIDTH));
    impl_->actual_height = static_cast<int>(impl_->cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    impl_->actual_fps    = impl_->cap.get(cv::CAP_PROP_FPS);

    if (impl_->actual_width <= 0)  impl_->actual_width  = config_.width;
    if (impl_->actual_height <= 0) impl_->actual_height = config_.height;
    if (impl_->actual_fps <= 0)    impl_->actual_fps    = config_.framerate;

    impl_->opened = true;

    std::cout << "V4L2Source: opened " << impl_->device_path
              << " at " << impl_->actual_width << "x" << impl_->actual_height
              << " @ " << impl_->actual_fps << " FPS" << std::endl;

    return true;
}

bool V4L2Source::is_open() const {
    return impl_->opened && impl_->cap.isOpened();
}

void V4L2Source::close() {
    if (impl_->cap.isOpened()) {
        impl_->cap.release();
    }
    impl_->opened = false;
}

bool V4L2Source::read(cv::Mat& frame) {
    if (!is_open()) return false;
    return impl_->cap.read(frame) && !frame.empty();
}

int V4L2Source::width() const  { return impl_->actual_width; }
int V4L2Source::height() const { return impl_->actual_height; }
double V4L2Source::fps() const { return impl_->actual_fps; }

std::string V4L2Source::description() const {
    std::ostringstream oss;
    oss << "V4L2 Camera (" << impl_->device_path << ")";
    return oss.str();
}

bool V4L2Source::is_available() {
    // Cheap check: does /dev/video0 (or any /dev/video*) exist?
    for (int i = 0; i < 64; ++i) {
        struct stat st{};
        const std::string p = "/dev/video" + std::to_string(i);
        if (stat(p.c_str(), &st) == 0) return true;
    }
    return false;
}
