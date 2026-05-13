#ifdef HAVE_LIBCAMERA

#include "video_source.h"
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <cstring>
#include <sys/mman.h>
#include <functional>

#include <libcamera/libcamera.h>

using namespace libcamera;

// LibcameraSource implementation

// Callback handler class - defined before Impl so it can be used
class LibcameraCallbackHandler {
public:
    using CallbackFunc = std::function<void(Request*)>;

    void setCallback(CallbackFunc func) {
        callback_ = std::move(func);
    }

    void onRequestComplete(Request* request) {
        if (callback_) {
            callback_(request);
        }
    }

private:
    CallbackFunc callback_;
};

struct LibcameraSource::Impl {
    std::unique_ptr<CameraManager> cm;
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraConfiguration> config;
    std::unique_ptr<FrameBufferAllocator> allocator;
    std::vector<std::unique_ptr<Request>> requests;

    // Frame queue for async->sync conversion
    std::queue<cv::Mat> frame_queue;
    std::mutex queue_mutex;
    std::condition_variable frame_ready;
    static constexpr size_t MAX_QUEUE_SIZE = 3;

    // State
    std::atomic<bool> running{false};
    int actual_width = 0;
    int actual_height = 0;
    double actual_fps = 0;
    PixelFormat pixel_format;

    // Buffer mapping
    std::map<FrameBuffer*, std::vector<uint8_t>> mapped_buffers;

    // Callback handler
    LibcameraCallbackHandler callback_handler;

    void requestComplete(Request* request);
    bool convertToMat(FrameBuffer* buffer, cv::Mat& output);
};

LibcameraSource::LibcameraSource(const VideoSourceConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
}

LibcameraSource::~LibcameraSource() {
    close();
}

bool LibcameraSource::open() {
    // Create camera manager
    impl_->cm = std::make_unique<CameraManager>();
    int ret = impl_->cm->start();
    if (ret < 0) {
        std::cerr << "LibcameraSource: Failed to start camera manager" << std::endl;
        return false;
    }

    // Get camera list
    auto cameras = impl_->cm->cameras();
    if (cameras.empty()) {
        std::cerr << "LibcameraSource: No cameras found" << std::endl;
        return false;
    }

    // Select camera by ID or index
    int camera_index = 0;
    try {
        camera_index = std::stoi(config_.camera_id);
    } catch (...) {
        // Try to find by name
        for (size_t i = 0; i < cameras.size(); i++) {
            if (cameras[i]->id().find(config_.camera_id) != std::string::npos) {
                camera_index = static_cast<int>(i);
                break;
            }
        }
    }

    if (camera_index < 0 || camera_index >= static_cast<int>(cameras.size())) {
        std::cerr << "LibcameraSource: Camera index out of range" << std::endl;
        return false;
    }

    impl_->camera = cameras[camera_index];
    std::cout << "LibcameraSource: Using camera: " << impl_->camera->id() << std::endl;

    // Acquire camera
    ret = impl_->camera->acquire();
    if (ret < 0) {
        std::cerr << "LibcameraSource: Failed to acquire camera" << std::endl;
        return false;
    }

    // Configure camera
    impl_->config = impl_->camera->generateConfiguration({StreamRole::VideoRecording});
    if (!impl_->config || impl_->config->empty()) {
        std::cerr << "LibcameraSource: Failed to generate configuration" << std::endl;
        impl_->camera->release();
        return false;
    }

    StreamConfiguration& cfg = impl_->config->at(0);

    // Set requested resolution
    cfg.size.width = config_.width;
    cfg.size.height = config_.height;

    // Try to use a format that's easy to convert to BGR
    // Prefer YUV420 (NV12) or RGB formats
    cfg.pixelFormat = formats::YUV420;

    // Validate configuration
    CameraConfiguration::Status status = impl_->config->validate();
    if (status == CameraConfiguration::Invalid) {
        std::cerr << "LibcameraSource: Invalid camera configuration" << std::endl;
        impl_->camera->release();
        return false;
    }
    if (status == CameraConfiguration::Adjusted) {
        std::cout << "LibcameraSource: Configuration adjusted to: "
                  << cfg.size.width << "x" << cfg.size.height
                  << " format=" << cfg.pixelFormat.toString() << std::endl;
    }

    // Apply configuration
    ret = impl_->camera->configure(impl_->config.get());
    if (ret < 0) {
        std::cerr << "LibcameraSource: Failed to configure camera" << std::endl;
        impl_->camera->release();
        return false;
    }

    impl_->actual_width = cfg.size.width;
    impl_->actual_height = cfg.size.height;
    impl_->pixel_format = cfg.pixelFormat;
    impl_->actual_fps = config_.framerate;  // libcamera doesn't always report fps

    // Allocate buffers
    impl_->allocator = std::make_unique<FrameBufferAllocator>(impl_->camera);
    ret = impl_->allocator->allocate(cfg.stream());
    if (ret < 0) {
        std::cerr << "LibcameraSource: Failed to allocate buffers" << std::endl;
        impl_->camera->release();
        return false;
    }

    const std::vector<std::unique_ptr<FrameBuffer>>& buffers =
        impl_->allocator->buffers(cfg.stream());

    std::cout << "LibcameraSource: Allocated " << buffers.size() << " buffers" << std::endl;

    // Create requests and map buffers
    for (const auto& buffer : buffers) {
        // Map buffer memory
        size_t buffer_size = 0;
        for (const auto& plane : buffer->planes()) {
            buffer_size += plane.length;
        }
        impl_->mapped_buffers[buffer.get()].resize(buffer_size);

        // Create request
        auto request = impl_->camera->createRequest();
        if (!request) {
            std::cerr << "LibcameraSource: Failed to create request" << std::endl;
            impl_->camera->release();
            return false;
        }

        ret = request->addBuffer(cfg.stream(), buffer.get());
        if (ret < 0) {
            std::cerr << "LibcameraSource: Failed to add buffer to request" << std::endl;
            impl_->camera->release();
            return false;
        }

        impl_->requests.push_back(std::move(request));
    }

    // Set up callback and connect to signal
    impl_->callback_handler.setCallback([this](Request* request) {
        impl_->requestComplete(request);
    });
    impl_->camera->requestCompleted.connect(&impl_->callback_handler,
                                            &LibcameraCallbackHandler::onRequestComplete);

    // Start camera
    ret = impl_->camera->start();
    if (ret < 0) {
        std::cerr << "LibcameraSource: Failed to start camera" << std::endl;
        impl_->camera->release();
        return false;
    }

    impl_->running = true;

    // Queue all requests
    for (auto& request : impl_->requests) {
        impl_->camera->queueRequest(request.get());
    }

    std::cout << "LibcameraSource: Camera started at "
              << impl_->actual_width << "x" << impl_->actual_height
              << " format=" << impl_->pixel_format.toString() << std::endl;

    return true;
}

void LibcameraSource::Impl::requestComplete(Request* request) {
    if (!running || request->status() == Request::RequestCancelled) {
        return;
    }

    // Get the buffer
    const Request::BufferMap& buffers = request->buffers();
    for (auto& [stream, buffer] : buffers) {
        cv::Mat frame;
        if (convertToMat(buffer, frame)) {
            // Add to queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (frame_queue.size() >= MAX_QUEUE_SIZE) {
                    frame_queue.pop();  // Drop oldest frame
                }
                frame_queue.push(frame.clone());
            }
            frame_ready.notify_one();
        }
    }

    // Requeue the request
    request->reuse(Request::ReuseBuffers);
    if (running) {
        camera->queueRequest(request);
    }
}

bool LibcameraSource::Impl::convertToMat(FrameBuffer* buffer, cv::Mat& output) {
    // Map buffer planes using mmap
    const auto& planes = buffer->planes();
    if (planes.empty()) {
        return false;
    }

    // Read data from DMA buffer
    // Note: This is a simplified version. For production, use proper mmap.
    const FrameBuffer::Plane& plane = planes[0];

    // Memory map the file descriptor
    void* data = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED,
                      plane.fd.get(), plane.offset);
    if (data == MAP_FAILED) {
        return false;
    }

    // Convert based on pixel format
    if (pixel_format == formats::YUV420 || pixel_format == formats::NV12) {
        // YUV420/NV12 to BGR conversion
        cv::Mat yuv(actual_height * 3 / 2, actual_width, CV_8UC1, data);
        cv::cvtColor(yuv, output, cv::COLOR_YUV2BGR_NV12);
    } else if (pixel_format == formats::RGB888) {
        cv::Mat rgb(actual_height, actual_width, CV_8UC3, data);
        cv::cvtColor(rgb, output, cv::COLOR_RGB2BGR);
    } else if (pixel_format == formats::BGR888) {
        cv::Mat bgr(actual_height, actual_width, CV_8UC3, data);
        output = bgr.clone();
    } else if (pixel_format == formats::YUYV) {
        cv::Mat yuyv(actual_height, actual_width, CV_8UC2, data);
        cv::cvtColor(yuyv, output, cv::COLOR_YUV2BGR_YUYV);
    } else {
        // Unsupported format
        munmap(data, plane.length);
        return false;
    }

    munmap(data, plane.length);
    return true;
}

bool LibcameraSource::is_open() const {
    return impl_->running;
}

void LibcameraSource::close() {
    impl_->running = false;

    // Notify any waiting reads
    impl_->frame_ready.notify_all();

    if (impl_->camera) {
        impl_->camera->stop();
        impl_->camera->requestCompleted.disconnect();

        impl_->requests.clear();

        if (impl_->allocator) {
            impl_->allocator->free(impl_->config->at(0).stream());
            impl_->allocator.reset();
        }

        impl_->camera->release();
        impl_->camera.reset();
    }

    if (impl_->cm) {
        impl_->cm->stop();
        impl_->cm.reset();
    }

    // Clear frame queue
    {
        std::lock_guard<std::mutex> lock(impl_->queue_mutex);
        while (!impl_->frame_queue.empty()) {
            impl_->frame_queue.pop();
        }
    }
}

bool LibcameraSource::read(cv::Mat& frame) {
    if (!impl_->running) {
        return false;
    }

    std::unique_lock<std::mutex> lock(impl_->queue_mutex);

    // Wait for a frame with timeout
    auto timeout = std::chrono::milliseconds(1000);
    if (!impl_->frame_ready.wait_for(lock, timeout, [this] {
        return !impl_->frame_queue.empty() || !impl_->running;
    })) {
        // Timeout
        return false;
    }

    if (!impl_->running || impl_->frame_queue.empty()) {
        return false;
    }

    frame = impl_->frame_queue.front();
    impl_->frame_queue.pop();
    return true;
}

int LibcameraSource::width() const {
    return impl_->actual_width;
}

int LibcameraSource::height() const {
    return impl_->actual_height;
}

double LibcameraSource::fps() const {
    return impl_->actual_fps;
}

std::string LibcameraSource::description() const {
    std::string desc = "libcamera (";
    if (impl_->camera) {
        desc += impl_->camera->id();
    } else {
        desc += "camera_id=" + config_.camera_id;
    }
    desc += ")";
    return desc;
}

bool LibcameraSource::is_available() {
    CameraManager cm;
    int ret = cm.start();
    if (ret < 0) {
        return false;
    }
    bool available = !cm.cameras().empty();
    cm.stop();
    return available;
}

std::vector<std::string> LibcameraSource::list_cameras() {
    std::vector<std::string> result;

    CameraManager cm;
    if (cm.start() < 0) {
        return result;
    }

    for (const auto& camera : cm.cameras()) {
        result.push_back(camera->id());
    }

    cm.stop();
    return result;
}

#endif // HAVE_LIBCAMERA
