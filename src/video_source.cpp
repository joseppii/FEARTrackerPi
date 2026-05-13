#include "video_source.h"
#include <iostream>
#include <algorithm>

std::unique_ptr<IVideoSource> create_video_source(
    const std::string& input,
    bool is_camera,
    const std::string& backend,
    const VideoSourceConfig& config) {

    if (!is_camera) {
        // File source
        auto source = std::make_unique<VideoSourceFile>(input);
        if (!source->open()) {
            return nullptr;
        }
        return source;
    }

    // Camera source - select backend
    std::string backend_lower = backend;
    std::transform(backend_lower.begin(), backend_lower.end(),
                   backend_lower.begin(), ::tolower);

    if (backend_lower == "gstreamer") {
        // Force GStreamer
        if (!GStreamerSource::is_available()) {
            std::cerr << "GStreamer backend requested but not available" << std::endl;
            return nullptr;
        }
        auto source = std::make_unique<GStreamerSource>(config);
        if (!source->open()) {
            return nullptr;
        }
        return source;
    }

#ifdef HAVE_LIBCAMERA
    if (backend_lower == "libcamera") {
        // Force libcamera
        if (!LibcameraSource::is_available()) {
            std::cerr << "libcamera backend requested but not available" << std::endl;
            return nullptr;
        }
        auto source = std::make_unique<LibcameraSource>(config);
        if (!source->open()) {
            return nullptr;
        }
        return source;
    }
#else
    if (backend_lower == "libcamera") {
        std::cerr << "libcamera backend requested but not compiled in" << std::endl;
        return nullptr;
    }
#endif

    // Auto-detect: try libcamera first, then GStreamer
    if (backend_lower == "auto" || backend_lower.empty()) {
#ifdef HAVE_LIBCAMERA
        // Try libcamera first (better performance)
        if (LibcameraSource::is_available()) {
            std::cout << "Auto-detected: trying libcamera backend..." << std::endl;
            auto source = std::make_unique<LibcameraSource>(config);
            if (source->open()) {
                return source;
            }
            std::cerr << "libcamera failed, falling back to GStreamer..." << std::endl;
        }
#endif

        // Try GStreamer
        if (GStreamerSource::is_available()) {
            std::cout << "Auto-detected: trying GStreamer backend..." << std::endl;
            auto source = std::make_unique<GStreamerSource>(config);
            if (source->open()) {
                return source;
            }
        }

        std::cerr << "No camera backend available" << std::endl;
        return nullptr;
    }

    std::cerr << "Unknown camera backend: " << backend << std::endl;
    return nullptr;
}

std::unique_ptr<IVideoSource> create_rtp_video_source(const RtpH265Config& config) {
    if (!RtpH265Source::is_available()) {
        std::cerr << "RTP H265 source requires OpenCV with GStreamer support" << std::endl;
        return nullptr;
    }
    auto source = std::make_unique<RtpH265Source>(config);
    if (!source->open()) {
        return nullptr;
    }
    return source;
}

std::vector<std::string> list_available_cameras() {
    std::vector<std::string> cameras;

#ifdef HAVE_LIBCAMERA
    if (LibcameraSource::is_available()) {
        cameras = LibcameraSource::list_cameras();
        if (!cameras.empty()) {
            return cameras;
        }
    }
#endif

    // Fallback: try to detect via /dev/video*
    // This is a simple heuristic - real detection requires probing
    for (int i = 0; i < 10; i++) {
        std::string dev = "/dev/video" + std::to_string(i);
        FILE* f = fopen(dev.c_str(), "r");
        if (f) {
            fclose(f);
            cameras.push_back(dev);
        }
    }

    return cameras;
}
