#include "video_source.h"
#include <iostream>

// VideoSourceFile implementation

VideoSourceFile::VideoSourceFile(const std::string& filename)
    : filename_(filename) {
}

VideoSourceFile::~VideoSourceFile() {
    close();
}

bool VideoSourceFile::open() {
    if (filename_.empty()) {
        std::cerr << "VideoSourceFile: No filename specified" << std::endl;
        return false;
    }

    cap_.open(filename_);
    if (!cap_.isOpened()) {
        std::cerr << "VideoSourceFile: Failed to open " << filename_ << std::endl;
        return false;
    }

    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = cap_.get(cv::CAP_PROP_FPS);
    total_frames_ = static_cast<int64_t>(cap_.get(cv::CAP_PROP_FRAME_COUNT));

    // Handle invalid FPS (some files report 0)
    if (fps_ <= 0) {
        fps_ = 30.0;
    }

    return true;
}

bool VideoSourceFile::is_open() const {
    return cap_.isOpened();
}

void VideoSourceFile::close() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoSourceFile::read(cv::Mat& frame) {
    if (!cap_.isOpened()) {
        return false;
    }
    return cap_.read(frame) && !frame.empty();
}

int VideoSourceFile::width() const {
    return width_;
}

int VideoSourceFile::height() const {
    return height_;
}

double VideoSourceFile::fps() const {
    return fps_;
}

int64_t VideoSourceFile::frame_count() const {
    return total_frames_;
}

std::string VideoSourceFile::description() const {
    return "File: " + filename_;
}
