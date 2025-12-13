#pragma once

#include "pi5_image.h"
#include <string>
#include <memory>

// Forward declarations for FFmpeg types
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

// FFmpegVideoReader - replaces cv::VideoCapture
class FFmpegVideoReader {
public:
    FFmpegVideoReader();
    ~FFmpegVideoReader();

    // Non-copyable
    FFmpegVideoReader(const FFmpegVideoReader&) = delete;
    FFmpegVideoReader& operator=(const FFmpegVideoReader&) = delete;

    // Movable
    FFmpegVideoReader(FFmpegVideoReader&& other) noexcept;
    FFmpegVideoReader& operator=(FFmpegVideoReader&& other) noexcept;

    // Open video file
    bool open(const std::string& filename);

    // Check if video is open
    bool is_open() const;

    // Close video file
    void close();

    // Read next frame
    // Returns true if frame was successfully read
    // frame is output in BGR8 format (like cv::VideoCapture)
    bool read(Pi5Image& frame);

    // Get video properties
    int width() const;
    int height() const;
    double fps() const;
    int64_t frame_count() const;
    int64_t current_frame() const;

    // Seek to specific frame number
    bool seek(int64_t frame_number);

    // Get duration in seconds
    double duration() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// FFmpegVideoWriter - replaces cv::VideoWriter
class FFmpegVideoWriter {
public:
    FFmpegVideoWriter();
    ~FFmpegVideoWriter();

    // Non-copyable
    FFmpegVideoWriter(const FFmpegVideoWriter&) = delete;
    FFmpegVideoWriter& operator=(const FFmpegVideoWriter&) = delete;

    // Movable
    FFmpegVideoWriter(FFmpegVideoWriter&& other) noexcept;
    FFmpegVideoWriter& operator=(FFmpegVideoWriter&& other) noexcept;

    // Open video file for writing
    // codec can be: "h264", "h265", "mjpeg", "mpeg4"
    bool open(const std::string& filename, int width, int height,
              double fps, const std::string& codec = "h264");

    // Check if writer is open
    bool is_open() const;

    // Close video file
    void close();

    // Write frame (expects BGR8 format)
    bool write(const Pi5Image& frame);

    // Get video properties
    int width() const;
    int height() const;
    double fps() const;
    int64_t frames_written() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
