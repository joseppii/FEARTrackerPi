#pragma once

#include "pi5_types.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

// Image pixel formats
enum class Pi5PixelFormat {
    BGR8,       // 3-channel BGR, 8-bit per channel (like cv::Mat CV_8UC3)
    RGB8,       // 3-channel RGB, 8-bit per channel
    GRAY8,      // 1-channel grayscale, 8-bit
    FLOAT32,    // 1-channel float32 (for normalized data)
    FLOAT32_C3  // 3-channel float32 (for normalized RGB/BGR)
};

// Get bytes per pixel for a format
inline int get_bytes_per_pixel(Pi5PixelFormat format) {
    switch (format) {
        case Pi5PixelFormat::BGR8:
        case Pi5PixelFormat::RGB8:
            return 3;
        case Pi5PixelFormat::GRAY8:
            return 1;
        case Pi5PixelFormat::FLOAT32:
            return 4;
        case Pi5PixelFormat::FLOAT32_C3:
            return 12;
        default:
            return 0;
    }
}

// Get number of channels for a format
inline int get_channels(Pi5PixelFormat format) {
    switch (format) {
        case Pi5PixelFormat::BGR8:
        case Pi5PixelFormat::RGB8:
        case Pi5PixelFormat::FLOAT32_C3:
            return 3;
        case Pi5PixelFormat::GRAY8:
        case Pi5PixelFormat::FLOAT32:
            return 1;
        default:
            return 0;
    }
}

// Custom deleter for aligned memory
struct AlignedDeleter {
    void operator()(uint8_t* ptr) const {
        if (ptr) {
            // Use aligned_alloc compatible free
            std::free(ptr);
        }
    }
};

// Pi5Image - replaces cv::Mat
// Features:
// - 32-byte aligned memory for NEON SIMD operations
// - 64-byte stride alignment for cache optimization
// - Reference counting via shared_ptr
// - ROI support without data copy
class Pi5Image {
public:
    // Default constructor - creates empty image
    Pi5Image();

    // Create image with specified dimensions and format
    Pi5Image(int width, int height, Pi5PixelFormat format);

    // Create image from Pi5Size
    Pi5Image(const Pi5Size& size, Pi5PixelFormat format);

    // Create image from external data (does NOT take ownership)
    // stride = 0 means calculate from width
    Pi5Image(uint8_t* external_data, int width, int height,
             Pi5PixelFormat format, int stride = 0);

    // Copy constructor - shares data (like cv::Mat)
    Pi5Image(const Pi5Image& other);

    // Move constructor
    Pi5Image(Pi5Image&& other) noexcept;

    // Copy assignment - shares data
    Pi5Image& operator=(const Pi5Image& other);

    // Move assignment
    Pi5Image& operator=(Pi5Image&& other) noexcept;

    // Destructor
    ~Pi5Image() = default;

    // Create a deep copy of the image
    Pi5Image clone() const;

    // Check if image is empty (no data allocated)
    bool empty() const { return !data_ || width_ <= 0 || height_ <= 0; }

    // Get image dimensions
    int width() const { return width_; }
    int height() const { return height_; }
    Pi5Size size() const { return Pi5Size(width_, height_); }

    // Get pixel format and related info
    Pi5PixelFormat format() const { return format_; }
    int channels() const { return get_channels(format_); }
    int bytes_per_pixel() const { return get_bytes_per_pixel(format_); }

    // Get stride (bytes per row, may include padding)
    int stride() const { return stride_; }

    // Get total data size in bytes
    size_t data_size() const { return static_cast<size_t>(stride_) * height_; }

    // Get raw data pointer
    uint8_t* data() { return data_.get() + data_offset_; }
    const uint8_t* data() const { return data_.get() + data_offset_; }

    // Get pointer to specific row
    uint8_t* row_ptr(int y) {
        return data_.get() + data_offset_ + static_cast<size_t>(y) * stride_;
    }
    const uint8_t* row_ptr(int y) const {
        return data_.get() + data_offset_ + static_cast<size_t>(y) * stride_;
    }

    // Get pointer to specific pixel
    uint8_t* pixel_ptr(int x, int y) {
        return row_ptr(y) + static_cast<size_t>(x) * bytes_per_pixel();
    }
    const uint8_t* pixel_ptr(int x, int y) const {
        return row_ptr(y) + static_cast<size_t>(x) * bytes_per_pixel();
    }

    // Access pixel as specific type (no bounds checking for performance)
    template<typename T>
    T& at(int x, int y) {
        return *reinterpret_cast<T*>(pixel_ptr(x, y));
    }
    template<typename T>
    const T& at(int x, int y) const {
        return *reinterpret_cast<const T*>(pixel_ptr(x, y));
    }

    // Create ROI (region of interest) - shares data, no copy
    // Note: ROI may not be aligned if x is not aligned
    Pi5Image roi(const Pi5Rect& rect) const;
    Pi5Image roi(int x, int y, int width, int height) const;

    // Check if this image owns its data (vs ROI or external data)
    bool owns_data() const { return owns_data_; }

    // Check if data is 32-byte aligned (required for NEON)
    bool is_aligned() const {
        return (reinterpret_cast<uintptr_t>(data()) & 31) == 0;
    }

    // Create new image with same dimensions/format but fresh data
    Pi5Image create_similar() const;

    // Fill entire image with a value
    void fill(uint8_t value);
    void fill(const Pi5Scalar& value);  // For multi-channel

    // Swap contents with another image
    void swap(Pi5Image& other);

    // cv::Mat compatibility helpers
    int rows() const { return height_; }
    int cols() const { return width_; }
    int type() const;  // Returns CV_8UC3-like value for compatibility

private:
    // Allocate aligned memory
    void allocate(int width, int height, Pi5PixelFormat format);

    // Calculate aligned stride
    static int calculate_aligned_stride(int width, Pi5PixelFormat format);

    std::shared_ptr<uint8_t> data_;  // Shared pointer for reference counting
    int width_;
    int height_;
    int stride_;
    Pi5PixelFormat format_;
    size_t data_offset_;  // Offset into data_ for ROIs
    bool owns_data_;      // True if we allocated the memory

    // Memory alignment constants
    static constexpr size_t ALIGNMENT = 32;        // 32-byte alignment for NEON
    static constexpr size_t STRIDE_ALIGNMENT = 64; // 64-byte stride for cache
};

// Utility function to convert cv::Mat depth to Pi5PixelFormat
// CV_8U = 0, CV_8UC3 would be type 16
inline Pi5PixelFormat format_from_cv_type(int cv_type) {
    int depth = cv_type & 7;      // Lower 3 bits = depth
    int cn = (cv_type >> 3) + 1;  // Upper bits = channels - 1

    if (depth == 0) {  // CV_8U
        if (cn == 3) return Pi5PixelFormat::BGR8;
        if (cn == 1) return Pi5PixelFormat::GRAY8;
    } else if (depth == 5) {  // CV_32F
        if (cn == 1) return Pi5PixelFormat::FLOAT32;
        if (cn == 3) return Pi5PixelFormat::FLOAT32_C3;
    }
    throw std::runtime_error("Unsupported cv::Mat type");
}
