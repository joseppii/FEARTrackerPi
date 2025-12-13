#include "pi5_image.h"
#include <cstring>
#include <stdexcept>

Pi5Image::Pi5Image()
    : data_(nullptr)
    , width_(0)
    , height_(0)
    , stride_(0)
    , format_(Pi5PixelFormat::BGR8)
    , data_offset_(0)
    , owns_data_(false)
{
}

Pi5Image::Pi5Image(int width, int height, Pi5PixelFormat format)
    : data_(nullptr)
    , width_(0)
    , height_(0)
    , stride_(0)
    , format_(format)
    , data_offset_(0)
    , owns_data_(false)
{
    allocate(width, height, format);
}

Pi5Image::Pi5Image(const Pi5Size& size, Pi5PixelFormat format)
    : Pi5Image(size.width, size.height, format)
{
}

Pi5Image::Pi5Image(uint8_t* external_data, int width, int height,
                   Pi5PixelFormat format, int stride)
    : width_(width)
    , height_(height)
    , format_(format)
    , data_offset_(0)
    , owns_data_(false)
{
    if (!external_data || width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid external data or dimensions");
    }

    // Use no-op deleter since we don't own the memory
    data_ = std::shared_ptr<uint8_t>(external_data, [](uint8_t*) {});

    // Calculate stride if not provided
    stride_ = (stride > 0) ? stride : width * get_bytes_per_pixel(format);
}

Pi5Image::Pi5Image(const Pi5Image& other)
    : data_(other.data_)
    , width_(other.width_)
    , height_(other.height_)
    , stride_(other.stride_)
    , format_(other.format_)
    , data_offset_(other.data_offset_)
    , owns_data_(other.owns_data_)
{
}

Pi5Image::Pi5Image(Pi5Image&& other) noexcept
    : data_(std::move(other.data_))
    , width_(other.width_)
    , height_(other.height_)
    , stride_(other.stride_)
    , format_(other.format_)
    , data_offset_(other.data_offset_)
    , owns_data_(other.owns_data_)
{
    other.width_ = 0;
    other.height_ = 0;
    other.stride_ = 0;
    other.data_offset_ = 0;
    other.owns_data_ = false;
}

Pi5Image& Pi5Image::operator=(const Pi5Image& other) {
    if (this != &other) {
        data_ = other.data_;
        width_ = other.width_;
        height_ = other.height_;
        stride_ = other.stride_;
        format_ = other.format_;
        data_offset_ = other.data_offset_;
        owns_data_ = other.owns_data_;
    }
    return *this;
}

Pi5Image& Pi5Image::operator=(Pi5Image&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        width_ = other.width_;
        height_ = other.height_;
        stride_ = other.stride_;
        format_ = other.format_;
        data_offset_ = other.data_offset_;
        owns_data_ = other.owns_data_;

        other.width_ = 0;
        other.height_ = 0;
        other.stride_ = 0;
        other.data_offset_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

void Pi5Image::allocate(int width, int height, Pi5PixelFormat format) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid image dimensions");
    }

    width_ = width;
    height_ = height;
    format_ = format;
    stride_ = calculate_aligned_stride(width, format);
    data_offset_ = 0;
    owns_data_ = true;

    size_t total_size = static_cast<size_t>(stride_) * height_;

    // Allocate aligned memory
    // aligned_alloc requires size to be multiple of alignment
    size_t aligned_size = ((total_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

    void* ptr = std::aligned_alloc(ALIGNMENT, aligned_size);
    if (!ptr) {
        throw std::bad_alloc();
    }

    // Zero-initialize
    std::memset(ptr, 0, aligned_size);

    data_ = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr), AlignedDeleter());
}

int Pi5Image::calculate_aligned_stride(int width, Pi5PixelFormat format) {
    int bytes_per_row = width * get_bytes_per_pixel(format);
    // Align stride to 64 bytes for cache optimization
    return ((bytes_per_row + STRIDE_ALIGNMENT - 1) / STRIDE_ALIGNMENT) * STRIDE_ALIGNMENT;
}

Pi5Image Pi5Image::clone() const {
    if (empty()) {
        return Pi5Image();
    }

    Pi5Image result(width_, height_, format_);

    // Copy row by row (handles different strides)
    int row_bytes = width_ * bytes_per_pixel();
    for (int y = 0; y < height_; ++y) {
        std::memcpy(result.row_ptr(y), row_ptr(y), row_bytes);
    }

    return result;
}

Pi5Image Pi5Image::roi(const Pi5Rect& rect) const {
    return roi(rect.x, rect.y, rect.width, rect.height);
}

Pi5Image Pi5Image::roi(int x, int y, int width, int height) const {
    // Validate ROI bounds
    if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x + width > width_ || y + height > height_) {
        throw std::out_of_range("ROI out of image bounds");
    }

    Pi5Image result;
    result.data_ = data_;  // Share data
    result.width_ = width;
    result.height_ = height;
    result.stride_ = stride_;  // Same stride as parent
    result.format_ = format_;
    result.owns_data_ = false;  // ROI doesn't own data

    // Calculate offset to the ROI start
    result.data_offset_ = data_offset_ +
                          static_cast<size_t>(y) * stride_ +
                          static_cast<size_t>(x) * bytes_per_pixel();

    return result;
}

Pi5Image Pi5Image::create_similar() const {
    if (empty()) {
        return Pi5Image();
    }
    return Pi5Image(width_, height_, format_);
}

void Pi5Image::fill(uint8_t value) {
    if (empty()) return;

    int row_bytes = width_ * bytes_per_pixel();
    for (int y = 0; y < height_; ++y) {
        std::memset(row_ptr(y), value, row_bytes);
    }
}

void Pi5Image::fill(const Pi5Scalar& value) {
    if (empty()) return;

    int bpp = bytes_per_pixel();
    int cn = channels();

    for (int y = 0; y < height_; ++y) {
        uint8_t* row = row_ptr(y);
        for (int x = 0; x < width_; ++x) {
            uint8_t* pixel = row + x * bpp;

            if (format_ == Pi5PixelFormat::BGR8 || format_ == Pi5PixelFormat::RGB8) {
                for (int c = 0; c < cn; ++c) {
                    pixel[c] = static_cast<uint8_t>(std::clamp(value[c], 0.0f, 255.0f));
                }
            } else if (format_ == Pi5PixelFormat::GRAY8) {
                pixel[0] = static_cast<uint8_t>(std::clamp(value[0], 0.0f, 255.0f));
            } else if (format_ == Pi5PixelFormat::FLOAT32) {
                *reinterpret_cast<float*>(pixel) = value[0];
            } else if (format_ == Pi5PixelFormat::FLOAT32_C3) {
                float* fp = reinterpret_cast<float*>(pixel);
                for (int c = 0; c < cn; ++c) {
                    fp[c] = value[c];
                }
            }
        }
    }
}

void Pi5Image::swap(Pi5Image& other) {
    std::swap(data_, other.data_);
    std::swap(width_, other.width_);
    std::swap(height_, other.height_);
    std::swap(stride_, other.stride_);
    std::swap(format_, other.format_);
    std::swap(data_offset_, other.data_offset_);
    std::swap(owns_data_, other.owns_data_);
}

int Pi5Image::type() const {
    // Return cv::Mat compatible type value
    // CV_8UC3 = 16, CV_8UC1 = 0, CV_32FC1 = 5, CV_32FC3 = 21
    switch (format_) {
        case Pi5PixelFormat::BGR8:
        case Pi5PixelFormat::RGB8:
            return 16;  // CV_8UC3
        case Pi5PixelFormat::GRAY8:
            return 0;   // CV_8UC1
        case Pi5PixelFormat::FLOAT32:
            return 5;   // CV_32FC1
        case Pi5PixelFormat::FLOAT32_C3:
            return 21;  // CV_32FC3
        default:
            return -1;
    }
}
