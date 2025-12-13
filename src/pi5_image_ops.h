#pragma once

#include "pi5_image.h"
#include "pi5_types.h"
#include <vector>

namespace pi5 {

// ============================================================================
// Image Resize Operations
// ============================================================================

// Bilinear interpolation resize - MUST match OpenCV behavior for tracking accuracy
// Uses the formula: src_coord = (dst_coord + 0.5) * src_size / dst_size - 0.5
Pi5Image resize_bilinear(const Pi5Image& src, int dst_width, int dst_height);
Pi5Image resize_bilinear(const Pi5Image& src, const Pi5Size& dst_size);

// ============================================================================
// Color Space Conversions
// ============================================================================

// BGR to RGB conversion (and vice versa - same operation)
Pi5Image bgr_to_rgb(const Pi5Image& src);
void bgr_to_rgb_inplace(Pi5Image& img);

// Convert uint8 image to float32 normalized [0, 1]
Pi5Image convert_to_float32(const Pi5Image& src);

// Convert uint8 image to float32 with ImageNet normalization
// Applies: (pixel / 255.0 - mean) / std
// mean = [0.485, 0.456, 0.406] (RGB order)
// std = [0.229, 0.224, 0.225]
Pi5Image normalize_imagenet(const Pi5Image& src);

// ============================================================================
// Padding Operations
// ============================================================================

// Add constant border padding (like cv::copyMakeBorder with BORDER_CONSTANT)
Pi5Image copy_make_border(const Pi5Image& src,
                          int top, int bottom, int left, int right,
                          const Pi5Scalar& value);

// ============================================================================
// Statistical Operations
// ============================================================================

// Calculate mean color of image (like cv::mean)
Pi5Scalar calculate_mean(const Pi5Image& src);

// Calculate mean of ROI region
Pi5Scalar calculate_mean(const Pi5Image& src, const Pi5Rect& roi);

// ============================================================================
// Channel Operations
// ============================================================================

// Split HWC image to CHW format (required for ONNX input)
// Returns vector of channels, each is a single-channel image
std::vector<Pi5Image> split_channels(const Pi5Image& src);

// Split and flatten to contiguous CHW float array (for ONNX tensor input)
// Output format: [C, H, W] where each element is float32
std::vector<float> to_chw_float(const Pi5Image& src);

// Split with ImageNet normalization directly to CHW float array
// Combines convert_to_float32, normalize_imagenet, and to_chw_float
std::vector<float> to_chw_normalized(const Pi5Image& src);

// ============================================================================
// Crop Operations
// ============================================================================

// Extract a crop from source image, handling out-of-bounds with padding
// Returns the cropped image padded with mean_color where needed
Pi5Image extract_crop_with_padding(const Pi5Image& src,
                                   const Pi5Rect& crop_rect,
                                   const Pi5Scalar& pad_color);

// Extract crop centered at a point with given size
Pi5Image extract_centered_crop(const Pi5Image& src,
                               const Pi5Point2f& center,
                               const Pi5Size& crop_size,
                               const Pi5Scalar& pad_color);

// ============================================================================
// Utility Operations
// ============================================================================

// Copy one image into another at specified location
void copy_to(const Pi5Image& src, Pi5Image& dst, int dst_x, int dst_y);

// Create a cropped copy of image
Pi5Image crop(const Pi5Image& src, const Pi5Rect& rect);

// L2 norm of a vector (for distance calculations)
float norm_l2(const std::vector<float>& vec);

// L2 distance between two points represented as vectors
float distance_l2(const std::vector<float>& v1, const std::vector<float>& v2);

}  // namespace pi5
