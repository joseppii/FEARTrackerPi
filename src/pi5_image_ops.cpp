#include "pi5_image_ops.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

namespace pi5 {

// ============================================================================
// Image Resize - Bilinear Interpolation
// ============================================================================

// Helper: Clamp value to range
template<typename T>
static inline T clamp_val(T val, T min_val, T max_val) {
    return std::max(min_val, std::min(val, max_val));
}

Pi5Image resize_bilinear(const Pi5Image& src, int dst_width, int dst_height) {
    if (src.empty()) {
        throw std::invalid_argument("Cannot resize empty image");
    }

    if (dst_width <= 0 || dst_height <= 0) {
        throw std::invalid_argument("Invalid destination dimensions");
    }

    Pi5Image dst(dst_width, dst_height, src.format());

    const int src_width = src.width();
    const int src_height = src.height();
    const int channels = src.channels();
    const int bpp = src.bytes_per_pixel();

    const float scale_x = static_cast<float>(src_width) / dst_width;
    const float scale_y = static_cast<float>(src_height) / dst_height;

    // Handle different pixel formats
    if (src.format() == Pi5PixelFormat::BGR8 ||
        src.format() == Pi5PixelFormat::RGB8) {

#if USE_NEON
        // NEON-optimized 8-bit 3-channel resize
        // Precompute x coordinates and weights for the entire row
        std::vector<int> x0_arr(dst_width);
        std::vector<int> x1_arr(dst_width);
        std::vector<float> fx_arr(dst_width);

        for (int dx = 0; dx < dst_width; ++dx) {
            float sx = (dx + 0.5f) * scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            x0_arr[dx] = clamp_val(x0, 0, src_width - 1);
            x1_arr[dx] = clamp_val(x0 + 1, 0, src_width - 1);
            fx_arr[dx] = sx - x0;
        }

        for (int dy = 0; dy < dst_height; ++dy) {
            uint8_t* dst_row = dst.row_ptr(dy);

            float sy = (dy + 0.5f) * scale_y - 0.5f;
            int y0 = static_cast<int>(std::floor(sy));
            int y1 = y0 + 1;
            float fy = sy - y0;

            y0 = clamp_val(y0, 0, src_height - 1);
            y1 = clamp_val(y1, 0, src_height - 1);

            const uint8_t* src_row0 = src.row_ptr(y0);
            const uint8_t* src_row1 = src.row_ptr(y1);

            const float fy1 = 1.0f - fy;

            // Process 4 destination pixels at a time
            int dx = 0;
            const int neon_width = (dst_width / 4) * 4;

            for (; dx < neon_width; dx += 4) {
                // Compute 4 pixels and store in arrays
                float b_arr[4], g_arr[4], r_arr[4];

                for (int i = 0; i < 4; ++i) {
                    int idx = dx + i;
                    int x0 = x0_arr[idx];
                    int x1 = x1_arr[idx];
                    float fx = fx_arr[idx];
                    float fx1 = 1.0f - fx;

                    const uint8_t* p00 = src_row0 + x0 * 3;
                    const uint8_t* p10 = src_row0 + x1 * 3;
                    const uint8_t* p01 = src_row1 + x0 * 3;
                    const uint8_t* p11 = src_row1 + x1 * 3;

                    float w00 = fx1 * fy1;
                    float w10 = fx * fy1;
                    float w01 = fx1 * fy;
                    float w11 = fx * fy;

                    b_arr[i] = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;
                    g_arr[i] = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
                    r_arr[i] = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
                }

                // Load into NEON vectors
                float32x4_t result_b = vld1q_f32(b_arr);
                float32x4_t result_g = vld1q_f32(g_arr);
                float32x4_t result_r = vld1q_f32(r_arr);

                // Add 0.5 for rounding
                const float32x4_t half = vdupq_n_f32(0.5f);
                result_b = vaddq_f32(result_b, half);
                result_g = vaddq_f32(result_g, half);
                result_r = vaddq_f32(result_r, half);

                // Convert to uint32, then narrow to uint8
                uint32x4_t b_u32 = vcvtq_u32_f32(result_b);
                uint32x4_t g_u32 = vcvtq_u32_f32(result_g);
                uint32x4_t r_u32 = vcvtq_u32_f32(result_r);

                // Narrow and saturate
                uint16x4_t b_u16 = vqmovn_u32(b_u32);
                uint16x4_t g_u16 = vqmovn_u32(g_u32);
                uint16x4_t r_u16 = vqmovn_u32(r_u32);

                uint8x8_t b_u8 = vqmovn_u16(vcombine_u16(b_u16, b_u16));
                uint8x8_t g_u8 = vqmovn_u16(vcombine_u16(g_u16, g_u16));
                uint8x8_t r_u8 = vqmovn_u16(vcombine_u16(r_u16, r_u16));

                // Store 4 pixels interleaved
                uint8_t* dst_pixel = dst_row + dx * 3;
                dst_pixel[0] = vget_lane_u8(b_u8, 0);
                dst_pixel[1] = vget_lane_u8(g_u8, 0);
                dst_pixel[2] = vget_lane_u8(r_u8, 0);
                dst_pixel[3] = vget_lane_u8(b_u8, 1);
                dst_pixel[4] = vget_lane_u8(g_u8, 1);
                dst_pixel[5] = vget_lane_u8(r_u8, 1);
                dst_pixel[6] = vget_lane_u8(b_u8, 2);
                dst_pixel[7] = vget_lane_u8(g_u8, 2);
                dst_pixel[8] = vget_lane_u8(r_u8, 2);
                dst_pixel[9] = vget_lane_u8(b_u8, 3);
                dst_pixel[10] = vget_lane_u8(g_u8, 3);
                dst_pixel[11] = vget_lane_u8(r_u8, 3);
            }

            // Handle remaining pixels
            for (; dx < dst_width; ++dx) {
                int x0 = x0_arr[dx];
                int x1 = x1_arr[dx];
                float fx = fx_arr[dx];

                const uint8_t* p00 = src_row0 + x0 * 3;
                const uint8_t* p10 = src_row0 + x1 * 3;
                const uint8_t* p01 = src_row1 + x0 * 3;
                const uint8_t* p11 = src_row1 + x1 * 3;

                float w00 = (1.0f - fx) * fy1;
                float w10 = fx * fy1;
                float w01 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                uint8_t* dst_pixel = dst_row + dx * 3;
                for (int c = 0; c < 3; ++c) {
                    float val = p00[c] * w00 + p10[c] * w10 + p01[c] * w01 + p11[c] * w11;
                    dst_pixel[c] = static_cast<uint8_t>(clamp_val(val + 0.5f, 0.0f, 255.0f));
                }
            }
        }
#else
        // Scalar 8-bit 3-channel
        for (int dy = 0; dy < dst_height; ++dy) {
            uint8_t* dst_row = dst.row_ptr(dy);

            // OpenCV formula for coordinate mapping
            float sy = (dy + 0.5f) * scale_y - 0.5f;
            int y0 = static_cast<int>(std::floor(sy));
            int y1 = y0 + 1;
            float fy = sy - y0;

            y0 = clamp_val(y0, 0, src_height - 1);
            y1 = clamp_val(y1, 0, src_height - 1);

            const uint8_t* src_row0 = src.row_ptr(y0);
            const uint8_t* src_row1 = src.row_ptr(y1);

            for (int dx = 0; dx < dst_width; ++dx) {
                float sx = (dx + 0.5f) * scale_x - 0.5f;
                int x0 = static_cast<int>(std::floor(sx));
                int x1 = x0 + 1;
                float fx = sx - x0;

                x0 = clamp_val(x0, 0, src_width - 1);
                x1 = clamp_val(x1, 0, src_width - 1);

                // Get 4 neighboring pixels
                const uint8_t* p00 = src_row0 + x0 * 3;
                const uint8_t* p10 = src_row0 + x1 * 3;
                const uint8_t* p01 = src_row1 + x0 * 3;
                const uint8_t* p11 = src_row1 + x1 * 3;

                // Bilinear interpolation weights
                float w00 = (1.0f - fx) * (1.0f - fy);
                float w10 = fx * (1.0f - fy);
                float w01 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                // Interpolate each channel
                uint8_t* dst_pixel = dst_row + dx * 3;
                for (int c = 0; c < 3; ++c) {
                    float val = p00[c] * w00 + p10[c] * w10 +
                                p01[c] * w01 + p11[c] * w11;
                    dst_pixel[c] = static_cast<uint8_t>(clamp_val(val + 0.5f, 0.0f, 255.0f));
                }
            }
        }
#endif
    } else if (src.format() == Pi5PixelFormat::GRAY8) {
        // 8-bit grayscale
        for (int dy = 0; dy < dst_height; ++dy) {
            uint8_t* dst_row = dst.row_ptr(dy);

            float sy = (dy + 0.5f) * scale_y - 0.5f;
            int y0 = static_cast<int>(std::floor(sy));
            int y1 = y0 + 1;
            float fy = sy - y0;

            y0 = clamp_val(y0, 0, src_height - 1);
            y1 = clamp_val(y1, 0, src_height - 1);

            const uint8_t* src_row0 = src.row_ptr(y0);
            const uint8_t* src_row1 = src.row_ptr(y1);

            for (int dx = 0; dx < dst_width; ++dx) {
                float sx = (dx + 0.5f) * scale_x - 0.5f;
                int x0 = static_cast<int>(std::floor(sx));
                int x1 = x0 + 1;
                float fx = sx - x0;

                x0 = clamp_val(x0, 0, src_width - 1);
                x1 = clamp_val(x1, 0, src_width - 1);

                float w00 = (1.0f - fx) * (1.0f - fy);
                float w10 = fx * (1.0f - fy);
                float w01 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                float val = src_row0[x0] * w00 + src_row0[x1] * w10 +
                            src_row1[x0] * w01 + src_row1[x1] * w11;
                dst_row[dx] = static_cast<uint8_t>(clamp_val(val + 0.5f, 0.0f, 255.0f));
            }
        }
    } else if (src.format() == Pi5PixelFormat::FLOAT32_C3) {
        // Float 3-channel
        for (int dy = 0; dy < dst_height; ++dy) {
            float* dst_row = reinterpret_cast<float*>(dst.row_ptr(dy));

            float sy = (dy + 0.5f) * scale_y - 0.5f;
            int y0 = static_cast<int>(std::floor(sy));
            int y1 = y0 + 1;
            float fy = sy - y0;

            y0 = clamp_val(y0, 0, src_height - 1);
            y1 = clamp_val(y1, 0, src_height - 1);

            const float* src_row0 = reinterpret_cast<const float*>(src.row_ptr(y0));
            const float* src_row1 = reinterpret_cast<const float*>(src.row_ptr(y1));

            for (int dx = 0; dx < dst_width; ++dx) {
                float sx = (dx + 0.5f) * scale_x - 0.5f;
                int x0 = static_cast<int>(std::floor(sx));
                int x1 = x0 + 1;
                float fx = sx - x0;

                x0 = clamp_val(x0, 0, src_width - 1);
                x1 = clamp_val(x1, 0, src_width - 1);

                const float* p00 = src_row0 + x0 * 3;
                const float* p10 = src_row0 + x1 * 3;
                const float* p01 = src_row1 + x0 * 3;
                const float* p11 = src_row1 + x1 * 3;

                float w00 = (1.0f - fx) * (1.0f - fy);
                float w10 = fx * (1.0f - fy);
                float w01 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                float* dst_pixel = dst_row + dx * 3;
                for (int c = 0; c < 3; ++c) {
                    dst_pixel[c] = p00[c] * w00 + p10[c] * w10 +
                                   p01[c] * w01 + p11[c] * w11;
                }
            }
        }
    } else {
        throw std::invalid_argument("Unsupported pixel format for resize");
    }

    return dst;
}

Pi5Image resize_bilinear(const Pi5Image& src, const Pi5Size& dst_size) {
    return resize_bilinear(src, dst_size.width, dst_size.height);
}

// ============================================================================
// Color Space Conversions
// ============================================================================

Pi5Image bgr_to_rgb(const Pi5Image& src) {
    if (src.empty()) {
        return Pi5Image();
    }

    if (src.format() != Pi5PixelFormat::BGR8 &&
        src.format() != Pi5PixelFormat::RGB8) {
        throw std::invalid_argument("bgr_to_rgb requires BGR8 or RGB8 format");
    }

    Pi5Image dst(src.width(), src.height(), src.format());

    const int width = src.width();
    const int height = src.height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        uint8_t* dst_row = dst.row_ptr(y);

        for (int x = 0; x < width; ++x) {
            const uint8_t* sp = src_row + x * 3;
            uint8_t* dp = dst_row + x * 3;
            // Swap B and R channels
            dp[0] = sp[2];
            dp[1] = sp[1];
            dp[2] = sp[0];
        }
    }

    return dst;
}

void bgr_to_rgb_inplace(Pi5Image& img) {
    if (img.empty()) return;

    if (img.format() != Pi5PixelFormat::BGR8 &&
        img.format() != Pi5PixelFormat::RGB8) {
        throw std::invalid_argument("bgr_to_rgb_inplace requires BGR8 or RGB8 format");
    }

    const int width = img.width();
    const int height = img.height();

#if USE_NEON
    // NEON-optimized: process 16 pixels at a time
    const int neon_width = (width / 16) * 16;

    for (int y = 0; y < height; ++y) {
        uint8_t* row = img.row_ptr(y);
        int x = 0;

        // Process 16 pixels (48 bytes) at a time using NEON
        for (; x < neon_width; x += 16) {
            uint8_t* p = row + x * 3;

            // Load 48 bytes as 3 interleaved channels (BGR BGR BGR...)
            uint8x16x3_t bgr = vld3q_u8(p);

            // Swap B and R channels
            uint8x16_t temp = bgr.val[0];
            bgr.val[0] = bgr.val[2];
            bgr.val[2] = temp;

            // Store back as RGB RGB RGB...
            vst3q_u8(p, bgr);
        }

        // Handle remaining pixels with scalar code
        for (; x < width; ++x) {
            uint8_t* p = row + x * 3;
            std::swap(p[0], p[2]);
        }
    }
#else
    // Scalar fallback
    for (int y = 0; y < height; ++y) {
        uint8_t* row = img.row_ptr(y);
        for (int x = 0; x < width; ++x) {
            uint8_t* p = row + x * 3;
            std::swap(p[0], p[2]);
        }
    }
#endif
}

Pi5Image convert_to_float32(const Pi5Image& src) {
    if (src.empty()) {
        return Pi5Image();
    }

    if (src.format() != Pi5PixelFormat::BGR8 &&
        src.format() != Pi5PixelFormat::RGB8) {
        throw std::invalid_argument("convert_to_float32 requires BGR8 or RGB8 format");
    }

    Pi5Image dst(src.width(), src.height(), Pi5PixelFormat::FLOAT32_C3);

    const int width = src.width();
    const int height = src.height();

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        float* dst_row = reinterpret_cast<float*>(dst.row_ptr(y));

        for (int x = 0; x < width; ++x) {
            const uint8_t* sp = src_row + x * 3;
            float* dp = dst_row + x * 3;
            dp[0] = sp[0] / 255.0f;
            dp[1] = sp[1] / 255.0f;
            dp[2] = sp[2] / 255.0f;
        }
    }

    return dst;
}

Pi5Image normalize_imagenet(const Pi5Image& src) {
    if (src.empty()) {
        return Pi5Image();
    }

    if (src.format() != Pi5PixelFormat::BGR8 &&
        src.format() != Pi5PixelFormat::RGB8) {
        throw std::invalid_argument("normalize_imagenet requires BGR8 or RGB8 format");
    }

    // ImageNet normalization values (RGB order)
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_inv[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    const float scale = 1.0f / 255.0f;

    Pi5Image dst(src.width(), src.height(), Pi5PixelFormat::FLOAT32_C3);

    const int width = src.width();
    const int height = src.height();
    const bool is_bgr = (src.format() == Pi5PixelFormat::BGR8);

#if USE_NEON
    // NEON constants
    const float32x4_t v_scale = vdupq_n_f32(scale);
    const float32x4_t v_mean_r = vdupq_n_f32(mean[0]);
    const float32x4_t v_mean_g = vdupq_n_f32(mean[1]);
    const float32x4_t v_mean_b = vdupq_n_f32(mean[2]);
    const float32x4_t v_std_inv_r = vdupq_n_f32(std_inv[0]);
    const float32x4_t v_std_inv_g = vdupq_n_f32(std_inv[1]);
    const float32x4_t v_std_inv_b = vdupq_n_f32(std_inv[2]);

    const int neon_width = (width / 4) * 4;

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        float* dst_row = reinterpret_cast<float*>(dst.row_ptr(y));
        int x = 0;

        for (; x < neon_width; x += 4) {
            const uint8_t* sp = src_row + x * 3;

            // Extract BGR/RGB values - manual deinterleave for 4 pixels
            uint8_t b0 = sp[0], g0 = sp[1], r0 = sp[2];
            uint8_t b1 = sp[3], g1 = sp[4], r1 = sp[5];
            uint8_t b2 = sp[6], g2 = sp[7], r2 = sp[8];
            uint8_t b3 = sp[9], g3 = sp[10], r3 = sp[11];

            float32x4_t v_ch0, v_ch1, v_ch2;

            if (is_bgr) {
                // BGR -> need to swap to RGB
                float ch0_arr[4] = {(float)r0, (float)r1, (float)r2, (float)r3};  // R
                float ch1_arr[4] = {(float)g0, (float)g1, (float)g2, (float)g3};  // G
                float ch2_arr[4] = {(float)b0, (float)b1, (float)b2, (float)b3};  // B
                v_ch0 = vld1q_f32(ch0_arr);
                v_ch1 = vld1q_f32(ch1_arr);
                v_ch2 = vld1q_f32(ch2_arr);
            } else {
                // Already RGB
                float ch0_arr[4] = {(float)b0, (float)b1, (float)b2, (float)b3};  // R (first channel)
                float ch1_arr[4] = {(float)g0, (float)g1, (float)g2, (float)g3};  // G
                float ch2_arr[4] = {(float)r0, (float)r1, (float)r2, (float)r3};  // B
                v_ch0 = vld1q_f32(ch0_arr);
                v_ch1 = vld1q_f32(ch1_arr);
                v_ch2 = vld1q_f32(ch2_arr);
            }

            // Scale to [0,1]
            v_ch0 = vmulq_f32(v_ch0, v_scale);
            v_ch1 = vmulq_f32(v_ch1, v_scale);
            v_ch2 = vmulq_f32(v_ch2, v_scale);

            // Apply normalization: (x - mean) * std_inv
            v_ch0 = vmulq_f32(vsubq_f32(v_ch0, v_mean_r), v_std_inv_r);
            v_ch1 = vmulq_f32(vsubq_f32(v_ch1, v_mean_g), v_std_inv_g);
            v_ch2 = vmulq_f32(vsubq_f32(v_ch2, v_mean_b), v_std_inv_b);

            // Store interleaved RGB floats
            float* dp = dst_row + x * 3;
            float32x4x3_t rgb;
            rgb.val[0] = v_ch0;
            rgb.val[1] = v_ch1;
            rgb.val[2] = v_ch2;
            vst3q_f32(dp, rgb);
        }

        // Handle remaining pixels
        for (; x < width; ++x) {
            const uint8_t* sp = src_row + x * 3;
            float* dp = dst_row + x * 3;

            float r = (is_bgr ? sp[2] : sp[0]) * scale;
            float g = sp[1] * scale;
            float b = (is_bgr ? sp[0] : sp[2]) * scale;

            dp[0] = (r - mean[0]) * std_inv[0];
            dp[1] = (g - mean[1]) * std_inv[1];
            dp[2] = (b - mean[2]) * std_inv[2];
        }
    }
#else
    // Scalar fallback
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        float* dst_row = reinterpret_cast<float*>(dst.row_ptr(y));

        for (int x = 0; x < width; ++x) {
            const uint8_t* sp = src_row + x * 3;
            float* dp = dst_row + x * 3;

            float r = (is_bgr ? sp[2] : sp[0]) * scale;
            float g = sp[1] * scale;
            float b = (is_bgr ? sp[0] : sp[2]) * scale;

            dp[0] = (r - mean[0]) * std_inv[0];
            dp[1] = (g - mean[1]) * std_inv[1];
            dp[2] = (b - mean[2]) * std_inv[2];
        }
    }
#endif

    return dst;
}

// ============================================================================
// Padding Operations
// ============================================================================

Pi5Image copy_make_border(const Pi5Image& src,
                          int top, int bottom, int left, int right,
                          const Pi5Scalar& value) {
    if (src.empty()) {
        throw std::invalid_argument("Cannot add border to empty image");
    }

    int new_width = src.width() + left + right;
    int new_height = src.height() + top + bottom;

    Pi5Image dst(new_width, new_height, src.format());

    // Fill with border value
    dst.fill(value);

    // Copy source image to center
    copy_to(src, dst, left, top);

    return dst;
}

// ============================================================================
// Statistical Operations
// ============================================================================

Pi5Scalar calculate_mean(const Pi5Image& src) {
    if (src.empty()) {
        return Pi5Scalar();
    }

    const int width = src.width();
    const int height = src.height();
    const int channels = src.channels();
    const int total_pixels = width * height;

    double sum[4] = {0.0, 0.0, 0.0, 0.0};

    if (src.format() == Pi5PixelFormat::BGR8 ||
        src.format() == Pi5PixelFormat::RGB8) {
        for (int y = 0; y < height; ++y) {
            const uint8_t* row = src.row_ptr(y);
            for (int x = 0; x < width; ++x) {
                const uint8_t* p = row + x * 3;
                sum[0] += p[0];
                sum[1] += p[1];
                sum[2] += p[2];
            }
        }
    } else if (src.format() == Pi5PixelFormat::GRAY8) {
        for (int y = 0; y < height; ++y) {
            const uint8_t* row = src.row_ptr(y);
            for (int x = 0; x < width; ++x) {
                sum[0] += row[x];
            }
        }
    } else if (src.format() == Pi5PixelFormat::FLOAT32_C3) {
        for (int y = 0; y < height; ++y) {
            const float* row = reinterpret_cast<const float*>(src.row_ptr(y));
            for (int x = 0; x < width; ++x) {
                const float* p = row + x * 3;
                sum[0] += p[0];
                sum[1] += p[1];
                sum[2] += p[2];
            }
        }
    }

    Pi5Scalar result;
    for (int c = 0; c < channels; ++c) {
        result[c] = static_cast<float>(sum[c] / total_pixels);
    }

    return result;
}

Pi5Scalar calculate_mean(const Pi5Image& src, const Pi5Rect& roi) {
    // Create ROI and calculate mean
    Pi5Image roi_img = src.roi(roi);
    return calculate_mean(roi_img);
}

// ============================================================================
// Channel Operations
// ============================================================================

std::vector<Pi5Image> split_channels(const Pi5Image& src) {
    if (src.empty()) {
        return std::vector<Pi5Image>();
    }

    const int width = src.width();
    const int height = src.height();
    const int channels = src.channels();

    std::vector<Pi5Image> result;

    if (src.format() == Pi5PixelFormat::BGR8 ||
        src.format() == Pi5PixelFormat::RGB8) {
        for (int c = 0; c < channels; ++c) {
            Pi5Image channel(width, height, Pi5PixelFormat::GRAY8);
            for (int y = 0; y < height; ++y) {
                const uint8_t* src_row = src.row_ptr(y);
                uint8_t* dst_row = channel.row_ptr(y);
                for (int x = 0; x < width; ++x) {
                    dst_row[x] = src_row[x * 3 + c];
                }
            }
            result.push_back(std::move(channel));
        }
    } else if (src.format() == Pi5PixelFormat::FLOAT32_C3) {
        for (int c = 0; c < channels; ++c) {
            Pi5Image channel(width, height, Pi5PixelFormat::FLOAT32);
            for (int y = 0; y < height; ++y) {
                const float* src_row = reinterpret_cast<const float*>(src.row_ptr(y));
                float* dst_row = reinterpret_cast<float*>(channel.row_ptr(y));
                for (int x = 0; x < width; ++x) {
                    dst_row[x] = src_row[x * 3 + c];
                }
            }
            result.push_back(std::move(channel));
        }
    }

    return result;
}

std::vector<float> to_chw_float(const Pi5Image& src) {
    if (src.empty()) {
        return std::vector<float>();
    }

    const int width = src.width();
    const int height = src.height();
    const int channels = src.channels();
    const int spatial_size = width * height;

    std::vector<float> result(channels * spatial_size);

    if (src.format() == Pi5PixelFormat::BGR8 ||
        src.format() == Pi5PixelFormat::RGB8) {
        for (int c = 0; c < channels; ++c) {
            float* channel_data = result.data() + c * spatial_size;
            for (int y = 0; y < height; ++y) {
                const uint8_t* src_row = src.row_ptr(y);
                for (int x = 0; x < width; ++x) {
                    channel_data[y * width + x] = src_row[x * 3 + c] / 255.0f;
                }
            }
        }
    } else if (src.format() == Pi5PixelFormat::FLOAT32_C3) {
        for (int c = 0; c < channels; ++c) {
            float* channel_data = result.data() + c * spatial_size;
            for (int y = 0; y < height; ++y) {
                const float* src_row = reinterpret_cast<const float*>(src.row_ptr(y));
                for (int x = 0; x < width; ++x) {
                    channel_data[y * width + x] = src_row[x * 3 + c];
                }
            }
        }
    }

    return result;
}

std::vector<float> to_chw_normalized(const Pi5Image& src) {
    if (src.empty()) {
        return std::vector<float>();
    }

    if (src.format() != Pi5PixelFormat::BGR8 &&
        src.format() != Pi5PixelFormat::RGB8) {
        throw std::invalid_argument("to_chw_normalized requires BGR8 or RGB8 format");
    }

    // ImageNet normalization values (RGB order)
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_inv[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};
    const float scale = 1.0f / 255.0f;

    const int width = src.width();
    const int height = src.height();
    const int spatial_size = width * height;
    const bool is_bgr = (src.format() == Pi5PixelFormat::BGR8);

    // Output is in RGB order, CHW layout
    std::vector<float> result(3 * spatial_size);

    float* r_data = result.data();                    // R channel
    float* g_data = result.data() + spatial_size;    // G channel
    float* b_data = result.data() + 2 * spatial_size; // B channel

#if USE_NEON
    // NEON constants
    const float32x4_t v_scale = vdupq_n_f32(scale);
    const float32x4_t v_mean_r = vdupq_n_f32(mean[0]);
    const float32x4_t v_mean_g = vdupq_n_f32(mean[1]);
    const float32x4_t v_mean_b = vdupq_n_f32(mean[2]);
    const float32x4_t v_std_inv_r = vdupq_n_f32(std_inv[0]);
    const float32x4_t v_std_inv_g = vdupq_n_f32(std_inv[1]);
    const float32x4_t v_std_inv_b = vdupq_n_f32(std_inv[2]);

    // Process 8 pixels at a time
    const int neon_width = (width / 8) * 8;

    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        int base_idx = y * width;
        int x = 0;

        for (; x < neon_width; x += 8) {
            const uint8_t* p = src_row + x * 3;
            int idx = base_idx + x;

            // Load 8 pixels (24 bytes) as interleaved BGR/RGB
            uint8x8x3_t pixels = vld3_u8(p);

            // Get the channel data based on BGR vs RGB order
            uint8x8_t ch0 = is_bgr ? pixels.val[2] : pixels.val[0];  // R
            uint8x8_t ch1 = pixels.val[1];                           // G
            uint8x8_t ch2 = is_bgr ? pixels.val[0] : pixels.val[2];  // B

            // Convert to 16-bit then 32-bit float
            uint16x8_t ch0_16 = vmovl_u8(ch0);
            uint16x8_t ch1_16 = vmovl_u8(ch1);
            uint16x8_t ch2_16 = vmovl_u8(ch2);

            // Process first 4 pixels
            float32x4_t r_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ch0_16)));
            float32x4_t g_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ch1_16)));
            float32x4_t b_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(ch2_16)));

            // Scale to [0,1] and normalize
            r_lo = vmulq_f32(r_lo, v_scale);
            g_lo = vmulq_f32(g_lo, v_scale);
            b_lo = vmulq_f32(b_lo, v_scale);

            r_lo = vmulq_f32(vsubq_f32(r_lo, v_mean_r), v_std_inv_r);
            g_lo = vmulq_f32(vsubq_f32(g_lo, v_mean_g), v_std_inv_g);
            b_lo = vmulq_f32(vsubq_f32(b_lo, v_mean_b), v_std_inv_b);

            // Store first 4 pixels
            vst1q_f32(r_data + idx, r_lo);
            vst1q_f32(g_data + idx, g_lo);
            vst1q_f32(b_data + idx, b_lo);

            // Process next 4 pixels
            float32x4_t r_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ch0_16)));
            float32x4_t g_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ch1_16)));
            float32x4_t b_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(ch2_16)));

            r_hi = vmulq_f32(r_hi, v_scale);
            g_hi = vmulq_f32(g_hi, v_scale);
            b_hi = vmulq_f32(b_hi, v_scale);

            r_hi = vmulq_f32(vsubq_f32(r_hi, v_mean_r), v_std_inv_r);
            g_hi = vmulq_f32(vsubq_f32(g_hi, v_mean_g), v_std_inv_g);
            b_hi = vmulq_f32(vsubq_f32(b_hi, v_mean_b), v_std_inv_b);

            // Store next 4 pixels
            vst1q_f32(r_data + idx + 4, r_hi);
            vst1q_f32(g_data + idx + 4, g_hi);
            vst1q_f32(b_data + idx + 4, b_hi);
        }

        // Handle remaining pixels with scalar code
        for (; x < width; ++x) {
            const uint8_t* p = src_row + x * 3;
            int idx = base_idx + x;

            float r = (is_bgr ? p[2] : p[0]) * scale;
            float g = p[1] * scale;
            float b = (is_bgr ? p[0] : p[2]) * scale;

            r_data[idx] = (r - mean[0]) * std_inv[0];
            g_data[idx] = (g - mean[1]) * std_inv[1];
            b_data[idx] = (b - mean[2]) * std_inv[2];
        }
    }
#else
    // Scalar fallback
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        for (int x = 0; x < width; ++x) {
            const uint8_t* p = src_row + x * 3;
            int idx = y * width + x;

            float r = (is_bgr ? p[2] : p[0]) * scale;
            float g = p[1] * scale;
            float b = (is_bgr ? p[0] : p[2]) * scale;

            r_data[idx] = (r - mean[0]) * std_inv[0];
            g_data[idx] = (g - mean[1]) * std_inv[1];
            b_data[idx] = (b - mean[2]) * std_inv[2];
        }
    }
#endif

    return result;
}

// ============================================================================
// Crop Operations
// ============================================================================

Pi5Image extract_crop_with_padding(const Pi5Image& src,
                                   const Pi5Rect& crop_rect,
                                   const Pi5Scalar& pad_color) {
    if (src.empty()) {
        throw std::invalid_argument("Cannot crop from empty image");
    }

    // Create output image
    Pi5Image dst(crop_rect.width, crop_rect.height, src.format());
    dst.fill(pad_color);

    // Calculate overlap between crop_rect and source image
    int src_x_start = std::max(0, crop_rect.x);
    int src_y_start = std::max(0, crop_rect.y);
    int src_x_end = std::min(src.width(), crop_rect.x + crop_rect.width);
    int src_y_end = std::min(src.height(), crop_rect.y + crop_rect.height);

    // Calculate destination offset
    int dst_x_start = src_x_start - crop_rect.x;
    int dst_y_start = src_y_start - crop_rect.y;

    // Copy overlapping region
    int copy_width = src_x_end - src_x_start;
    int copy_height = src_y_end - src_y_start;

    if (copy_width > 0 && copy_height > 0) {
        int bpp = src.bytes_per_pixel();
        for (int y = 0; y < copy_height; ++y) {
            const uint8_t* src_row = src.row_ptr(src_y_start + y) + src_x_start * bpp;
            uint8_t* dst_row = dst.row_ptr(dst_y_start + y) + dst_x_start * bpp;
            std::memcpy(dst_row, src_row, copy_width * bpp);
        }
    }

    return dst;
}

Pi5Image extract_centered_crop(const Pi5Image& src,
                               const Pi5Point2f& center,
                               const Pi5Size& crop_size,
                               const Pi5Scalar& pad_color) {
    int x = static_cast<int>(center.x - crop_size.width / 2.0f);
    int y = static_cast<int>(center.y - crop_size.height / 2.0f);
    Pi5Rect crop_rect(x, y, crop_size.width, crop_size.height);
    return extract_crop_with_padding(src, crop_rect, pad_color);
}

// ============================================================================
// Utility Operations
// ============================================================================

void copy_to(const Pi5Image& src, Pi5Image& dst, int dst_x, int dst_y) {
    if (src.empty() || dst.empty()) return;

    if (src.format() != dst.format()) {
        throw std::invalid_argument("Source and destination must have same format");
    }

    // Calculate copy region
    int copy_width = std::min(src.width(), dst.width() - dst_x);
    int copy_height = std::min(src.height(), dst.height() - dst_y);

    if (copy_width <= 0 || copy_height <= 0) return;

    int bpp = src.bytes_per_pixel();
    for (int y = 0; y < copy_height; ++y) {
        const uint8_t* src_row = src.row_ptr(y);
        uint8_t* dst_row = dst.row_ptr(dst_y + y) + dst_x * bpp;
        std::memcpy(dst_row, src_row, copy_width * bpp);
    }
}

Pi5Image crop(const Pi5Image& src, const Pi5Rect& rect) {
    if (src.empty()) {
        return Pi5Image();
    }

    // Validate bounds
    if (rect.x < 0 || rect.y < 0 ||
        rect.x + rect.width > src.width() ||
        rect.y + rect.height > src.height()) {
        throw std::out_of_range("Crop rectangle out of bounds");
    }

    // Create ROI and clone it
    return src.roi(rect).clone();
}

float norm_l2(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float v : vec) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

float distance_l2(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have same size");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

}  // namespace pi5
