#include "ffmpeg_video.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#define USE_NEON_YUV 1
#else
#define USE_NEON_YUV 0
#endif

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

// ============================================================================
// NEON-optimized YUV420P to BGR24 conversion for Raspberry Pi 5
// ============================================================================

#if USE_NEON_YUV

// Convert YUV420P to BGR24 using NEON SIMD
// This is significantly faster than FFmpeg's swscale on Pi5
static void yuv420p_to_bgr24_neon(
    const uint8_t* y_plane, int y_stride,
    const uint8_t* u_plane, int u_stride,
    const uint8_t* v_plane, int v_stride,
    uint8_t* bgr_out, int bgr_stride,
    int width, int height)
{
    // YUV to RGB conversion coefficients (BT.601)
    // R = Y + 1.402 * (V - 128)
    // G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
    // B = Y + 1.772 * (U - 128)

    // Fixed point coefficients (Q8 format, multiply by 256)
    // 1.402 * 256 = 359
    // 0.344 * 256 = 88
    // 0.714 * 256 = 183
    // 1.772 * 256 = 454

    const int16x8_t v_359 = vdupq_n_s16(359);
    const int16x8_t v_88 = vdupq_n_s16(88);
    const int16x8_t v_183 = vdupq_n_s16(183);
    const int16x8_t v_454 = vdupq_n_s16(454);
    const int16x8_t v_128 = vdupq_n_s16(128);

    for (int y = 0; y < height; y += 2) {
        const uint8_t* y_row0 = y_plane + y * y_stride;
        const uint8_t* y_row1 = y_plane + (y + 1) * y_stride;
        const uint8_t* u_row = u_plane + (y / 2) * u_stride;
        const uint8_t* v_row = v_plane + (y / 2) * v_stride;
        uint8_t* bgr_row0 = bgr_out + y * bgr_stride;
        uint8_t* bgr_row1 = bgr_out + (y + 1) * bgr_stride;

        int x = 0;

        // Process 16 pixels (8 from each row) at a time
        for (; x <= width - 16; x += 16) {
            // Load 16 Y values from each row
            uint8x16_t y0_vec = vld1q_u8(y_row0 + x);
            uint8x16_t y1_vec = vld1q_u8(y_row1 + x);

            // Load 8 U and V values (shared between 2x2 pixel blocks)
            uint8x8_t u_vec = vld1_u8(u_row + x / 2);
            uint8x8_t v_vec = vld1_u8(v_row + x / 2);

            // Expand U and V to 16 values by duplicating (for 2x2 blocks)
            uint8x8x2_t u_zip = vzip_u8(u_vec, u_vec);
            uint8x8x2_t v_zip = vzip_u8(v_vec, v_vec);
            uint8x16_t u_expanded = vcombine_u8(u_zip.val[0], u_zip.val[1]);
            uint8x16_t v_expanded = vcombine_u8(v_zip.val[0], v_zip.val[1]);

            // Process first 8 pixels of row 0
            {
                int16x8_t y_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y0_vec)));
                int16x8_t u_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_expanded)));
                int16x8_t v_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_expanded)));

                // Center U and V around 0
                u_s16 = vsubq_s16(u_s16, v_128);
                v_s16 = vsubq_s16(v_s16, v_128);

                // Calculate R, G, B (in Q8 format)
                int16x8_t r = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_359, v_s16));
                int16x8_t g = vsubq_s16(vsubq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_88, u_s16)), vmulq_s16(v_183, v_s16));
                int16x8_t b = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_454, u_s16));

                // Shift back from Q8 format
                r = vshrq_n_s16(r, 8);
                g = vshrq_n_s16(g, 8);
                b = vshrq_n_s16(b, 8);

                // Clamp to [0, 255]
                uint8x8_t r_u8 = vqmovun_s16(r);
                uint8x8_t g_u8 = vqmovun_s16(g);
                uint8x8_t b_u8 = vqmovun_s16(b);

                // Store as BGR interleaved
                uint8x8x3_t bgr;
                bgr.val[0] = b_u8;
                bgr.val[1] = g_u8;
                bgr.val[2] = r_u8;
                vst3_u8(bgr_row0 + x * 3, bgr);
            }

            // Process second 8 pixels of row 0
            {
                int16x8_t y_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y0_vec)));
                int16x8_t u_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_expanded)));
                int16x8_t v_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_expanded)));

                u_s16 = vsubq_s16(u_s16, v_128);
                v_s16 = vsubq_s16(v_s16, v_128);

                int16x8_t r = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_359, v_s16));
                int16x8_t g = vsubq_s16(vsubq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_88, u_s16)), vmulq_s16(v_183, v_s16));
                int16x8_t b = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_454, u_s16));

                r = vshrq_n_s16(r, 8);
                g = vshrq_n_s16(g, 8);
                b = vshrq_n_s16(b, 8);

                uint8x8_t r_u8 = vqmovun_s16(r);
                uint8x8_t g_u8 = vqmovun_s16(g);
                uint8x8_t b_u8 = vqmovun_s16(b);

                uint8x8x3_t bgr;
                bgr.val[0] = b_u8;
                bgr.val[1] = g_u8;
                bgr.val[2] = r_u8;
                vst3_u8(bgr_row0 + (x + 8) * 3, bgr);
            }

            // Process first 8 pixels of row 1
            {
                int16x8_t y_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y1_vec)));
                int16x8_t u_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_expanded)));
                int16x8_t v_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_expanded)));

                u_s16 = vsubq_s16(u_s16, v_128);
                v_s16 = vsubq_s16(v_s16, v_128);

                int16x8_t r = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_359, v_s16));
                int16x8_t g = vsubq_s16(vsubq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_88, u_s16)), vmulq_s16(v_183, v_s16));
                int16x8_t b = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_454, u_s16));

                r = vshrq_n_s16(r, 8);
                g = vshrq_n_s16(g, 8);
                b = vshrq_n_s16(b, 8);

                uint8x8_t r_u8 = vqmovun_s16(r);
                uint8x8_t g_u8 = vqmovun_s16(g);
                uint8x8_t b_u8 = vqmovun_s16(b);

                uint8x8x3_t bgr;
                bgr.val[0] = b_u8;
                bgr.val[1] = g_u8;
                bgr.val[2] = r_u8;
                vst3_u8(bgr_row1 + x * 3, bgr);
            }

            // Process second 8 pixels of row 1
            {
                int16x8_t y_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y1_vec)));
                int16x8_t u_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_expanded)));
                int16x8_t v_s16 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_expanded)));

                u_s16 = vsubq_s16(u_s16, v_128);
                v_s16 = vsubq_s16(v_s16, v_128);

                int16x8_t r = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_359, v_s16));
                int16x8_t g = vsubq_s16(vsubq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_88, u_s16)), vmulq_s16(v_183, v_s16));
                int16x8_t b = vaddq_s16(vshlq_n_s16(y_s16, 8), vmulq_s16(v_454, u_s16));

                r = vshrq_n_s16(r, 8);
                g = vshrq_n_s16(g, 8);
                b = vshrq_n_s16(b, 8);

                uint8x8_t r_u8 = vqmovun_s16(r);
                uint8x8_t g_u8 = vqmovun_s16(g);
                uint8x8_t b_u8 = vqmovun_s16(b);

                uint8x8x3_t bgr;
                bgr.val[0] = b_u8;
                bgr.val[1] = g_u8;
                bgr.val[2] = r_u8;
                vst3_u8(bgr_row1 + (x + 8) * 3, bgr);
            }
        }

        // Handle remaining pixels with scalar code
        for (; x < width; x += 2) {
            int u_val = u_row[x / 2] - 128;
            int v_val = v_row[x / 2] - 128;

            for (int dx = 0; dx < 2 && (x + dx) < width; ++dx) {
                // Row 0
                int y0 = y_row0[x + dx];
                int r0 = y0 + ((359 * v_val + 128) >> 8);
                int g0 = y0 - ((88 * u_val + 183 * v_val + 128) >> 8);
                int b0 = y0 + ((454 * u_val + 128) >> 8);

                bgr_row0[(x + dx) * 3 + 0] = (uint8_t)(b0 < 0 ? 0 : (b0 > 255 ? 255 : b0));
                bgr_row0[(x + dx) * 3 + 1] = (uint8_t)(g0 < 0 ? 0 : (g0 > 255 ? 255 : g0));
                bgr_row0[(x + dx) * 3 + 2] = (uint8_t)(r0 < 0 ? 0 : (r0 > 255 ? 255 : r0));

                // Row 1
                if (y + 1 < height) {
                    int y1 = y_row1[x + dx];
                    int r1 = y1 + ((359 * v_val + 128) >> 8);
                    int g1 = y1 - ((88 * u_val + 183 * v_val + 128) >> 8);
                    int b1 = y1 + ((454 * u_val + 128) >> 8);

                    bgr_row1[(x + dx) * 3 + 0] = (uint8_t)(b1 < 0 ? 0 : (b1 > 255 ? 255 : b1));
                    bgr_row1[(x + dx) * 3 + 1] = (uint8_t)(g1 < 0 ? 0 : (g1 > 255 ? 255 : g1));
                    bgr_row1[(x + dx) * 3 + 2] = (uint8_t)(r1 < 0 ? 0 : (r1 > 255 ? 255 : r1));
                }
            }
        }
    }
}

#endif  // USE_NEON_YUV

// ============================================================================
// FFmpegVideoReader Implementation
// ============================================================================

struct FFmpegVideoReader::Impl {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    SwsContext* sws_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* bgr_frame = nullptr;
    AVPacket* packet = nullptr;

    int video_stream_idx = -1;
    int64_t current_frame_num = 0;
    int64_t total_frames = 0;
    double fps = 0.0;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
        if (bgr_frame) {
            av_frame_free(&bgr_frame);
            bgr_frame = nullptr;
        }
        if (frame) {
            av_frame_free(&frame);
            frame = nullptr;
        }
        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
            format_ctx = nullptr;
        }
        video_stream_idx = -1;
        current_frame_num = 0;
        total_frames = 0;
        fps = 0.0;
    }
};

FFmpegVideoReader::FFmpegVideoReader()
    : impl_(std::make_unique<Impl>())
{
}

FFmpegVideoReader::~FFmpegVideoReader() = default;

FFmpegVideoReader::FFmpegVideoReader(FFmpegVideoReader&& other) noexcept
    : impl_(std::move(other.impl_))
{
    other.impl_ = std::make_unique<Impl>();
}

FFmpegVideoReader& FFmpegVideoReader::operator=(FFmpegVideoReader&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        other.impl_ = std::make_unique<Impl>();
    }
    return *this;
}

bool FFmpegVideoReader::open(const std::string& filename) {
    close();

    // Open input file
    if (avformat_open_input(&impl_->format_ctx, filename.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "FFmpeg: Could not open file: " << filename << std::endl;
        return false;
    }

    // Find stream info
    if (avformat_find_stream_info(impl_->format_ctx, nullptr) < 0) {
        std::cerr << "FFmpeg: Could not find stream info" << std::endl;
        close();
        return false;
    }

    // Find video stream
    for (unsigned int i = 0; i < impl_->format_ctx->nb_streams; ++i) {
        if (impl_->format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            impl_->video_stream_idx = i;
            break;
        }
    }

    if (impl_->video_stream_idx < 0) {
        std::cerr << "FFmpeg: Could not find video stream" << std::endl;
        close();
        return false;
    }

    AVStream* video_stream = impl_->format_ctx->streams[impl_->video_stream_idx];
    AVCodecParameters* codecpar = video_stream->codecpar;

    // Find decoder
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "FFmpeg: Could not find decoder" << std::endl;
        close();
        return false;
    }

    // Allocate codec context
    impl_->codec_ctx = avcodec_alloc_context3(codec);
    if (!impl_->codec_ctx) {
        std::cerr << "FFmpeg: Could not allocate codec context" << std::endl;
        close();
        return false;
    }

    // Copy codec parameters
    if (avcodec_parameters_to_context(impl_->codec_ctx, codecpar) < 0) {
        std::cerr << "FFmpeg: Could not copy codec parameters" << std::endl;
        close();
        return false;
    }

    // Enable multi-threaded decoding for better performance on Pi5 (4 cores)
    impl_->codec_ctx->thread_count = 4;
    impl_->codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

    // Open codec
    if (avcodec_open2(impl_->codec_ctx, codec, nullptr) < 0) {
        std::cerr << "FFmpeg: Could not open codec" << std::endl;
        close();
        return false;
    }

    // Allocate frames
    impl_->frame = av_frame_alloc();
    impl_->bgr_frame = av_frame_alloc();
    impl_->packet = av_packet_alloc();

    if (!impl_->frame || !impl_->bgr_frame || !impl_->packet) {
        std::cerr << "FFmpeg: Could not allocate frames/packet" << std::endl;
        close();
        return false;
    }

    // Setup BGR frame
    impl_->bgr_frame->format = AV_PIX_FMT_BGR24;
    impl_->bgr_frame->width = impl_->codec_ctx->width;
    impl_->bgr_frame->height = impl_->codec_ctx->height;

    if (av_frame_get_buffer(impl_->bgr_frame, 32) < 0) {
        std::cerr << "FFmpeg: Could not allocate BGR frame buffer" << std::endl;
        close();
        return false;
    }

    // Create scaler context - use FAST_BILINEAR for speed since we're not actually scaling
    // SWS_FAST_BILINEAR is faster than SWS_BILINEAR for color conversion
    impl_->sws_ctx = sws_getContext(
        impl_->codec_ctx->width, impl_->codec_ctx->height, impl_->codec_ctx->pix_fmt,
        impl_->codec_ctx->width, impl_->codec_ctx->height, AV_PIX_FMT_BGR24,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    if (!impl_->sws_ctx) {
        std::cerr << "FFmpeg: Could not create scaler context" << std::endl;
        close();
        return false;
    }

    // Calculate FPS
    AVRational frame_rate = av_guess_frame_rate(impl_->format_ctx, video_stream, nullptr);
    impl_->fps = (frame_rate.den > 0) ? static_cast<double>(frame_rate.num) / frame_rate.den : 30.0;

    // Calculate total frames
    if (video_stream->nb_frames > 0) {
        impl_->total_frames = video_stream->nb_frames;
    } else if (video_stream->duration > 0) {
        double duration_sec = video_stream->duration * av_q2d(video_stream->time_base);
        impl_->total_frames = static_cast<int64_t>(duration_sec * impl_->fps);
    } else if (impl_->format_ctx->duration > 0) {
        double duration_sec = impl_->format_ctx->duration / static_cast<double>(AV_TIME_BASE);
        impl_->total_frames = static_cast<int64_t>(duration_sec * impl_->fps);
    }

    impl_->current_frame_num = 0;
    return true;
}

bool FFmpegVideoReader::is_open() const {
    return impl_->format_ctx != nullptr && impl_->codec_ctx != nullptr;
}

void FFmpegVideoReader::close() {
    impl_->cleanup();
}

bool FFmpegVideoReader::read(Pi5Image& frame) {
    if (!is_open()) {
        return false;
    }

    while (av_read_frame(impl_->format_ctx, impl_->packet) >= 0) {
        if (impl_->packet->stream_index == impl_->video_stream_idx) {
            // Send packet to decoder
            int ret = avcodec_send_packet(impl_->codec_ctx, impl_->packet);
            av_packet_unref(impl_->packet);

            if (ret < 0) {
                continue;
            }

            // Receive frame from decoder
            ret = avcodec_receive_frame(impl_->codec_ctx, impl_->frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                continue;
            } else if (ret < 0) {
                return false;
            }

            int w = impl_->codec_ctx->width;
            int h = impl_->codec_ctx->height;

            // Reuse existing buffer if correct size
            if (frame.empty() || frame.width() != w || frame.height() != h ||
                frame.format() != Pi5PixelFormat::BGR8) {
                frame = Pi5Image(w, h, Pi5PixelFormat::BGR8);
            }

#if USE_NEON_YUV
            // Use our NEON-optimized YUV420P to BGR24 converter
            if (impl_->frame->format == AV_PIX_FMT_YUV420P ||
                impl_->frame->format == AV_PIX_FMT_YUVJ420P) {
                yuv420p_to_bgr24_neon(
                    impl_->frame->data[0], impl_->frame->linesize[0],  // Y plane
                    impl_->frame->data[1], impl_->frame->linesize[1],  // U plane
                    impl_->frame->data[2], impl_->frame->linesize[2],  // V plane
                    frame.data(), frame.stride(),
                    w, h);
            } else
#endif
            {
                // Fallback to swscale for other formats
                sws_scale(impl_->sws_ctx,
                          impl_->frame->data, impl_->frame->linesize,
                          0, h,
                          impl_->bgr_frame->data, impl_->bgr_frame->linesize);

                // Copy from bgr_frame to Pi5Image
                const int src_stride = impl_->bgr_frame->linesize[0];
                const int dst_stride = frame.stride();
                const int row_bytes = w * 3;

                if (src_stride == row_bytes && dst_stride == row_bytes) {
                    std::memcpy(frame.data(), impl_->bgr_frame->data[0], row_bytes * h);
                } else {
                    const uint8_t* src = impl_->bgr_frame->data[0];
                    uint8_t* dst = frame.data();
                    for (int y = 0; y < h; ++y) {
                        std::memcpy(dst, src, row_bytes);
                        src += src_stride;
                        dst += dst_stride;
                    }
                }
            }

            impl_->current_frame_num++;
            return true;
        }
        av_packet_unref(impl_->packet);
    }

    return false;
}

int FFmpegVideoReader::width() const {
    return impl_->codec_ctx ? impl_->codec_ctx->width : 0;
}

int FFmpegVideoReader::height() const {
    return impl_->codec_ctx ? impl_->codec_ctx->height : 0;
}

double FFmpegVideoReader::fps() const {
    return impl_->fps;
}

int64_t FFmpegVideoReader::frame_count() const {
    return impl_->total_frames;
}

int64_t FFmpegVideoReader::current_frame() const {
    return impl_->current_frame_num;
}

bool FFmpegVideoReader::seek(int64_t frame_number) {
    if (!is_open() || frame_number < 0) {
        return false;
    }

    AVStream* video_stream = impl_->format_ctx->streams[impl_->video_stream_idx];

    // Convert frame number to timestamp
    int64_t timestamp = av_rescale_q(
        frame_number,
        av_make_q(1, static_cast<int>(impl_->fps)),
        video_stream->time_base);

    // Seek to keyframe before target
    int ret = av_seek_frame(impl_->format_ctx, impl_->video_stream_idx,
                            timestamp, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        return false;
    }

    // Flush codec buffers
    avcodec_flush_buffers(impl_->codec_ctx);

    // Read frames until we reach the target
    Pi5Image dummy;
    impl_->current_frame_num = 0;  // Reset counter

    while (impl_->current_frame_num < frame_number) {
        if (!read(dummy)) {
            return false;
        }
    }

    return true;
}

double FFmpegVideoReader::duration() const {
    if (impl_->fps > 0 && impl_->total_frames > 0) {
        return impl_->total_frames / impl_->fps;
    }
    if (impl_->format_ctx) {
        return impl_->format_ctx->duration / static_cast<double>(AV_TIME_BASE);
    }
    return 0.0;
}

// ============================================================================
// FFmpegVideoWriter Implementation
// ============================================================================

struct FFmpegVideoWriter::Impl {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    SwsContext* sws_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    AVStream* video_stream = nullptr;

    int width_ = 0;
    int height_ = 0;
    double fps_ = 0.0;
    int64_t frames_written_ = 0;
    int64_t pts_ = 0;

    ~Impl() {
        cleanup();
    }

    void cleanup() {
        // Flush encoder
        if (codec_ctx && format_ctx) {
            avcodec_send_frame(codec_ctx, nullptr);
            while (true) {
                int ret = avcodec_receive_packet(codec_ctx, packet);
                if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) break;
                if (ret < 0) break;
                av_interleaved_write_frame(format_ctx, packet);
                av_packet_unref(packet);
            }
        }

        // Write trailer
        if (format_ctx && format_ctx->pb) {
            av_write_trailer(format_ctx);
        }

        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
        if (frame) {
            av_frame_free(&frame);
            frame = nullptr;
        }
        if (sws_ctx) {
            sws_freeContext(sws_ctx);
            sws_ctx = nullptr;
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
        if (format_ctx) {
            if (format_ctx->pb) {
                avio_closep(&format_ctx->pb);
            }
            avformat_free_context(format_ctx);
            format_ctx = nullptr;
        }
        video_stream = nullptr;
        width_ = 0;
        height_ = 0;
        fps_ = 0.0;
        frames_written_ = 0;
        pts_ = 0;
    }
};

FFmpegVideoWriter::FFmpegVideoWriter()
    : impl_(std::make_unique<Impl>())
{
}

FFmpegVideoWriter::~FFmpegVideoWriter() = default;

FFmpegVideoWriter::FFmpegVideoWriter(FFmpegVideoWriter&& other) noexcept
    : impl_(std::move(other.impl_))
{
    other.impl_ = std::make_unique<Impl>();
}

FFmpegVideoWriter& FFmpegVideoWriter::operator=(FFmpegVideoWriter&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        other.impl_ = std::make_unique<Impl>();
    }
    return *this;
}

bool FFmpegVideoWriter::open(const std::string& filename, int width, int height,
                             double fps, const std::string& codec_name) {
    close();

    impl_->width_ = width;
    impl_->height_ = height;
    impl_->fps_ = fps;

    // Allocate output format context
    int ret = avformat_alloc_output_context2(&impl_->format_ctx, nullptr, nullptr, filename.c_str());
    if (ret < 0 || !impl_->format_ctx) {
        std::cerr << "FFmpeg: Could not create output context" << std::endl;
        return false;
    }

    // Find encoder
    AVCodecID codec_id = AV_CODEC_ID_H264;
    if (codec_name == "h265" || codec_name == "hevc") {
        codec_id = AV_CODEC_ID_HEVC;
    } else if (codec_name == "mjpeg") {
        codec_id = AV_CODEC_ID_MJPEG;
    } else if (codec_name == "mpeg4") {
        codec_id = AV_CODEC_ID_MPEG4;
    }

    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec) {
        std::cerr << "FFmpeg: Could not find encoder for " << codec_name << std::endl;
        close();
        return false;
    }

    // Create stream
    impl_->video_stream = avformat_new_stream(impl_->format_ctx, nullptr);
    if (!impl_->video_stream) {
        std::cerr << "FFmpeg: Could not create video stream" << std::endl;
        close();
        return false;
    }

    // Allocate codec context
    impl_->codec_ctx = avcodec_alloc_context3(codec);
    if (!impl_->codec_ctx) {
        std::cerr << "FFmpeg: Could not allocate codec context" << std::endl;
        close();
        return false;
    }

    // Set codec parameters
    impl_->codec_ctx->width = width;
    impl_->codec_ctx->height = height;
    impl_->codec_ctx->time_base = av_make_q(1, static_cast<int>(fps));
    impl_->codec_ctx->framerate = av_make_q(static_cast<int>(fps), 1);
    impl_->codec_ctx->gop_size = 12;
    impl_->codec_ctx->max_b_frames = 2;
    impl_->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    // Set bitrate for good quality
    impl_->codec_ctx->bit_rate = width * height * 4;  // Reasonable quality

    if (impl_->format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        impl_->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Open codec
    AVDictionary* opts = nullptr;
    if (codec_id == AV_CODEC_ID_H264) {
        av_dict_set(&opts, "preset", "medium", 0);
        av_dict_set(&opts, "crf", "23", 0);
    }

    ret = avcodec_open2(impl_->codec_ctx, codec, &opts);
    av_dict_free(&opts);

    if (ret < 0) {
        std::cerr << "FFmpeg: Could not open codec" << std::endl;
        close();
        return false;
    }

    // Copy codec parameters to stream
    ret = avcodec_parameters_from_context(impl_->video_stream->codecpar, impl_->codec_ctx);
    if (ret < 0) {
        std::cerr << "FFmpeg: Could not copy codec parameters" << std::endl;
        close();
        return false;
    }

    impl_->video_stream->time_base = impl_->codec_ctx->time_base;

    // Open output file
    if (!(impl_->format_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&impl_->format_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            std::cerr << "FFmpeg: Could not open output file: " << filename << std::endl;
            close();
            return false;
        }
    }

    // Write header
    ret = avformat_write_header(impl_->format_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "FFmpeg: Could not write header" << std::endl;
        close();
        return false;
    }

    // Allocate frame
    impl_->frame = av_frame_alloc();
    impl_->frame->format = impl_->codec_ctx->pix_fmt;
    impl_->frame->width = width;
    impl_->frame->height = height;

    ret = av_frame_get_buffer(impl_->frame, 32);
    if (ret < 0) {
        std::cerr << "FFmpeg: Could not allocate frame buffer" << std::endl;
        close();
        return false;
    }

    // Allocate packet
    impl_->packet = av_packet_alloc();

    // Create scaler (BGR24 -> YUV420P)
    impl_->sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    if (!impl_->sws_ctx) {
        std::cerr << "FFmpeg: Could not create scaler context" << std::endl;
        close();
        return false;
    }

    return true;
}

bool FFmpegVideoWriter::is_open() const {
    return impl_->format_ctx != nullptr && impl_->codec_ctx != nullptr;
}

void FFmpegVideoWriter::close() {
    impl_->cleanup();
}

bool FFmpegVideoWriter::write(const Pi5Image& frame) {
    if (!is_open()) {
        return false;
    }

    if (frame.empty() || frame.width() != impl_->width_ || frame.height() != impl_->height_) {
        std::cerr << "FFmpeg: Invalid frame dimensions" << std::endl;
        return false;
    }

    if (frame.format() != Pi5PixelFormat::BGR8) {
        std::cerr << "FFmpeg: Frame must be BGR8 format" << std::endl;
        return false;
    }

    // Make frame writable
    int ret = av_frame_make_writable(impl_->frame);
    if (ret < 0) {
        return false;
    }

    // Convert BGR24 to YUV420P
    const uint8_t* src_data[1] = { frame.data() };
    int src_linesize[1] = { frame.stride() };

    sws_scale(impl_->sws_ctx,
              src_data, src_linesize,
              0, impl_->height_,
              impl_->frame->data, impl_->frame->linesize);

    impl_->frame->pts = impl_->pts_++;

    // Send frame to encoder
    ret = avcodec_send_frame(impl_->codec_ctx, impl_->frame);
    if (ret < 0) {
        return false;
    }

    // Receive encoded packets
    while (ret >= 0) {
        ret = avcodec_receive_packet(impl_->codec_ctx, impl_->packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            return false;
        }

        // Rescale timestamps
        av_packet_rescale_ts(impl_->packet, impl_->codec_ctx->time_base,
                             impl_->video_stream->time_base);
        impl_->packet->stream_index = impl_->video_stream->index;

        // Write packet
        ret = av_interleaved_write_frame(impl_->format_ctx, impl_->packet);
        av_packet_unref(impl_->packet);

        if (ret < 0) {
            return false;
        }
    }

    impl_->frames_written_++;
    return true;
}

int FFmpegVideoWriter::width() const {
    return impl_->width_;
}

int FFmpegVideoWriter::height() const {
    return impl_->height_;
}

double FFmpegVideoWriter::fps() const {
    return impl_->fps_;
}

int64_t FFmpegVideoWriter::frames_written() const {
    return impl_->frames_written_;
}
