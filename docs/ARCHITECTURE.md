# FEARTrackerPi Architecture

## Overview

FEARTrackerPi implements a highly optimized object tracking system specifically designed for Raspberry Pi 5's ARM64 architecture and VideoCore VII GPU. The architecture progressively evolves through 4 phases, each building upon the previous while adding platform-specific optimizations.

## System Architecture Evolution

### Phase 1: Python Foundation
```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FEARTrackerAC     │───▶│  ARM64 Python    │───▶│  ONNX Runtime   │
│   (Source)          │    │  Environment     │    │  CPU Provider   │
└─────────────────────┘    └──────────────────┘    └─────────────────┘
                                    │
                           ┌──────────────────┐
                           │    OpenCV        │
                           │  (ARM64 wheels)  │
                           └──────────────────┘
```

### Phase 2: C++ Implementation
```
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   FEARTracker    │───▶│ ONNXInference   │───▶│  ONNX Runtime    │
│   (Main Class)   │    │   Engine        │    │    C++ API       │
└──────────────────┘    └─────────────────┘    └──────────────────┘
          │
          ▼
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  ImageProcessor  │───▶│    OpenCV       │───▶│   VideoCapture   │
│   (Preprocessing)│    │   C++ API       │    │   VideoWriter    │
└──────────────────┘    └─────────────────┘    └──────────────────┘
```

### Phase 3: Optimized Implementation
```
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   FEARTracker    │───▶│ ONNXInference   │───▶│  ONNX Runtime    │
│   (Optimized)    │    │   Engine        │    │    C++ API       │
└──────────────────┘    └─────────────────┘    └──────────────────┘
          │
          ▼
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Pi5OptimalImage  │───▶│  Custom Image   │───▶│   FFmpeg API     │
│  (Custom Class)  │    │   Processing    │    │   V4L2 Camera    │
└──────────────────┘    └─────────────────┘    └──────────────────┘
          │
          ▼
┌──────────────────┐
│  ARM64 NEON      │
│  Optimizations   │
└──────────────────┘
```

### Phase 4: GPU-Accelerated System
```
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   FEARTracker    │───▶│ ONNXInference   │───▶│ ONNX Runtime     │
│   (Final)        │    │   Engine        │    │ Multi-Provider   │
└──────────────────┘    └─────────────────┘    └──────────────────┘
          │                                             │
          │                                             ▼
          │                                    ┌──────────────────┐
          │                                    │ OpenCL Provider  │
          │                                    │ (VideoCore VII)  │
          │                                    └──────────────────┘
          ▼                                             │
┌──────────────────┐    ┌─────────────────┐            │
│ Pi5OptimalImage  │───▶│  ARM64 + NEON   │◀───────────┘
│ (GPU-aware)      │    │  Optimizations  │
└──────────────────┘    └─────────────────┘
```

## Core Classes and Components

### 1. FEARTracker (Main Class)

```cpp
class FEARTracker {
private:
    // Core components
    std::unique_ptr<ONNXInference> inference_engine_;
    std::unique_ptr<ImageProcessor> preprocessor_;
    std::unique_ptr<MemoryManager> memory_manager_;
    
    // Tracking state
    TrackingState current_state_;
    std::deque<TrackingState> state_history_;
    
    // Performance monitoring
    PerformanceMonitor perf_monitor_;
    ThermalManager thermal_manager_;
    
public:
    // Primary interface
    bool initialize(const Pi5OptimalImage& frame, const BoundingBox& initial_bbox);
    TrackingResult update(const Pi5OptimalImage& frame);
    
    // Configuration
    void set_tracking_parameters(const TrackingConfig& config);
    void enable_thermal_management(bool enable);
    
    // Performance monitoring
    PerformanceStats get_performance_stats() const;
    void reset_performance_counters();
};
```

### 2. Pi5OptimalImage (Optimized Image Class)

```cpp
class Pi5OptimalImage {
private:
    // Memory layout optimized for Pi5
    alignas(32) uint8_t* primary_data_;        // 32-byte aligned for NEON
    mutable alignas(32) float* inference_data_; // Lazy conversion for ONNX
    
    // Image metadata
    uint32_t width_, height_, channels_;
    uint32_t primary_stride_;                  // 64-byte cache-aligned
    PixelFormat format_;
    
    // Memory management
    size_t allocated_size_;
    bool owns_memory_;
    static Pi5ImagePool* memory_pool_;
    
    // Conversion state
    mutable bool inference_data_valid_;
    mutable ConversionParams last_conversion_;

public:
    // Pixel formats optimized for Pi5
    enum class PixelFormat {
        RGB888,      // Standard RGB, tightly packed
        RGBA8888,    // 4-channel, NEON-friendly
        BGR888,      // OpenCV compatibility
        YUV420,      // Camera input format
        GRAY8,       // Single channel
        FLOAT32_CHW  // ONNX inference format
    };
    
    // Construction and memory management
    Pi5OptimalImage(uint32_t width, uint32_t height, uint32_t channels);
    Pi5OptimalImage(const Pi5OptimalImage& other);
    Pi5OptimalImage& operator=(const Pi5OptimalImage& other);
    ~Pi5OptimalImage();
    
    // Fast pixel access (cache-friendly)
    inline uint8_t* row_ptr(int y) { 
        return primary_data_ + y * primary_stride_; 
    }
    inline const uint8_t* row_ptr(int y) const { 
        return primary_data_ + y * primary_stride_; 
    }
    
    // NEON-optimized operations
    void convert_bgr_to_rgb_neon();
    void resize_bilinear_neon(uint32_t new_width, uint32_t new_height);
    void normalize_imagenet_neon(const float mean[3], const float std[3]);
    void crop_and_resize_neon(const BoundingBox& roi, uint32_t target_width, uint32_t target_height);
    
    // ONNX inference support
    const float* get_inference_data() const;
    void prepare_for_inference(const InferenceConfig& config) const;
    
    // Format conversion
    void convert_to_format(PixelFormat new_format);
    Pi5OptimalImage extract_roi(const BoundingBox& roi) const;
    
    // Memory efficiency
    void release_inference_data() const;
    size_t memory_usage() const;
    
    // GPU integration (Phase 4)
    void* get_gpu_buffer() const;
    void sync_gpu_to_cpu() const;
    void sync_cpu_to_gpu() const;
};
```

### 3. ONNXInference (Inference Engine)

```cpp
class ONNXInference {
private:
    // ONNX Runtime sessions
    std::unique_ptr<Ort::Session> template_session_;
    std::unique_ptr<Ort::Session> search_session_;
    std::unique_ptr<Ort::Session> full_session_;
    
    // Execution environment
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    std::vector<const char*> execution_providers_;
    
    // Tensor management
    std::vector<Ort::Value> input_tensors_;
    std::vector<Ort::Value> output_tensors_;
    Ort::MemoryInfo memory_info_;
    
    // Model configuration
    ModelConfig config_;
    InferenceMode mode_;
    
    // Performance optimization
    TensorCache tensor_cache_;
    GPUMemoryPool gpu_memory_pool_;

public:
    enum class InferenceMode {
        FULL_MODEL,           // Single model processing
        TEMPLATE_SEARCH,      // Two-stage processing
        TEMPLATE_ONLY,        // Template extraction only
        SEARCH_ONLY          // Search with cached template
    };
    
    // Initialization
    bool load_models(const ModelPaths& paths);
    bool configure_execution_providers(const std::vector<std::string>& providers);
    void set_inference_mode(InferenceMode mode);
    
    // Template processing
    TemplateFeatures extract_template_features(const Pi5OptimalImage& template_image);
    void cache_template_features(const TemplateFeatures& features);
    
    // Search processing  
    TrackingResult process_search_region(const TemplateFeatures& template_features,
                                       const Pi5OptimalImage& search_image);
    
    // Full model processing
    TrackingResult process_full_model(const Pi5OptimalImage& template_image,
                                    const Pi5OptimalImage& search_image);
    
    // Performance monitoring
    InferenceStats get_inference_stats() const;
    void enable_profiling(bool enable);
    
    // GPU optimization (Phase 4)
    void enable_gpu_acceleration(bool enable);
    void optimize_for_thermal_throttling(bool enable);
};
```

### 4. Memory Management System

```cpp
class Pi5ImagePool {
private:
    // Memory pool categories
    static constexpr size_t MAX_SIZE_CATEGORIES = 16;
    static constexpr size_t ALIGNMENT = 32;  // ARM64 NEON alignment
    
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
    };
    
    // Pool storage
    std::array<std::vector<MemoryBlock>, MAX_SIZE_CATEGORIES> pools_;
    std::mutex pool_mutex_;
    
    // Statistics
    size_t total_allocated_;
    size_t total_freed_;
    size_t peak_usage_;

public:
    // Memory allocation
    void* acquire_aligned_memory(size_t size);
    void release_memory(void* ptr, size_t size);
    
    // Pool management
    void preallocate_common_sizes();
    void cleanup_unused_memory(std::chrono::seconds max_age);
    void defragment_pools();
    
    // Statistics
    MemoryStats get_memory_stats() const;
    void reset_statistics();
    
private:
    size_t get_size_category(size_t size) const;
    void* allocate_new_block(size_t size);
};

class MemoryManager {
private:
    Pi5ImagePool image_pool_;
    TensorPool tensor_pool_;
    GPUMemoryPool gpu_pool_;
    
public:
    // High-level allocation interface
    Pi5OptimalImage create_image(uint32_t width, uint32_t height, uint32_t channels);
    void release_image(Pi5OptimalImage&& image);
    
    // Tensor management
    Ort::Value create_tensor(const std::vector<int64_t>& shape, ONNXTensorElementDataType type);
    void release_tensor(Ort::Value&& tensor);
    
    // Memory optimization
    void optimize_for_sequence_length(size_t expected_frames);
    void enable_memory_pressure_callback(std::function<void(float)> callback);
    
    // GPU memory (Phase 4)
    void* allocate_gpu_buffer(size_t size);
    void free_gpu_buffer(void* ptr);
};
```

### 5. ARM64 NEON Optimizations

```cpp
namespace neon_ops {

// Image processing operations
void resize_bilinear_neon(const uint8_t* src, uint8_t* dst,
                         int src_width, int src_height, int src_stride,
                         int dst_width, int dst_height, int dst_stride,
                         int channels);

void convert_bgr_to_rgb_neon(const uint8_t* src, uint8_t* dst, int pixels);

void normalize_imagenet_neon(const uint8_t* src, float* dst, int pixels,
                           const float mean[3], const float std[3]);

void crop_and_resize_neon(const uint8_t* src, uint8_t* dst,
                         int src_width, int src_height, int src_stride,
                         const BoundingBox& crop_region,
                         int dst_width, int dst_height, int dst_stride);

// Mathematical operations
void vector_add_neon(const float* a, const float* b, float* result, int count);
void vector_multiply_neon(const float* a, const float* b, float* result, int count);
void matrix_multiply_neon(const float* a, const float* b, float* c,
                         int m, int n, int k);

// Utility functions
void memcpy_neon(void* dst, const void* src, size_t size);
void memset_neon(void* ptr, int value, size_t size);

} // namespace neon_ops
```

### 6. Video I/O System (Phase 3+)

```cpp
class VideoReader {
private:
    // FFmpeg components
    AVFormatContext* format_ctx_;
    AVCodecContext* video_codec_ctx_;
    AVCodecContext* audio_codec_ctx_;
    AVStream* video_stream_;
    SwsContext* sws_ctx_;
    
    // Frame management
    AVFrame* av_frame_;
    AVPacket* packet_;
    int video_stream_index_;
    
    // Configuration
    VideoConfig config_;
    bool is_open_;

public:
    struct VideoInfo {
        int width, height;
        double fps;
        int64_t duration_us;
        int64_t total_frames;
        AVPixelFormat pixel_format;
    };
    
    // File operations
    bool open(const std::string& filename);
    bool read_frame(Pi5OptimalImage& frame);
    bool seek_to_timestamp(int64_t timestamp_us);
    void close();
    
    // Information
    VideoInfo get_video_info() const;
    int64_t get_current_timestamp() const;
    
    // Configuration
    void set_pixel_format(Pi5OptimalImage::PixelFormat format);
    void enable_hardware_acceleration(bool enable);
};

class VideoWriter {
private:
    // FFmpeg components
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    AVStream* stream_;
    SwsContext* sws_ctx_;
    
    // Frame management
    AVFrame* av_frame_;
    AVPacket* packet_;
    
    // Configuration
    VideoWriterConfig config_;
    bool is_open_;

public:
    struct VideoWriterConfig {
        int width, height;
        double fps;
        int bitrate;
        AVCodecID codec_id;
        AVPixelFormat pixel_format;
    };
    
    // File operations
    bool open(const std::string& filename, const VideoWriterConfig& config);
    bool write_frame(const Pi5OptimalImage& frame);
    void close();
    
    // Configuration
    void set_quality(int quality_0_to_100);
    void enable_hardware_encoding(bool enable);
};

class CameraCapture {
private:
    // V4L2 components
    int camera_fd_;
    struct v4l2_format format_;
    struct v4l2_buffer* buffers_;
    void** buffer_ptrs_;
    int num_buffers_;
    
    // Configuration
    CameraConfig config_;
    bool is_streaming_;

public:
    struct CameraConfig {
        int device_id;
        int width, height;
        double fps;
        v4l2_pixel_format pixel_format;
        int num_buffers;
    };
    
    // Camera operations
    bool open(const CameraConfig& config);
    bool start_streaming();
    bool capture_frame(Pi5OptimalImage& frame);
    bool stop_streaming();
    void close();
    
    // Configuration
    std::vector<CameraConfig> enumerate_cameras();
    bool set_camera_control(int control_id, int value);
    
    // Information
    CameraConfig get_current_config() const;
    bool is_streaming() const { return is_streaming_; }
};
```

### 7. GPU Integration (Phase 4)

```cpp
class OpenCLProvider {
private:
    // OpenCL context
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue command_queue_;
    
    // Kernel management
    std::map<std::string, cl_kernel> kernels_;
    std::map<std::string, cl_program> programs_;
    
    // Memory management
    GPUMemoryPool memory_pool_;
    std::vector<cl_mem> buffer_cache_;

public:
    // Initialization
    bool initialize_videocore_vii();
    bool compile_kernels(const std::string& kernel_source_dir);
    
    // Memory management
    cl_mem allocate_buffer(size_t size, cl_mem_flags flags);
    void release_buffer(cl_mem buffer);
    
    // Kernel execution
    bool execute_kernel(const std::string& kernel_name,
                       const std::vector<cl_mem>& buffers,
                       const std::vector<size_t>& global_work_size,
                       const std::vector<size_t>& local_work_size = {});
    
    // Data transfer
    bool copy_to_device(cl_mem device_buffer, const void* host_data, size_t size);
    bool copy_to_host(void* host_data, cl_mem device_buffer, size_t size);
    
    // Performance monitoring
    double get_kernel_execution_time(const std::string& kernel_name);
    GPUMemoryStats get_memory_stats();
    
    // Thermal management
    void enable_thermal_throttling(bool enable);
    float get_gpu_temperature();
};

class GPUMemoryPool {
private:
    struct GPUBuffer {
        cl_mem buffer;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
    };
    
    std::vector<GPUBuffer> buffers_;
    cl_context context_;
    size_t total_allocated_;
    
public:
    // Buffer management
    cl_mem acquire_buffer(size_t size, cl_mem_flags flags);
    void release_buffer(cl_mem buffer);
    
    // Pool optimization
    void preallocate_common_sizes(const std::vector<size_t>& sizes);
    void cleanup_unused_buffers(std::chrono::seconds max_age);
    
    // Statistics
    GPUMemoryStats get_stats() const;
};
```

## Data Flow Architecture

### Template Extraction Flow
```
Pi5OptimalImage (Template)
    │
    ▼ (NEON preprocessing)
Pi5OptimalImage (Normalized)
    │
    ▼ (Memory layout conversion)
Float32 Tensor (CHW format)
    │
    ▼ (ONNX inference)
Template Features (Float32)
    │
    ▼ (Caching)
TemplateFeatures (Cached)
```

### Search Processing Flow
```
Pi5OptimalImage (Search Frame)
    │
    ▼ (ROI extraction + NEON resize)
Pi5OptimalImage (Search Region)
    │
    ▼ (NEON preprocessing)
Pi5OptimalImage (Normalized)
    │
    ▼ (Memory layout conversion)
Float32 Tensor (CHW format)
    │
    ▼ (ONNX inference + cached template)
Classification + Regression Maps
    │
    ▼ (Postprocessing)
BoundingBox (Final result)
```

### GPU-Accelerated Flow (Phase 4)
```
Pi5OptimalImage (Host Memory)
    │
    ▼ (GPU upload)
GPU Buffer (VideoCore VII)
    │
    ▼ (OpenCL kernels)
Processed GPU Buffer
    │
    ▼ (ONNX GPU inference)
Result GPU Buffer
    │
    ▼ (GPU download)
BoundingBox (Host Memory)
```

## Performance Optimization Strategies

### Memory Layout Optimization
1. **32-byte alignment**: All image data aligned for ARM64 NEON operations
2. **Cache-friendly stride**: 64-byte aligned row stride for optimal cache utilization
3. **Memory pooling**: Pre-allocated buffers to eliminate malloc/free overhead
4. **Lazy conversion**: On-demand format conversion between processing and inference

### CPU Optimization
1. **NEON vectorization**: 4x speedup for image processing operations
2. **Cache optimization**: Minimize cache misses through optimal data access patterns
3. **Thread affinity**: Pin threads to performance cores (Cortex-A76)
4. **Memory prefetching**: Explicit prefetch instructions for predictable data access

### GPU Optimization
1. **Memory coalescing**: Organize data access for optimal GPU memory bandwidth
2. **Workgroup sizing**: Optimize for VideoCore VII architecture
3. **Mixed precision**: FP16 for GPU operations, INT8 for memory-bound operations
4. **Pipeline optimization**: Overlap CPU preprocessing with GPU inference

### Thermal Management
1. **Dynamic scaling**: Reduce precision or resolution under thermal stress
2. **Workload balancing**: Shift workload between CPU and GPU based on temperature
3. **Performance monitoring**: Track FPS, temperature, and power consumption
4. **Graceful degradation**: Maintain functionality while reducing performance

## Testing and Validation Framework

### Unit Testing
```cpp
class Pi5ImageTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;
    
    // Test utilities
    Pi5OptimalImage create_test_image(int width, int height, int channels);
    void verify_neon_operation_correctness();
    void benchmark_neon_vs_scalar(const std::string& operation);
};

class ONNXInferenceTest : public ::testing::Test {
protected:
    // Inference accuracy tests
    void test_template_extraction_accuracy();
    void test_search_processing_accuracy();
    void test_end_to_end_tracking_accuracy();
    
    // Performance tests
    void benchmark_inference_performance();
    void test_gpu_vs_cpu_accuracy();
};
```

### Performance Testing
```cpp
class PerformanceBenchmark {
public:
    struct BenchmarkResults {
        double avg_fps;
        double min_fps, max_fps;
        double avg_latency_ms;
        size_t peak_memory_usage;
        double avg_cpu_usage;
        double avg_gpu_usage;
        double avg_temperature;
    };
    
    BenchmarkResults run_tracking_benchmark(const std::string& video_path,
                                          int num_iterations);
    BenchmarkResults run_thermal_stress_test(int duration_seconds);
    BenchmarkResults compare_phase_performance();
};
```

This architecture provides a comprehensive foundation for the FEARTrackerPi implementation, with clear separation of concerns, optimal memory management, and progressive optimization capabilities across all four development phases.