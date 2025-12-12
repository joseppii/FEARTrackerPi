# Raspberry Pi 5 Hardware Optimizations Guide

## Overview

This document provides comprehensive details on Raspberry Pi 5 hardware-specific optimizations for the FEARTracker implementation. The Pi5 offers significant performance improvements through ARM64 NEON SIMD operations and VideoCore VII GPU acceleration.

## Raspberry Pi 5 Hardware Architecture

### CPU: BCM2712 (Broadcom)
- **Cores**: 4x ARM Cortex-A76 @ 2.4GHz (performance cores)
- **Architecture**: ARM64 (aarch64) only
- **SIMD**: ARM64 NEON (Advanced SIMD)
- **Cache**: 512KB L2 per core, 2MB shared L3
- **Memory Interface**: 128-bit LPDDR4X-4267

### GPU: VideoCore VII
- **Compute**: OpenGL ES 3.1, Vulkan 1.2, OpenCL 1.2
- **Memory**: Unified memory architecture (shared with CPU)
- **Precision**: Native FP16 support, INT8 acceleration
- **Bandwidth**: High-bandwidth unified memory access

### Memory System
- **Type**: LPDDR4X-4267 (unified CPU/GPU)
- **Capacity**: 4GB/8GB options
- **Bandwidth**: ~17GB/s theoretical peak
- **Architecture**: Unified memory eliminates GPU↔CPU transfers

---

## ARM64 NEON Optimizations

### Memory Alignment Requirements

#### 32-Byte Alignment for NEON Operations
```cpp
// Optimal NEON alignment
class Pi5OptimalImage {
private:
    alignas(32) uint8_t* primary_data_;     // 32-byte aligned for NEON
    alignas(32) float* inference_data_;     // ONNX inference data
    uint32_t primary_stride_;               // 64-byte cache alignment
};

// Memory allocation with proper alignment
void* aligned_alloc_neon(size_t size) {
    void* ptr;
    if (posix_memalign(&ptr, 32, size) != 0) {
        return nullptr;
    }
    return ptr;
}
```

#### Cache-Friendly Stride Calculation
```cpp
// Calculate stride for 64-byte cache line optimization
uint32_t calculate_optimal_stride(uint32_t width, uint32_t channels) {
    uint32_t bytes_per_row = width * channels;
    // Round up to next 64-byte boundary
    return (bytes_per_row + 63) & ~63;
}
```

### NEON-Optimized Image Operations

#### Color Space Conversion (BGR ↔ RGB)
```cpp
void convert_bgr_to_rgb_neon(uint8_t* data, uint32_t pixels) {
    const uint32_t vectors = pixels / 16;  // 16 pixels per NEON vector
    
    for (uint32_t i = 0; i < vectors; i++) {
        uint8x16x3_t bgr = vld3q_u8(data + i * 48);
        
        // Swap B and R channels
        uint8x16x3_t rgb;
        rgb.val[0] = bgr.val[2];  // R = B
        rgb.val[1] = bgr.val[1];  // G = G
        rgb.val[2] = bgr.val[0];  // B = R
        
        vst3q_u8(data + i * 48, rgb);
    }
    
    // Handle remaining pixels (scalar)
    handle_remaining_pixels_scalar(data + vectors * 48, 
                                  pixels - vectors * 16);
}
```

#### Bilinear Interpolation Resize
```cpp
void resize_bilinear_neon(const uint8_t* src, uint8_t* dst,
                         uint32_t src_w, uint32_t src_h,
                         uint32_t dst_w, uint32_t dst_h) {
    
    const float x_ratio = (float)src_w / dst_w;
    const float y_ratio = (float)src_h / dst_h;
    
    for (uint32_t y = 0; y < dst_h; y++) {
        const float src_y = y * y_ratio;
        const uint32_t y1 = (uint32_t)src_y;
        const uint32_t y2 = std::min(y1 + 1, src_h - 1);
        const float y_weight = src_y - y1;
        
        // Process 8 pixels at once with NEON
        for (uint32_t x = 0; x < dst_w; x += 8) {
            float32x4_t x_coords_1 = {x * x_ratio, (x+1) * x_ratio, 
                                     (x+2) * x_ratio, (x+3) * x_ratio};
            float32x4_t x_coords_2 = {(x+4) * x_ratio, (x+5) * x_ratio,
                                     (x+6) * x_ratio, (x+7) * x_ratio};
            
            // Convert to integer coordinates and perform interpolation
            uint32x4_t x1_coords_1 = vcvtq_u32_f32(x_coords_1);
            uint32x4_t x1_coords_2 = vcvtq_u32_f32(x_coords_2);
            
            // Vectorized bilinear interpolation implementation
            perform_bilinear_interpolation_neon(src, dst, x1_coords_1, x1_coords_2,
                                               y1, y2, y_weight, src_w);
        }
    }
}
```

#### ImageNet Normalization
```cpp
void normalize_imagenet_neon(const uint8_t* src, float* dst, uint32_t pixels) {
    // ImageNet normalization constants
    const float32x4_t mean = {0.485f, 0.456f, 0.406f, 0.0f};
    const float32x4_t std_inv = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f, 0.0f};
    const float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
    
    for (uint32_t i = 0; i < pixels; i += 4) {
        // Load 4 RGB pixels (12 bytes)
        uint8x16_t pixels_u8 = vld1q_u8(src + i * 3);
        
        // Convert to float and normalize
        float32x4_t r = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(pixels_u8)))));
        float32x4_t g = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(pixels_u8)))));
        float32x4_t b = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(pixels_u8)))));
        
        // Apply normalization: (pixel/255.0 - mean) / std
        r = vmulq_f32(vsubq_f32(vmulq_f32(r, scale), vdupq_lane_f32(vget_low_f32(mean), 0)),
                      vdupq_lane_f32(vget_low_f32(std_inv), 0));
        g = vmulq_f32(vsubq_f32(vmulq_f32(g, scale), vdupq_lane_f32(vget_low_f32(mean), 1)),
                      vdupq_lane_f32(vget_low_f32(std_inv), 1));
        b = vmulq_f32(vsubq_f32(vmulq_f32(b, scale), vdupq_lane_f32(vget_high_f32(mean), 0)),
                      vdupq_lane_f32(vget_high_f32(std_inv), 0));
        
        // Store normalized values
        vst1q_f32(dst + i * 3, r);
        vst1q_f32(dst + i * 3 + 4, g);
        vst1q_f32(dst + i * 3 + 8, b);
    }
}
```

### Custom Convolution Implementation
```cpp
void conv2d_neon_3x3(const float* input, const float* weights, float* output,
                     uint32_t width, uint32_t height, uint32_t channels) {
    
    for (uint32_t y = 1; y < height - 1; y++) {
        for (uint32_t x = 1; x < width - 1; x += 4) {  // Process 4 pixels at once
            
            float32x4_t acc = vdupq_n_f32(0.0f);
            
            // 3x3 convolution with NEON
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    const float* input_ptr = input + ((y + dy) * width + x + dx) * channels;
                    const float weight = weights[(dy + 1) * 3 + (dx + 1)];
                    
                    // Load 4 pixels and multiply by weight
                    float32x4_t pixels = vld1q_f32(input_ptr);
                    acc = vmlaq_n_f32(acc, pixels, weight);
                }
            }
            
            // Store result
            vst1q_f32(output + (y * width + x) * channels, acc);
        }
    }
}
```

---

## VideoCore VII GPU Optimizations

### OpenCL Configuration for VideoCore VII

#### Optimal OpenCL Setup
```cpp
class VideoCore7Provider {
private:
    cl_context context_;
    cl_command_queue queue_;
    cl_device_id device_;
    
public:
    bool initialize() {
        // Get VideoCore VII GPU device
        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, nullptr);
        
        cl_device_type device_type = CL_DEVICE_TYPE_GPU;
        clGetDeviceIDs(platform, device_type, 1, &device_, nullptr);
        
        // Create context with unified memory optimization
        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            CL_CONTEXT_MEMORY_INITIALIZE_KHR, CL_CONTEXT_MEMORY_INITIALIZE_LOCAL_KHR,
            0
        };
        
        context_ = clCreateContext(props, 1, &device_, nullptr, nullptr, nullptr);
        queue_ = clCreateCommandQueue(context_, device_, 
                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, nullptr);
        
        return true;
    }
};
```

### Mixed Precision Strategy

#### FP16 Operations for GPU
```cpp
// OpenCL kernel optimized for VideoCore VII FP16
const char* conv_fp16_kernel = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void conv2d_fp16(
    __global const half* input,
    __global const half* weights,
    __global half* output,
    int width, int height, int channels) {
    
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    
    if (gid_x >= width - 2 || gid_y >= height - 2) return;
    
    half4 acc = (half4)(0.0h);
    
    // 3x3 convolution optimized for VideoCore VII
    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            int input_idx = ((gid_y + dy) * width + gid_x + dx) * channels;
            int weight_idx = dy * 3 + dx;
            
            half4 pixel = vload4(input_idx / 4, input);
            half weight = weights[weight_idx];
            
            acc += pixel * weight;
        }
    }
    
    int output_idx = (gid_y * width + gid_x) * channels;
    vstore4(acc, output_idx / 4, output);
}
)";
```

#### Intelligent Operator Assignment
```cpp
class HybridInferenceEngine {
private:
    struct OperatorProfile {
        float cpu_time_ms;
        float gpu_time_ms;
        float memory_transfer_ms;
        bool prefers_gpu;
    };
    
    std::map<std::string, OperatorProfile> profiles_;
    
public:
    ExecutionProvider select_provider(const std::string& op_type, 
                                    const TensorShape& input_shape,
                                    float current_temp) {
        auto& profile = profiles_[op_type];
        
        // Thermal throttling consideration
        if (current_temp > 70.0f) {
            return ExecutionProvider::CPU;  // Reduce GPU load when hot
        }
        
        // For large tensors, consider memory transfer cost
        size_t tensor_bytes = input_shape.total_size() * sizeof(float);
        if (tensor_bytes > 1024 * 1024 && profile.memory_transfer_ms > 2.0f) {
            return ExecutionProvider::CPU;
        }
        
        return profile.prefers_gpu ? ExecutionProvider::GPU : ExecutionProvider::CPU;
    }
};
```

### Memory Coalescing Optimization
```cpp
// Optimize memory access patterns for VideoCore VII
__kernel void optimized_memory_access(
    __global const float4* input,   // Use float4 for coalesced access
    __global float4* output,
    int width, int height) {
    
    int gid = get_global_id(0);
    int total_pixels = width * height / 4;
    
    if (gid >= total_pixels) return;
    
    // Coalesced read - all work items in workgroup read consecutive memory
    float4 pixel = input[gid];
    
    // Process pixel (example: simple brightness adjustment)
    pixel = pixel * 1.2f;
    
    // Coalesced write
    output[gid] = pixel;
}
```

---

## Memory Management Optimizations

### Pi5ImagePool Implementation
```cpp
class Pi5ImagePool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        uint64_t last_used;
    };
    
    std::vector<MemoryBlock> pool_;
    std::mutex pool_mutex_;
    
public:
    void* allocate_aligned(size_t size, size_t alignment = 32) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        // Find suitable block in pool
        for (auto& block : pool_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                block.last_used = get_timestamp();
                return block.ptr;
            }
        }
        
        // Allocate new block
        void* ptr;
        if (posix_memalign(&ptr, alignment, size) == 0) {
            pool_.push_back({ptr, size, true, get_timestamp()});
            return ptr;
        }
        
        return nullptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        for (auto& block : pool_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                break;
            }
        }
    }
    
    void cleanup_unused(uint64_t max_age_ms = 30000) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        uint64_t current_time = get_timestamp();
        
        pool_.erase(
            std::remove_if(pool_.begin(), pool_.end(),
                [current_time, max_age_ms](const MemoryBlock& block) {
                    return !block.in_use && 
                           (current_time - block.last_used) > max_age_ms;
                }),
            pool_.end()
        );
    }
};
```

### Unified Memory Architecture Utilization
```cpp
// Leverage Pi5's unified memory for zero-copy GPU operations
class UnifiedMemoryManager {
private:
    cl_context opencl_context_;
    
public:
    cl_mem create_unified_buffer(size_t size, void* host_ptr = nullptr) {
        cl_int err;
        
        // Create buffer that can be accessed by both CPU and GPU
        cl_mem buffer = clCreateBuffer(
            opencl_context_,
            CL_MEM_READ_WRITE | (host_ptr ? CL_MEM_USE_HOST_PTR : CL_MEM_ALLOC_HOST_PTR),
            size,
            host_ptr,
            &err
        );
        
        return buffer;
    }
    
    void* map_for_cpu_access(cl_mem buffer, size_t size) {
        cl_int err;
        return clEnqueueMapBuffer(
            queue_, buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
            0, size, 0, nullptr, nullptr, &err
        );
    }
};
```

---

## Thermal Management

### Dynamic Performance Scaling
```cpp
class ThermalManager {
private:
    float current_temp_;
    float max_safe_temp_;
    PerformanceLevel current_level_;
    
public:
    enum class PerformanceLevel {
        FULL_PERFORMANCE,    // < 60°C
        REDUCED_GPU,         // 60-70°C
        CPU_ONLY,           // 70-80°C
        THROTTLED          // > 80°C
    };
    
    void update_thermal_state() {
        current_temp_ = read_cpu_temperature();
        
        if (current_temp_ > 80.0f) {
            current_level_ = PerformanceLevel::THROTTLED;
            set_cpu_frequency(1200000);  // 1.2GHz
        } else if (current_temp_ > 70.0f) {
            current_level_ = PerformanceLevel::CPU_ONLY;
            set_cpu_frequency(1800000);  // 1.8GHz
        } else if (current_temp_ > 60.0f) {
            current_level_ = PerformanceLevel::REDUCED_GPU;
            set_cpu_frequency(2000000);  // 2.0GHz
        } else {
            current_level_ = PerformanceLevel::FULL_PERFORMANCE;
            set_cpu_frequency(2400000);  // 2.4GHz max
        }
    }
    
    bool should_use_gpu() const {
        return current_level_ == PerformanceLevel::FULL_PERFORMANCE ||
               current_level_ == PerformanceLevel::REDUCED_GPU;
    }
    
private:
    float read_cpu_temperature() {
        std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
        if (temp_file.is_open()) {
            int temp_millidegree;
            temp_file >> temp_millidegree;
            return temp_millidegree / 1000.0f;
        }
        return 50.0f;  // Default safe temperature
    }
    
    void set_cpu_frequency(int frequency_hz) {
        for (int cpu = 0; cpu < 4; cpu++) {
            std::string freq_path = "/sys/devices/system/cpu/cpu" + 
                                   std::to_string(cpu) + 
                                   "/cpufreq/scaling_setspeed";
            std::ofstream freq_file(freq_path);
            if (freq_file.is_open()) {
                freq_file << frequency_hz;
            }
        }
    }
};
```

---

## Performance Monitoring

### Comprehensive Profiler
```cpp
class Pi5Profiler {
private:
    struct OperationStats {
        uint64_t total_time_us;
        uint64_t call_count;
        uint64_t memory_peak;
        uint64_t gpu_time_us;
    };
    
    std::map<std::string, OperationStats> stats_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    class ScopedTimer {
    private:
        Pi5Profiler* profiler_;
        std::string operation_name_;
        std::chrono::high_resolution_clock::time_point start_;
        
    public:
        ScopedTimer(Pi5Profiler* profiler, const std::string& name) 
            : profiler_(profiler), operation_name_(name) {
            start_ = std::chrono::high_resolution_clock::now();
        }
        
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            profiler_->record_operation(operation_name_, duration.count());
        }
    };
    
    void record_operation(const std::string& name, uint64_t time_us) {
        auto& stat = stats_[name];
        stat.total_time_us += time_us;
        stat.call_count++;
    }
    
    void print_performance_report() {
        printf("Performance Report:\n");
        printf("%-30s %10s %15s %15s\n", "Operation", "Calls", "Total (ms)", "Avg (ms)");
        printf("=================================================================\n");
        
        for (const auto& [name, stat] : stats_) {
            double total_ms = stat.total_time_us / 1000.0;
            double avg_ms = total_ms / stat.call_count;
            printf("%-30s %10lu %15.2f %15.2f\n", 
                   name.c_str(), stat.call_count, total_ms, avg_ms);
        }
    }
};

// Usage macro for easy profiling
#define PROFILE_SCOPE(profiler, name) \
    Pi5Profiler::ScopedTimer _timer(profiler, name)
```

---

## ONNX Runtime Provider Configuration

### Optimized Provider Setup
```cpp
std::unique_ptr<Ort::Session> create_optimized_session(const std::string& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FEARTrackerPi");
    Ort::SessionOptions session_options;
    
    // Enable optimizations
    session_options.SetIntraOpNumThreads(4);  // Use all 4 cores
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Configure execution providers
    std::vector<std::string> providers;
    
    // Try GPU provider first (VideoCore VII)
    try {
        OrtOpenCLProviderOptions opencl_options = {};
        opencl_options.device_type = 1;  // GPU
        opencl_options.enable_opencl_throttling = 1;
        session_options.AppendExecutionProvider_OpenCL(opencl_options);
        providers.push_back("OpenCLExecutionProvider");
    } catch (...) {
        // GPU provider failed, continue with CPU only
    }
    
    // CPU provider as fallback
    OrtCUDAProviderOptions cpu_options = {};
    session_options.AppendExecutionProvider_CPU(cpu_options);
    providers.push_back("CPUExecutionProvider");
    
    return std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
}
```

This comprehensive optimization guide provides the foundation for maximizing FEARTracker performance on Raspberry Pi 5 hardware through ARM64 NEON SIMD operations, VideoCore VII GPU acceleration, and intelligent thermal management.