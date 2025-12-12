# FEARTrackerPi Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the FEARTracker on Raspberry Pi 5, following the 4-phase development approach outlined in the development plan.

---

## Prerequisites

### Hardware Setup
1. **Raspberry Pi 5 (4GB or 8GB recommended)**
2. **MicroSD card (32GB+, Class 10) or NVMe SSD**
3. **Active cooling solution** (fan or heatsink)
4. **Camera module** (Pi Camera Module 3 or USB camera)
5. **HDMI display** for development and testing

### Software Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y git cmake build-essential pkg-config

# Install Python development tools
sudo apt install -y python3-dev python3-pip python3-venv

# Install system libraries for later phases
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
```

---

## Phase 1: Python Dependency Adaptation (Weeks 1-2)

### Step 1: Project Setup

1. **Clone source repositories**:
```bash
cd /home/pi
git clone <FEARTrackerAC-repo-url> FEARTrackerAC
git clone <FEARTrackerPi-repo-url> FEARTrackerPi
cd FEARTrackerPi
```

2. **Create Python environment**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 2: Dependency Analysis

1. **Analyze FEARTrackerAC requirements**:
```bash
cd ../FEARTrackerAC
cat requirements.txt
pip freeze > current_requirements.txt
```

2. **Create ARM64-compatible requirements**:
Create `requirements_pi5.txt`:
```txt
# Core ML libraries (ARM64 compatible)
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3

# ONNX Runtime (ARM64)
onnxruntime==1.15.1

# Computer Vision
opencv-python==4.8.0.74

# Configuration management
hydra-core==1.3.2
omegaconf==2.3.0

# Utilities
tqdm==4.65.0
matplotlib==3.7.1
Pillow==10.0.0

# Video processing
imageio==2.31.1
```

### Step 3: Environment Setup Script

Create `setup_pi5.sh`:
```bash
#!/bin/bash

# FEARTrackerPi Setup Script for Raspberry Pi 5
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}FEARTrackerPi Setup for Raspberry Pi 5${NC}"

# Detect hardware
if ! grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
    echo -e "${RED}Error: This script is designed for Raspberry Pi 5${NC}"
    exit 1
fi

echo -e "${YELLOW}Detected: $(cat /proc/device-tree/model)${NC}"

# Check architecture
if [ "$(uname -m)" != "aarch64" ]; then
    echo -e "${RED}Error: ARM64 architecture required${NC}"
    exit 1
fi

echo -e "${GREEN}✓ ARM64 architecture confirmed${NC}"

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    cmake build-essential pkg-config \
    libatlas-base-dev gfortran \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev libcanberra-gtk3-dev \
    libxvidcore-dev libx264-dev \
    libhdf5-dev libhdf5-serial-dev \
    libssl-dev libffi-dev

# Increase GPU memory split
echo -e "${YELLOW}Configuring GPU memory...${NC}"
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
fi

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install base packages
pip install --upgrade pip setuptools wheel

# Install PyTorch (ARM64)
echo -e "${YELLOW}Installing PyTorch for ARM64...${NC}"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements_pi5.txt

# Test installation
echo -e "${YELLOW}Testing installation...${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import onnxruntime as ort; print(f'ONNX Runtime version: {ort.__version__}')"

echo -e "${GREEN}✓ Setup complete!${NC}"
echo -e "${YELLOW}To activate the environment: source venv/bin/activate${NC}"
```

### Step 4: Code Adaptation

1. **Copy and adapt FEARTrackerAC source**:
```bash
# Copy source files
cp -r ../FEARTrackerAC/src ./src
cp -r ../FEARTrackerAC/models ./models
cp -r ../FEARTrackerAC/assets ./assets

# Create Pi5-specific demo
cp ../FEARTrackerAC/demo_video.py ./demo_pi5.py
```

2. **Modify ONNX Runtime configuration** in `src/inference_engine.py`:
```python
import onnxruntime as ort
import platform

class Pi5InferenceEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = self._create_session()
    
    def _create_session(self):
        # Configure providers for Pi5
        providers = ['CPUExecutionProvider']
        
        # Check for OpenCL provider (VideoCore VII)
        available_providers = ort.get_available_providers()
        if 'OpenCLExecutionProvider' in available_providers:
            providers.insert(0, 'OpenCLExecutionProvider')
            print("✓ Using VideoCore VII GPU acceleration")
        else:
            print("→ Using CPU execution only")
        
        # Session options for Pi5 optimization
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4  # Use all 4 cores
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return ort.InferenceSession(self.model_path, session_options, providers=providers)
```

### Step 5: Performance Baseline

Create `benchmark_pi5.py`:
```python
#!/usr/bin/env python3
import time
import psutil
import cv2
from src.fear_tracker import FEARTracker

class Pi5Benchmarker:
    def __init__(self):
        self.fps_history = []
        self.memory_history = []
        self.cpu_history = []
    
    def benchmark_video(self, video_path, bbox):
        tracker = FEARTracker()
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        total_time = 0
        
        # Read first frame and initialize
        ret, frame = cap.read()
        if not ret:
            return
        
        start_time = time.time()
        tracker.initialize(frame, bbox)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            result = tracker.update(frame)
            frame_time = time.time() - frame_start
            
            total_time += frame_time
            frame_count += 1
            
            # Record metrics
            fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_history.append(fps)
            self.memory_history.append(psutil.virtual_memory().percent)
            self.cpu_history.append(psutil.cpu_percent())
            
            if frame_count % 30 == 0:
                avg_fps = frame_count / total_time
                print(f"Processed {frame_count} frames, avg FPS: {avg_fps:.2f}")
        
        # Final metrics
        avg_fps = frame_count / total_time
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        
        print(f"\nBenchmark Results:")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Memory Usage: {avg_memory:.1f}%")
        print(f"Average CPU Usage: {avg_cpu:.1f}%")
        
        return {
            'fps': avg_fps,
            'memory': avg_memory,
            'cpu': avg_cpu,
            'frames': frame_count
        }

if __name__ == "__main__":
    benchmarker = Pi5Benchmarker()
    result = benchmarker.benchmark_video("assets/test.mp4", [163, 53, 45, 174])
```

---

## Phase 2: C++ Implementation (Weeks 3-4)

### Step 1: Build System Setup

Create `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(FEARTrackerPi VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for Pi5 Cortex-A76
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd -mtune=cortex-a76")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Find packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /usr/include/onnxruntime)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/lib/aarch64-linux-gnu)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(src)

# Source files
set(SOURCES
    src/fear_tracker.cpp
    src/onnx_inference.cpp
    src/image_processor.cpp
    src/tracking_state.cpp
    src/main.cpp
)

# Create executable
add_executable(feartracker_pi ${SOURCES})

# Link libraries
target_link_libraries(feartracker_pi 
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_LIB}
    pthread
)

# Install
install(TARGETS feartracker_pi DESTINATION bin)
```

### Step 2: Core Classes Implementation

Create `src/fear_tracker.h`:
```cpp
#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include "onnx_inference.h"
#include "image_processor.h"
#include "tracking_state.h"

class FEARTracker {
public:
    FEARTracker();
    ~FEARTracker();
    
    bool initialize(const std::string& model_path);
    bool start_tracking(const cv::Mat& frame, const cv::Rect& bbox);
    cv::Rect update(const cv::Mat& frame);
    
    float get_confidence() const;
    bool is_tracking() const;
    
private:
    std::unique_ptr<ONNXInference> inference_engine_;
    std::unique_ptr<ImageProcessor> preprocessor_;
    std::unique_ptr<TrackingState> state_;
    
    bool initialized_;
    bool tracking_;
    
    // Template features cached after initialization
    std::vector<float> template_features_;
};
```

Create `src/onnx_inference.h`:
```cpp
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ONNXInference {
public:
    ONNXInference();
    ~ONNXInference();
    
    bool load_model(const std::string& model_path);
    std::vector<float> extract_template_features(const cv::Mat& template_patch);
    cv::Rect process_search_region(const std::vector<float>& template_features,
                                  const cv::Mat& search_region,
                                  const cv::Point2f& search_center);
    
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    bool setup_providers();
    cv::Mat preprocess_for_inference(const cv::Mat& image, cv::Size target_size);
    std::vector<float> mat_to_tensor(const cv::Mat& mat);
};
```

### Step 3: ONNX Runtime Integration

Implement `src/onnx_inference.cpp`:
```cpp
#include "onnx_inference.h"
#include <iostream>
#include <algorithm>

ONNXInference::ONNXInference() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FEARTrackerPi");
    memory_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault));
}

bool ONNXInference::load_model(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        
        // Configure for Pi5 optimization
        session_options.SetIntraOpNumThreads(4);  // Use all 4 cores
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Setup execution providers
        if (!setup_providers()) {
            std::cout << "Warning: GPU provider setup failed, using CPU only" << std::endl;
        }
        
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // Get input/output metadata
        auto num_inputs = session_->GetInputCount();
        auto num_outputs = session_->GetOutputCount();
        
        for (size_t i = 0; i < num_inputs; i++) {
            input_names_.push_back(session_->GetInputName(i, Ort::AllocatorWithDefaultOptions()));
            input_shapes_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        
        for (size_t i = 0; i < num_outputs; i++) {
            output_names_.push_back(session_->GetOutputName(i, Ort::AllocatorWithDefaultOptions()));
            output_shapes_.push_back(session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

bool ONNXInference::setup_providers() {
    try {
        // Try to add OpenCL provider for VideoCore VII
        auto available_providers = Ort::GetAvailableProviders();
        
        for (const auto& provider : available_providers) {
            if (provider == "OpenCLExecutionProvider") {
                std::cout << "✓ VideoCore VII GPU acceleration available" << std::endl;
                return true;
            }
        }
        
        std::cout << "→ Using CPU execution only" << std::endl;
        return false;
    } catch (...) {
        return false;
    }
}

std::vector<float> ONNXInference::extract_template_features(const cv::Mat& template_patch) {
    // Preprocess template patch
    cv::Mat processed = preprocess_for_inference(template_patch, cv::Size(127, 127));
    std::vector<float> input_tensor = mat_to_tensor(processed);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 127, 127};
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        *memory_info_, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());
    
    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor_ort, 1,
        output_names_.data(), output_names_.size());
    
    // Extract features
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
}
```

---

## Phase 3: OpenCV Removal (Weeks 5-8)

### Step 1: Custom Image Class

Create `src/pi5_image.h`:
```cpp
#pragma once

#include <cstdint>
#include <memory>

class Pi5ImagePool;

class Pi5OptimalImage {
public:
    Pi5OptimalImage();
    Pi5OptimalImage(uint32_t width, uint32_t height, uint32_t channels);
    Pi5OptimalImage(const Pi5OptimalImage& other);
    Pi5OptimalImage& operator=(const Pi5OptimalImage& other);
    ~Pi5OptimalImage();
    
    // Accessors
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t channels() const { return channels_; }
    uint32_t stride() const { return primary_stride_; }
    
    // Memory access
    uint8_t* data() { return primary_data_; }
    const uint8_t* data() const { return primary_data_; }
    uint8_t* row_ptr(uint32_t y) { return primary_data_ + y * primary_stride_; }
    const uint8_t* row_ptr(uint32_t y) const { return primary_data_ + y * primary_stride_; }
    
    // NEON-optimized operations
    void convert_bgr_to_rgb();
    void resize_bilinear(uint32_t new_width, uint32_t new_height);
    void normalize_imagenet();
    Pi5OptimalImage crop(uint32_t x, uint32_t y, uint32_t w, uint32_t h) const;
    
    // ONNX inference preparation
    const float* get_inference_data() const;
    
private:
    alignas(32) uint8_t* primary_data_;        // 32-byte aligned for NEON
    mutable alignas(32) float* inference_data_; // Lazy conversion for ONNX
    uint32_t width_, height_, channels_;
    uint32_t primary_stride_;                  // 64-byte cache-aligned
    mutable bool inference_data_valid_;
    
    static Pi5ImagePool* memory_pool_;
    
    void allocate_memory();
    void free_memory();
    void update_inference_data() const;
    uint32_t calculate_optimal_stride(uint32_t width, uint32_t channels);
};
```

### Step 2: FFmpeg Video I/O

Create `src/video_reader.h`:
```cpp
#pragma once

#include "pi5_image.h"
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

class VideoReader {
public:
    VideoReader();
    ~VideoReader();
    
    bool open(const std::string& filename);
    bool read_frame(Pi5OptimalImage& frame);
    void close();
    
    // Properties
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    double get_fps() const { return fps_; }
    int get_total_frames() const { return total_frames_; }
    
private:
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    AVFrame* frame_;
    AVFrame* frame_rgb_;
    AVPacket* packet_;
    SwsContext* sws_ctx_;
    
    int video_stream_idx_;
    int width_, height_;
    double fps_;
    int total_frames_;
    
    bool setup_decoder();
    bool setup_scaler();
};
```

### Step 3: ARM64 NEON Operations

Create `src/neon_ops.h`:
```cpp
#pragma once

#include <arm_neon.h>
#include <cstdint>

class NeonOps {
public:
    // Color space conversion
    static void bgr_to_rgb_neon(uint8_t* data, uint32_t pixels);
    static void yuv420_to_rgb_neon(const uint8_t* y, const uint8_t* u, const uint8_t* v,
                                  uint8_t* rgb, uint32_t width, uint32_t height);
    
    // Resizing
    static void resize_bilinear_neon(const uint8_t* src, uint8_t* dst,
                                    uint32_t src_w, uint32_t src_h,
                                    uint32_t dst_w, uint32_t dst_h,
                                    uint32_t channels);
    
    // Normalization
    static void normalize_imagenet_neon(const uint8_t* src, float* dst, uint32_t pixels);
    
    // Utility functions
    static void copy_aligned_neon(const uint8_t* src, uint8_t* dst, uint32_t bytes);
    static void memset_aligned_neon(uint8_t* dst, uint8_t value, uint32_t bytes);
    
private:
    // Helper functions for complex operations
    static void resize_row_neon(const uint8_t* src, uint8_t* dst,
                               uint32_t src_width, uint32_t dst_width,
                               const float* x_weights, const uint32_t* x_indices);
};
```

---

## Phase 4: ONNX Model Optimization (Weeks 9-12)

### Step 1: Model Quantization

Create `tools/quantize_models.py`:
```python
#!/usr/bin/env python3

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import quantize_static, CalibrationDataReader
import numpy as np
import cv2
import os

class FEARCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset_path):
        self.dataset_path = calibration_dataset_path
        self.data_files = []
        self.current_index = 0
        
        # Load calibration data files
        for file in os.listdir(calibration_dataset_path):
            if file.endswith(('.jpg', '.png')):
                self.data_files.append(os.path.join(calibration_dataset_path, file))
    
    def get_next(self):
        if self.current_index >= len(self.data_files):
            return None
        
        # Load and preprocess image
        img_path = self.data_files[self.current_index]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (127, 127))
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor format (1, 3, 127, 127)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        self.current_index += 1
        return {'input': image}

def quantize_fear_models():
    models = [
        'models/fear_template.onnx',
        'models/fear_search.onnx'
    ]
    
    for model_path in models:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        base_name = os.path.splitext(model_path)[0]
        
        # Dynamic quantization (faster, good for CPU)
        print(f"Quantizing {model_path} (dynamic)...")
        dynamic_model = f"{base_name}_int8_dynamic.onnx"
        quantize_dynamic(
            model_input=model_path,
            model_output=dynamic_model,
            weight_type=QuantType.QInt8,
            activate_type=QuantType.QUInt8
        )
        
        # Static quantization (better accuracy, requires calibration data)
        if os.path.exists('calibration_data'):
            print(f"Quantizing {model_path} (static)...")
            static_model = f"{base_name}_int8_static.onnx"
            
            calibration_reader = FEARCalibrationDataReader('calibration_data')
            
            quantize_static(
                model_input=model_path,
                model_output=static_model,
                calibration_data_reader=calibration_reader,
                quant_format=QuantType.QInt8,
                per_channel=True  # Better accuracy for ARM64
            )

if __name__ == "__main__":
    quantize_fear_models()
```

### Step 2: Custom ARM64 Operators

Create `src/custom_operators.h`:
```cpp
#pragma once

#include <onnxruntime_cxx_api.h>
#include <arm_neon.h>

class ArmNeonConv2D {
public:
    static void conv2d_3x3_neon(const float* input, const float* weights, float* output,
                               int batch, int channels, int height, int width,
                               int out_channels);
    
    static void conv2d_1x1_neon(const float* input, const float* weights, float* output,
                               int batch, int channels, int height, int width,
                               int out_channels);
    
private:
    static void conv2d_3x3_kernel_neon(const float* input_ptr, const float* weight_ptr,
                                      float* output_ptr, int width);
};

class CustomOnnxOperators {
public:
    static void register_custom_ops(Ort::SessionOptions& session_options);
    
private:
    static void* CreateNeonConv2D(const OrtKernelInfo* info);
    static void ComputeNeonConv2D(void* op_kernel, OrtKernelContext* context);
    static void DestroyNeonConv2D(void* op_kernel);
};
```

### Step 3: Performance Monitoring Integration

Create `src/performance_monitor.h`:
```cpp
#pragma once

#include <chrono>
#include <unordered_map>
#include <fstream>
#include <vector>

class PerformanceMonitor {
public:
    struct Metrics {
        double avg_fps;
        double cpu_temp;
        double memory_usage_mb;
        double gpu_utilization;
        std::vector<double> frame_times;
    };
    
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    void start_frame();
    void end_frame();
    void record_temperature(double temp);
    void record_memory_usage(double mb);
    
    Metrics get_current_metrics() const;
    void save_metrics_to_csv(const std::string& filename) const;
    void print_summary() const;
    
private:
    std::chrono::high_resolution_clock::time_point frame_start_;
    std::vector<double> frame_times_;
    std::vector<double> temperatures_;
    std::vector<double> memory_usage_;
    
    mutable std::mutex metrics_mutex_;
    
    double read_cpu_temperature() const;
    double read_memory_usage() const;
};
```

### Step 4: Complete Integration

Create `src/main.cpp`:
```cpp
#include <iostream>
#include <string>
#include <getopt.h>
#include "fear_tracker.h"
#include "video_reader.h"
#include "camera_capture.h"
#include "performance_monitor.h"

struct AppConfig {
    std::string input_video;
    std::string output_video;
    std::string model_path = "models/fear_quantized.onnx";
    bool use_camera = false;
    int camera_id = 0;
    bool benchmark = false;
    bool display = false;
    cv::Rect initial_bbox;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -i, --input VIDEO      Input video file\n"
              << "  -o, --output VIDEO     Output video file\n"
              << "  -m, --model MODEL      ONNX model path\n"
              << "  -c, --camera ID        Use camera (default: 0)\n"
              << "  -b, --bbox X,Y,W,H     Initial bounding box\n"
              << "  --benchmark            Enable performance monitoring\n"
              << "  --display              Show tracking window\n"
              << "  -h, --help             Show this help\n";
}

AppConfig parse_arguments(int argc, char* argv[]) {
    AppConfig config;
    
    static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"model", required_argument, 0, 'm'},
        {"camera", optional_argument, 0, 'c'},
        {"bbox", required_argument, 0, 'b'},
        {"benchmark", no_argument, 0, 1000},
        {"display", no_argument, 0, 1001},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int c;
    while ((c = getopt_long(argc, argv, "i:o:m:c::b:h", long_options, nullptr)) != -1) {
        switch (c) {
            case 'i':
                config.input_video = optarg;
                break;
            case 'o':
                config.output_video = optarg;
                break;
            case 'm':
                config.model_path = optarg;
                break;
            case 'c':
                config.use_camera = true;
                if (optarg) config.camera_id = std::stoi(optarg);
                break;
            case 'b': {
                // Parse bbox: x,y,w,h
                int x, y, w, h;
                if (sscanf(optarg, "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
                    config.initial_bbox = cv::Rect(x, y, w, h);
                }
                break;
            }
            case 1000:
                config.benchmark = true;
                break;
            case 1001:
                config.display = true;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    AppConfig config = parse_arguments(argc, argv);
    
    // Initialize tracker
    FEARTracker tracker;
    if (!tracker.initialize(config.model_path)) {
        std::cerr << "Failed to initialize tracker" << std::endl;
        return 1;
    }
    
    // Initialize performance monitor
    std::unique_ptr<PerformanceMonitor> monitor;
    if (config.benchmark) {
        monitor = std::make_unique<PerformanceMonitor>();
    }
    
    if (config.use_camera) {
        // Camera tracking mode
        std::cout << "Starting camera tracking..." << std::endl;
        // Implementation for camera tracking
    } else if (!config.input_video.empty()) {
        // Video file processing
        VideoReader reader;
        if (!reader.open(config.input_video)) {
            std::cerr << "Failed to open video: " << config.input_video << std::endl;
            return 1;
        }
        
        Pi5OptimalImage frame;
        bool tracking_initialized = false;
        int frame_count = 0;
        
        std::cout << "Processing video: " << config.input_video << std::endl;
        
        while (reader.read_frame(frame)) {
            if (monitor) monitor->start_frame();
            
            if (!tracking_initialized && !config.initial_bbox.empty()) {
                tracker.start_tracking(frame, config.initial_bbox);
                tracking_initialized = true;
                std::cout << "Tracking initialized" << std::endl;
            }
            
            if (tracking_initialized) {
                cv::Rect result = tracker.update(frame);
                
                if (config.display) {
                    // Draw bounding box and display frame
                    // Implementation for visualization
                }
            }
            
            if (monitor) monitor->end_frame();
            frame_count++;
            
            if (frame_count % 30 == 0) {
                std::cout << "Processed " << frame_count << " frames" << std::endl;
            }
        }
        
        if (monitor) {
            monitor->print_summary();
            monitor->save_metrics_to_csv("benchmark_results.csv");
        }
        
        reader.close();
        std::cout << "Processing complete. Total frames: " << frame_count << std::endl;
    } else {
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
```

---

## Testing and Validation

### Performance Test Script
Create `test_performance.sh`:
```bash
#!/bin/bash

echo "FEARTrackerPi Performance Test Suite"
echo "===================================="

# Test video path
TEST_VIDEO="assets/test.mp4"
BBOX="163,53,45,174"

if [ ! -f "$TEST_VIDEO" ]; then
    echo "Error: Test video not found: $TEST_VIDEO"
    exit 1
fi

# Test Phase 1 (Python)
echo "Testing Phase 1 (Python)..."
source venv/bin/activate
python3 benchmark_pi5.py --video "$TEST_VIDEO" --bbox "$BBOX" > phase1_results.txt

# Test Phase 2 (C++ with OpenCV)
echo "Testing Phase 2 (C++)..."
./build/feartracker_pi --input "$TEST_VIDEO" --bbox "$BBOX" --benchmark > phase2_results.txt

# Test Phase 3 (Custom implementation)
echo "Testing Phase 3 (Custom)..."
./build/feartracker_pi_custom --input "$TEST_VIDEO" --bbox "$BBOX" --benchmark > phase3_results.txt

# Test Phase 4 (Optimized)
echo "Testing Phase 4 (Optimized)..."
./build/feartracker_pi_optimized --input "$TEST_VIDEO" --bbox "$BBOX" --benchmark > phase4_results.txt

echo "Performance test complete. Results saved to phase*_results.txt"
echo "Temperature monitoring:"
watch -n 1 'vcgencmd measure_temp && cat /sys/class/thermal/thermal_zone0/temp'
```

This comprehensive implementation guide provides detailed step-by-step instructions for each phase of the FEARTrackerPi development, from initial Python adaptation through final ONNX optimization, ensuring successful deployment on Raspberry Pi 5 hardware.