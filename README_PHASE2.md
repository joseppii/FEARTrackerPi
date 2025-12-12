# FEARTrackerPi Phase 2 - C++ Implementation

## Overview

Phase 2 provides a complete C++ implementation of FEARTracker optimized for Raspberry Pi 5, featuring:

- **Native C++ Performance**: Significant speed improvement over Python
- **ONNX Runtime Integration**: ARM64-optimized inference engine
- **OpenCV Integration**: Efficient image processing pipeline
- **Performance Monitoring**: Real-time FPS, temperature, and system metrics
- **Pi5 Optimizations**: ARM Cortex-A76 and VideoCore VII GPU support

## Build Requirements

### System Dependencies
```bash
sudo apt update
sudo apt install -y \
    cmake build-essential pkg-config \
    libopencv-dev \
    libonnxruntime-dev
```

### Hardware Requirements
- **Raspberry Pi 5** with Broadcom BCM2712 SoC
- **4GB+ RAM** recommended for optimal performance
- **Active cooling** recommended for sustained workloads
- **GPU memory**: Configure `gpu_mem=128` in `/boot/config.txt`

## Build Instructions

### Quick Build
```bash
# Build with default settings (Release mode)
./build.sh

# Build with debug symbols
./build.sh --debug

# Clean build
./build.sh --clean

# Custom build directory
./build.sh --build-dir custom_build --jobs 4
```

### Manual Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Outputs
- `feartracker_pi` - Main tracking application
- `feartracker_benchmark` - Performance testing tool
- `libfeartracker_lib.a` - Static library for integration

## Usage Examples

### Basic Tracking
```bash
./build/feartracker_pi \
    --input video.mp4 \
    --output tracked.mp4 \
    --bbox 100,50,80,120 \
    --template models/fear_net_template.onnx \
    --search models/fear_net_search.onnx
```

### Performance Monitoring
```bash
./build/feartracker_pi \
    --input assets/test.mp4 \
    --bbox 163,53,45,174 \
    --benchmark \
    --display
```

### Benchmarking
```bash
# Quick benchmark
./build/feartracker_benchmark

# Stress test (5 minutes)
./build/feartracker_benchmark --stress-test --stress-duration 5

# Custom benchmark
./build/feartracker_benchmark \
    --video custom_video.mp4 \
    --bbox 200,100,50,80 \
    --frames 500 \
    --output results.json
```

## Configuration Options

### Core Parameters
- `--input VIDEO` - Input video file
- `--output VIDEO` - Output video with tracking visualization
- `--bbox X,Y,W,H` - Initial bounding box (required)
- `--template MODEL` - Template ONNX model path
- `--search MODEL` - Search ONNX model path

### Performance Options
- `--benchmark` - Enable performance monitoring
- `--threads N` - Number of ONNX inference threads (default: 4)
- `--fps-limit N` - Limit processing speed to N FPS

### Output Options
- `--display` - Show real-time visualization
- `--save-frames` - Save individual frames to disk
- `--output-dir DIR` - Output directory for frames/logs

## Architecture Overview

### Core Classes

#### FEARTracker
Main tracking class that orchestrates the entire pipeline:
```cpp
FEARTracker tracker;
tracker.initialize(template_model, search_model);
tracker.start_tracking(frame, bbox);
TrackingResult result = tracker.update(frame);
```

#### ONNXInference
ONNX Runtime wrapper optimized for Pi5:
- Automatic provider selection (OpenCL GPU → CPU fallback)
- ARM64 optimizations (4 threads, optimized memory access)
- ImageNet normalization and preprocessing

#### ImageProcessor
Computer vision utilities:
- Template/search crop extraction
- Padding and resizing operations
- Bbox regression decoding
- Confidence penalty application

#### TrackingState
State management and history:
- Bounding box history tracking
- Confidence trend analysis
- Velocity and scale change computation
- Tracking quality assessment

#### PerformanceMonitor
Real-time system monitoring:
- FPS calculation and trending
- CPU usage and memory consumption
- Pi5 temperature monitoring
- Operation timing and profiling

### Performance Optimizations

#### ARM64 Cortex-A76 Optimizations
```cpp
// Compiler flags in CMakeLists.txt
-march=armv8.2-a+fp16+simd -mtune=cortex-a76
```

#### ONNX Runtime Configuration
```cpp
session_options.SetIntraOpNumThreads(4);  // Use all 4 cores
session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
```

#### VideoCore VII GPU Support
- Automatic OpenCL provider detection
- Fallback to optimized CPU execution
- Unified memory architecture utilization

## Performance Targets

### Expected Performance (Phase 2)
- **FPS**: 15-25 FPS (vs 8-12 FPS Python)
- **Initialization**: <2 seconds
- **Memory**: 200-300 MB peak usage
- **Temperature**: <70°C sustained with active cooling

### Benchmark Results Format
```json
{
  "performance": {
    "avg_fps": 20.5,
    "min_fps": 18.2,
    "max_fps": 23.1,
    "avg_confidence": 0.847,
    "initialization_time_ms": 1650
  },
  "thermal": {
    "avg_temperature_c": 62.5,
    "max_temperature_c": 68.2
  }
}
```

## Development Notes

### Code Organization
```
src/
├── fear_tracker.{h,cpp}      # Main tracker implementation
├── onnx_inference.{h,cpp}    # ONNX Runtime wrapper
├── image_processor.{h,cpp}   # Image processing utilities
├── tracking_state.{h,cpp}    # State management
├── performance_monitor.{h,cpp} # Performance monitoring
├── utils.{h,cpp}             # General utilities
├── main.cpp                  # Main application
└── benchmark.cpp             # Benchmark application
```

### Error Handling
- Exception-safe RAII design
- Graceful degradation on provider failures
- Comprehensive input validation
- Detailed error messages with context

### Memory Management
- RAII for automatic cleanup
- Smart pointers for dynamic allocation
- Zero-copy operations where possible
- Memory pool considerations for future optimization

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Missing OpenCV
sudo apt install libopencv-dev

# Missing ONNX Runtime
# Download ARM64 release from https://github.com/microsoft/onnxruntime/releases
# Extract and set CMAKE_PREFIX_PATH
```

#### Runtime Issues
```bash
# Model loading failures
ls -la models/*.onnx  # Verify models exist
file models/*.onnx    # Check file integrity

# Permission issues
chmod +x feartracker_pi feartracker_benchmark

# GPU provider failures (expected, will fallback to CPU)
# Check: dmesg | grep -i gpu
```

#### Performance Issues
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Should be "performance" for maximum speed

# Check thermal throttling
watch -n1 'vcgencmd measure_temp && vcgencmd measure_clock arm'

# Monitor resources during execution
htop  # CPU/memory usage
```

### Debugging
```bash
# Debug build with symbols
./build.sh --debug

# Run with GDB
gdb ./build/feartracker_pi
(gdb) run --input video.mp4 --bbox 100,50,80,120

# Valgrind memory checking (if available on ARM64)
valgrind --tool=memcheck ./build/feartracker_pi
```

## Next Steps

Phase 2 C++ implementation provides the foundation for:

- **Phase 3**: Custom ARM64 NEON implementations (replacing OpenCV)
- **Phase 4**: ONNX model quantization and GPU optimizations
- **Integration**: Embedding into robotics or IoT applications

The modular architecture allows selective optimization of bottleneck components while maintaining API compatibility.