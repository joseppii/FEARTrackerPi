# FEARTrackerPi

A Raspberry Pi 5 optimized implementation of the FEAR (Fast, Efficient, Accurate and Robust) visual object tracker, specifically adapted for ARM64 architecture and VideoCore VII GPU acceleration.

## Overview

FEARTrackerPi transforms the FEARTrackerAC Python ONNX tracker into a highly optimized C++ application designed for Raspberry Pi 5. The project follows a 4-phase development approach, progressively optimizing from basic Python compatibility to a fully optimized embedded tracking system.

## Hardware Requirements

### Minimum Requirements
- **Board**: Raspberry Pi 5A or 5B
- **SoC**: Broadcom BCM2712 (RK3588-compatible)
- **Memory**: 4GB LPDDR4X (8GB recommended)
- **Storage**: 16GB+ microSD (Class 10) or NVMe SSD
- **Power**: Official 27W USB-C power supply

### Optimal Configuration
- **Board**: Raspberry Pi 5B with active cooling
- **Memory**: 8GB LPDDR4X
- **Storage**: NVMe SSD via M.2 HAT
- **Cooling**: Active fan or heatsink for sustained performance
- **Camera**: Raspberry Pi Camera Module 3 or USB camera

## Project Phases

### Phase 1: Python Dependency Adaptation ✨
**Status**: Ready for implementation
**Duration**: 2 weeks
**Goal**: Make FEARTrackerAC run on Raspberry Pi 5

- ARM64 dependency adaptation
- Automated Pi5 environment setup
- Performance baseline establishment
- Basic tracking functionality validation

### Phase 2: C++ Implementation 🚧
**Status**: Architecture planned
**Duration**: 2 weeks  
**Goal**: Native C++ application with OpenCV

- ONNX Runtime C++ integration
- Real-time camera tracking
- Performance improvements over Python
- Cross-compilation support

### Phase 3: OpenCV Removal ⚡
**Status**: Optimization planned
**Duration**: 4 weeks
**Goal**: Lightweight custom implementation

- Pi5OptimalImage class with ARM64 NEON
- FFmpeg video I/O replacement
- Custom visualization functions
- 50-100MB binary size reduction

### Phase 4: ONNX Model Optimization 🎯
**Status**: Advanced optimization planned
**Duration**: 4 weeks
**Goal**: Maximum Pi5 hardware utilization

- VideoCore VII GPU acceleration
- INT8/FP16 quantization
- Custom ARM64 NEON operators
- 2-3x performance improvement

## Quick Start

### Phase 1: Python Setup

```bash
# Clone the repository
git clone <repository-url>
cd FEARTrackerPi

# Run automated setup (Pi5 only)
chmod +x setup_pi5.sh
./setup_pi5.sh

# Activate environment
source venv/bin/activate

# Run basic demo
python demo_pi5.py --video assets/test.mp4
```

### Phase 2: C++ Build

```bash
# Install build dependencies
sudo apt install cmake build-essential libonnxruntime-dev

# Build C++ version
mkdir build && cd build
cmake ..
make -j4

# Run C++ demo
./feartracker_pi --video ../assets/test.mp4
```

## Performance Targets

| Phase | Configuration | Resolution | FPS | Memory | Notes |
|-------|--------------|------------|-----|--------|-------|
| 1 | Python + ONNX | 480p | 8-12 | ~2GB | Baseline |
| 2 | C++ + OpenCV | 480p | 15-25 | ~1.5GB | Real-time capable |
| 3 | C++ + Custom | 480p | 20-30 | ~1GB | Lightweight |
| 4 | C++ + GPU | 480p | 25-40 | ~800MB | Fully optimized |

## Key Features

### Current (FEARTrackerAC Compatible)
- ONNX model inference with CPU execution
- Two-stage tracking (template + search)
- Video file processing
- Bounding box visualization

### Planned (Pi5 Optimized)
- **ARM64 NEON optimizations** for 4x faster image processing
- **VideoCore VII GPU acceleration** via OpenCL
- **Real-time camera tracking** with V4L2 integration
- **Thermal management** with dynamic performance scaling
- **Memory optimization** with custom allocation strategies
- **Quantized models** for faster inference (INT8/FP16)

## Architecture

### Core Components

```
FEARTrackerPi
├── tracker/          # Core tracking logic
│   ├── fear_tracker.cpp        # Main tracker class
│   ├── onnx_inference.cpp      # ONNX Runtime integration
│   └── tracking_state.cpp      # State management
├── image/            # Custom image processing
│   ├── pi5_image.cpp           # Optimized image class
│   ├── neon_ops.cpp            # ARM64 SIMD operations
│   └── memory_pool.cpp         # Efficient allocation
├── video/            # Video I/O replacement
│   ├── ffmpeg_reader.cpp       # Video decoding
│   ├── ffmpeg_writer.cpp       # Video encoding
│   └── v4l2_camera.cpp         # Camera interface
└── gpu/              # GPU acceleration
    ├── opencl_provider.cpp     # VideoCore VII interface
    └── gpu_memory.cpp          # GPU memory management
```

### Memory Layout

```cpp
class Pi5OptimalImage {
    alignas(32) uint8_t* primary_data;     // NEON-aligned storage
    uint32_t width, height, channels;
    uint32_t stride;                       // Cache-aligned stride
    static Pi5ImagePool memory_pool;       // Efficient allocation
};
```

## Dependencies

### Phase 1 (Python)
- Python 3.9+ with ARM64 support
- ONNX Runtime ARM64 wheels
- OpenCV Python ARM64 wheels
- NumPy with ARM64 optimizations

### Phase 2-3 (C++)
- CMake 3.16+
- GCC 9+ with ARM64 support
- ONNX Runtime C++ library
- FFmpeg development libraries

### Phase 4 (Optimized)
- VideoCore VII GPU drivers
- OpenCL development headers
- ARM64 NEON intrinsics support

## Installation

### Automated Setup (Recommended)
```bash
./setup_pi5.sh
```

### Manual Setup
```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev python3-pip cmake build-essential

# Install Python dependencies  
pip install -r requirements_pi5.txt

# For C++ development
sudo apt install libonnxruntime-dev libavcodec-dev libavformat-dev
```

## Usage Examples

### Basic Video Tracking
```bash
# Python version
python demo_pi5.py \
    --video input.mp4 \
    --output tracked_output.mp4 \
    --initial_bbox 100,50,200,150

# C++ version (Phase 2+)
./feartracker_pi \
    --input input.mp4 \
    --output tracked_output.mp4 \
    --bbox 100,50,200,150
```

### Real-time Camera Tracking
```bash
# C++ with camera support
./feartracker_pi \
    --camera 0 \
    --display \
    --bbox_from_click
```

### Performance Monitoring
```bash
# Run with performance profiling
./feartracker_pi \
    --video input.mp4 \
    --benchmark \
    --thermal_monitor \
    --memory_profile
```

## Development

### Building from Source
```bash
git clone <repository-url>
cd FEARTrackerPi

# Phase 1: Python development
./setup_pi5.sh
source venv/bin/activate

# Phase 2+: C++ development  
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### Testing
```bash
# Run test suite
./run_tests.sh

# Performance benchmarks
./benchmark_pi5.sh

# Accuracy validation
./validate_accuracy.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the coding standards (see docs/CODING_STANDARDS.md)
4. Test on actual Pi5 hardware
5. Submit a pull request

## Optimization Notes

### ARM64 Specific
- Use 32-byte aligned memory for NEON operations
- Prefer NHWC memory layout over NCHW for ARM efficiency
- Utilize Pi5's unified memory architecture
- Optimize for Cortex-A76 cache hierarchy

### VideoCore VII GPU
- Use FP16 precision where possible
- Implement proper memory coalescing
- Balance GPU vs CPU workload based on thermal state
- Leverage OpenCL for compute-intensive operations

## Troubleshooting

### Common Issues

**Low FPS Performance**:
- Check thermal throttling: `vcgencmd measure_temp`
- Verify active cooling is working
- Monitor CPU frequency: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`

**Memory Issues**:
- Increase GPU memory split: `gpu_mem=128` in `/boot/config.txt`
- Monitor memory usage: `free -h`
- Check for memory leaks with valgrind

**GPU Acceleration Not Working**:
- Verify OpenCL installation: `clinfo`
- Check VideoCore VII drivers: `lsmod | grep v3d`
- Enable GPU memory: `dtoverlay=vc4-kms-v3d` in `/boot/config.txt`

## License

This project is licensed under the same terms as the original FEAR tracker. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original FEAR tracker by Borsuk et al. (ECCV 2022)
- FEARTrackerAC adaptation
- Raspberry Pi Foundation for Pi5 hardware
- ARM for Cortex-A76/VideoCore VII optimization resources

## References

- [FEAR: Fast, Efficient, Accurate and Robust Visual Tracker (ECCV 2022)](https://arxiv.org/abs/2112.07957)
- [Raspberry Pi 5 Technical Specifications](https://www.raspberrypi.org/products/raspberry-pi-5/)
- [VideoCore VII GPU Programming Guide](https://www.raspberrypi.org/documentation/hardware/raspberrypi/gpu.md)
- [ARM64 NEON Optimization Guide](https://developer.arm.com/documentation/102159/latest/)