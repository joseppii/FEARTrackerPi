#!/bin/bash

# FEARTrackerPi Build Script for Raspberry Pi 5
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX="/usr/local"
NUM_JOBS=$(nproc)
CLEAN_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "FEARTrackerPi Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --clean              Clean build directory first"
            echo "  --jobs N             Number of parallel jobs (default: $(nproc))"
            echo "  --prefix PATH        Install prefix (default: /usr/local)"
            echo "  --build-dir DIR      Build directory (default: build)"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  FEARTrackerPi Build Script${NC}"
echo -e "${GREEN}========================================${NC}"

print_status "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Build directory: $BUILD_DIR"
echo "  Install prefix: $INSTALL_PREFIX"
echo "  Parallel jobs: $NUM_JOBS"
echo "  Clean build: $CLEAN_BUILD"
echo ""

# Check if we're on Raspberry Pi 5
print_status "Checking system compatibility..."
if [ ! -f /proc/device-tree/model ]; then
    print_warning "Cannot detect device model"
else
    DEVICE_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
    echo "  Device: $DEVICE_MODEL"
    
    if [[ ! "$DEVICE_MODEL" == *"Raspberry Pi 5"* ]]; then
        print_warning "This build script is optimized for Raspberry Pi 5"
    fi
fi

# Check architecture
ARCH=$(uname -m)
echo "  Architecture: $ARCH"
if [ "$ARCH" != "aarch64" ]; then
    print_warning "Expected ARM64 architecture, got: $ARCH"
fi

# Check required tools
print_status "Checking build dependencies..."
REQUIRED_TOOLS=("cmake" "make" "g++" "pkg-config")
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        MISSING_TOOLS+=("$tool")
    fi
done

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    print_error "Missing required tools: ${MISSING_TOOLS[*]}"
    print_status "Install missing tools with:"
    echo "  sudo apt install cmake build-essential pkg-config"
    exit 1
fi

# Check OpenCV
print_status "Checking OpenCV..."
if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    print_success "OpenCV $OPENCV_VERSION found"
else
    print_error "OpenCV not found"
    print_status "Install OpenCV with:"
    echo "  sudo apt install libopencv-dev"
    exit 1
fi

# Check ONNX Runtime
print_status "Checking ONNX Runtime..."
ONNX_FOUND=false

# Check common locations for ONNX Runtime
ONNX_SEARCH_PATHS=(
    "/usr/include/onnxruntime"
    "/usr/local/include/onnxruntime"
    "/opt/onnxruntime/include"
)

for path in "${ONNX_SEARCH_PATHS[@]}"; do
    if [ -f "$path/onnxruntime_cxx_api.h" ]; then
        print_success "ONNX Runtime headers found in $path"
        ONNX_FOUND=true
        break
    fi
done

if [ "$ONNX_FOUND" = false ]; then
    print_error "ONNX Runtime headers not found"
    print_status "Install ONNX Runtime or build from source"
    print_status "See: https://onnxruntime.ai/docs/install/"
    exit 1
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
print_status "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

print_success "CMake configuration completed"

# Build
print_status "Building FEARTrackerPi (using $NUM_JOBS jobs)..."
make -j"$NUM_JOBS"

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build completed successfully!"

# Show build results
print_status "Build artifacts:"
if [ -f "feartracker_pi" ]; then
    echo "  Main executable: $(pwd)/feartracker_pi"
    echo "    Size: $(du -h feartracker_pi | cut -f1)"
fi

if [ -f "feartracker_benchmark" ]; then
    echo "  Benchmark tool: $(pwd)/feartracker_benchmark"
    echo "    Size: $(du -h feartracker_benchmark | cut -f1)"
fi

if [ -f "libfeartracker_lib.a" ]; then
    echo "  Static library: $(pwd)/libfeartracker_lib.a"
    echo "    Size: $(du -h libfeartracker_lib.a | cut -f1)"
fi

# Optional installation
echo ""
print_status "To install to system directories, run:"
echo "  cd $BUILD_DIR && sudo make install"
echo ""
print_status "To test the build, run:"
echo "  cd $(dirname $BUILD_DIR)"
echo "  ./$BUILD_DIR/feartracker_pi --help"
echo "  ./$BUILD_DIR/feartracker_benchmark --help"

# Performance optimization suggestions
echo ""
print_status "Performance optimization tips:"
echo "  - Ensure GPU memory is allocated: gpu_mem=128 in /boot/config.txt"
echo "  - For maximum performance, use Release build (current: $BUILD_TYPE)"
echo "  - Monitor temperature during intensive workloads"
echo "  - Use active cooling for sustained performance"

print_success "FEARTrackerPi build completed!"
echo ""