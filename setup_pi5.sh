#!/bin/bash

# FEARTrackerPi Setup Script for Raspberry Pi 5
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  FEARTrackerPi Setup for Raspberry Pi 5${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to print colored status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Detect hardware
print_status "Detecting hardware..."
if [ ! -f /proc/device-tree/model ]; then
    print_error "Cannot detect device model"
    exit 1
fi

DEVICE_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
if [[ ! "$DEVICE_MODEL" == *"Raspberry Pi 5"* ]]; then
    print_error "This script is designed for Raspberry Pi 5"
    print_error "Detected: $DEVICE_MODEL"
    exit 1
fi

print_success "Detected: $DEVICE_MODEL"

# Check architecture
print_status "Checking architecture..."
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    print_error "ARM64 architecture required, got: $ARCH"
    exit 1
fi

print_success "ARM64 architecture confirmed"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. Consider running as regular user."
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update -qq
if [ $? -ne 0 ]; then
    print_error "Failed to update package lists"
    exit 1
fi

print_status "Upgrading system packages..."
sudo apt upgrade -y -qq
if [ $? -ne 0 ]; then
    print_warning "Some packages failed to upgrade, continuing..."
fi

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    build-essential \
    pkg-config \
    libatlas-base-dev \
    gfortran \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    libcanberra-gtk3-dev \
    libxvidcore-dev \
    libx264-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libssl-dev \
    libffi-dev \
    git \
    wget \
    curl

if [ $? -ne 0 ]; then
    print_error "Failed to install system dependencies"
    exit 1
fi

print_success "System dependencies installed"

# Configure GPU memory
print_status "Configuring GPU memory..."
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    print_status "Adding GPU memory configuration to /boot/config.txt"
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt > /dev/null
    print_warning "GPU memory configuration added. Reboot required to take effect."
else
    print_success "GPU memory already configured"
fi

# Enable V4L2 (camera support)
if ! grep -q "start_x=1" /boot/config.txt; then
    print_status "Enabling camera support"
    echo "start_x=1" | sudo tee -a /boot/config.txt > /dev/null
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists, removing..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    print_error "Failed to create virtual environment"
    exit 1
fi

print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install base packages
print_status "Upgrading pip and installing base packages..."
pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    print_error "Failed to upgrade pip"
    exit 1
fi

# Install PyTorch for ARM64
print_status "Installing PyTorch for ARM64..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if [ $? -ne 0 ]; then
    print_error "Failed to install PyTorch"
    exit 1
fi

print_success "PyTorch installed successfully"

# Install other requirements
print_status "Installing Python dependencies..."
if [ ! -f "requirements_pi5.txt" ]; then
    print_error "requirements_pi5.txt not found"
    exit 1
fi

pip install -r requirements_pi5.txt
if [ $? -ne 0 ]; then
    print_error "Failed to install Python dependencies"
    exit 1
fi

print_success "Python dependencies installed"

# Test installation
print_status "Testing installation..."

echo -n "Testing PyTorch... "
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "PyTorch test failed"
    exit 1
fi

echo -n "Testing OpenCV... "
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "OpenCV test failed"
    exit 1
fi

echo -n "Testing ONNX Runtime... "
python3 -c "import onnxruntime as ort; print(f'✓ ONNX Runtime {ort.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "ONNX Runtime test failed"
    exit 1
fi

echo -n "Testing NumPy... "
python3 -c "import numpy as np; print(f'✓ NumPy {np.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "NumPy test failed"
    exit 1
fi

# Check ONNX Runtime providers
print_status "Checking ONNX Runtime execution providers..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available execution providers:')
for provider in providers:
    print(f'  - {provider}')
if 'OpenCLExecutionProvider' in providers:
    print('✓ GPU acceleration (OpenCL) available')
else:
    print('→ GPU acceleration not available, using CPU only')
"

# Check system temperature
TEMP=$(vcgencmd measure_temp 2>/dev/null | cut -d= -f2 | cut -d\' -f1)
if [ ! -z "$TEMP" ]; then
    print_status "Current CPU temperature: ${TEMP}°C"
    if (( $(echo "$TEMP > 70" | bc -l) )); then
        print_warning "High temperature detected. Ensure proper cooling."
    fi
fi

# Create src directory for future development
mkdir -p src

print_success "Setup complete!"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Device: $DEVICE_MODEL"
echo -e "Architecture: $ARCH"
echo -e "Virtual environment: $(pwd)/venv"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Activate environment: ${BLUE}source venv/bin/activate${NC}"
echo -e "2. Test with demo: ${BLUE}python3 demo_pi5.py --help${NC}"
echo -e "3. If GPU memory was configured, reboot for changes to take effect"
echo ""
echo -e "${GREEN}Happy tracking!${NC}"