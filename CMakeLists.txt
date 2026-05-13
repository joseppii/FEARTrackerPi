cmake_minimum_required(VERSION 3.16)
project(FEARTrackerPi VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Inference backend option
option(USE_NCNN "Use NCNN backend (with Vulkan GPU support)" ON)
option(USE_ONNX "Use ONNX Runtime backend" OFF)

# Optimize for Pi5 Cortex-A76
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16+simd -mtune=cortex-a76")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DDEBUG")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")

# Find required packages
find_package(PkgConfig REQUIRED)

# OpenCV (required)
find_package(OpenCV 4.0 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Inference backend selection
if(USE_NCNN)
    message(STATUS "Using NCNN backend with Vulkan GPU support")

    # Allow override: -DNCNN_ROOT=/path/to/ncnn-install
    set(NCNN_ROOT "$ENV{HOME}/opt/ncnn" CACHE PATH "NCNN install prefix")

    # Find NCNN
    find_package(ncnn QUIET)
    if(NOT ncnn_FOUND)
        # Try to find NCNN manually
        find_path(NCNN_INCLUDE_DIR
            NAMES ncnn/net.h
            PATHS
                ${NCNN_ROOT}/include
                /usr/local/include
                /usr/include
        )
        find_library(NCNN_LIB
            NAMES ncnn
            PATHS
                ${NCNN_ROOT}/lib
                /usr/local/lib
                /usr/lib
        )
        if(NCNN_INCLUDE_DIR AND NCNN_LIB)
            message(STATUS "NCNN found: ${NCNN_LIB}")
            set(NCNN_LIBRARIES ${NCNN_LIB})
            set(NCNN_INCLUDE_DIRS ${NCNN_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "NCNN not found. Set -DNCNN_ROOT=<install-prefix> or install to ${NCNN_ROOT}")
        endif()
    else()
        message(STATUS "NCNN found via CMake package")
        set(NCNN_LIBRARIES ncnn)
    endif()

    # Find Vulkan
    find_package(Vulkan REQUIRED)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")

    add_definitions(-DUSE_NCNN=1)

elseif(USE_ONNX)
    message(STATUS "Using ONNX Runtime backend")

    # ONNX Runtime
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            /usr/include/onnxruntime
            /usr/local/include/onnxruntime
            /opt/onnxruntime/include
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
            /opt/onnxruntime/lib
    )
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime for ARM64")
    endif()
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

    add_definitions(-DUSE_ONNX=1)

else()
    message(FATAL_ERROR "No inference backend selected. Set USE_NCNN=ON or USE_ONNX=ON")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${OpenCV_INCLUDE_DIRS})

if(USE_NCNN)
    include_directories(${NCNN_INCLUDE_DIRS})
elseif(USE_ONNX)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif()

# Source files (common)
set(FEAR_TRACKER_SOURCES
    src/fear_tracker.cpp
    src/image_processor.cpp
    src/tracking_state.cpp
    src/performance_monitor.cpp
    src/utils.cpp
)

# Add backend-specific inference source
if(USE_NCNN)
    list(APPEND FEAR_TRACKER_SOURCES src/ncnn_inference.cpp)
elseif(USE_ONNX)
    list(APPEND FEAR_TRACKER_SOURCES src/onnx_inference.cpp)
endif()

set(MAIN_SOURCES
    src/main.cpp
)

# Create library
add_library(feartracker_lib STATIC ${FEAR_TRACKER_SOURCES})

# Link libraries to feartracker_lib
if(USE_NCNN)
    find_package(OpenMP REQUIRED)

    # glslang libraries (built alongside NCNN for Vulkan shader compilation).
    # Derive from NCNN_LIB's directory; only include libs that actually exist
    # (glslang layout differs across versions — newer builds bundle several
    # archives into libglslang.a).
    get_filename_component(NCNN_LIB_DIR "${NCNN_LIB}" DIRECTORY)
    set(GLSLANG_LIBS "")
    foreach(_lib
            libglslang
            libglslang-default-resource-limits
            libMachineIndependent
            libGenericCodeGen
            libOSDependent
            libSPIRV)
        if(EXISTS "${NCNN_LIB_DIR}/${_lib}.a")
            list(APPEND GLSLANG_LIBS "${NCNN_LIB_DIR}/${_lib}.a")
        endif()
    endforeach()
    message(STATUS "glslang libs: ${GLSLANG_LIBS}")

    target_link_libraries(feartracker_lib
        ${OpenCV_LIBS}
        ${NCNN_LIBRARIES}
        ${GLSLANG_LIBS}
        ${Vulkan_LIBRARIES}
        OpenMP::OpenMP_CXX
        pthread
    )
elseif(USE_ONNX)
    target_link_libraries(feartracker_lib
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIB}
        pthread
    )
endif()

# Create main executable
add_executable(feartracker_pi ${MAIN_SOURCES})
target_link_libraries(feartracker_pi feartracker_lib)

# Create benchmark executable
add_executable(feartracker_benchmark src/benchmark.cpp)
target_link_libraries(feartracker_benchmark feartracker_lib)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(feartracker_lib PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif()

# Enable ARM NEON optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Enabling ARM64 NEON optimizations")
    target_compile_definitions(feartracker_lib PRIVATE USE_ARM_NEON=1)
endif()

# Install targets
install(TARGETS feartracker_pi feartracker_benchmark
    RUNTIME DESTINATION bin
)

install(DIRECTORY models/
    DESTINATION share/feartracker/models
    FILES_MATCHING PATTERN "*.onnx"
)

# Build information
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/build_info.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/build_info.h"
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Testing (optional)
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print summary
message(STATUS "")
message(STATUS "FEARTrackerPi Configuration Summary:")
message(STATUS "====================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "ARM NEON optimizations: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")