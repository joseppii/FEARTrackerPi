#include "onnx_inference.h"
#include <iostream>
#include <algorithm>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

ONNXInference::ONNXInference()
    : env_(nullptr)
    , template_session_(nullptr)
    , search_session_(nullptr)
    , memory_info_(nullptr)
    , session_options_(nullptr)
    , intra_op_threads_(4)  // Pi5 has 4 cores - use all for parallel ops within a single inference
    , inter_op_threads_(2)  // Allow some parallelism between independent ops
{
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FEARTrackerPi");
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, 
                                      OrtMemType::OrtMemTypeDefault));
        
        if (!create_session_options()) {
            throw std::runtime_error("Failed to create session options");
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime initialization error: " << e.what() << std::endl;
        throw;
    }
}

ONNXInference::~ONNXInference() = default;

void ONNXInference::initialize_buffers() {
    if (buffers_initialized_) return;

    // Pre-allocate buffers for template (128x128x3) and search (256x256x3)
    template_input_buffer_.resize(3 * 128 * 128);
    search_input_buffer_.resize(3 * 256 * 256);

    // Pre-allocate channel split buffers
    channel_buffers_.resize(3);

    buffers_initialized_ = true;
}

void ONNXInference::cache_name_pointers() {
    // Cache template model name pointers
    template_input_names_ptrs_.clear();
    template_output_names_ptrs_.clear();
    for (const auto& name : template_model_info_.input_names) {
        template_input_names_ptrs_.push_back(name.c_str());
    }
    for (const auto& name : template_model_info_.output_names) {
        template_output_names_ptrs_.push_back(name.c_str());
    }

    // Cache search model name pointers
    search_input_names_ptrs_.clear();
    search_output_names_ptrs_.clear();
    for (const auto& name : search_model_info_.input_names) {
        search_input_names_ptrs_.push_back(name.c_str());
    }
    for (const auto& name : search_model_info_.output_names) {
        search_output_names_ptrs_.push_back(name.c_str());
    }
}

bool ONNXInference::load_template_model(const std::string& model_path) {
    try {
        template_session_ = create_session(model_path);
        if (!template_session_) {
            return false;
        }
        
        template_model_info_ = extract_model_info(template_session_.get());
        
        std::cout << "Template model loaded: " << model_path << std::endl;
        std::cout << "  Inputs: ";
        for (const auto& name : template_model_info_.input_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load template model: " << e.what() << std::endl;
        return false;
    }
}

bool ONNXInference::load_search_model(const std::string& model_path) {
    try {
        search_session_ = create_session(model_path);
        if (!search_session_) {
            return false;
        }

        search_model_info_ = extract_model_info(search_session_.get());

        std::cout << "Search model loaded: " << model_path << std::endl;
        std::cout << "  Inputs: ";
        for (const auto& name : search_model_info_.input_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

        // Cache name pointers now that both models are loaded
        cache_name_pointers();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load search model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> ONNXInference::extract_template_features(const cv::Mat& template_patch) {
    if (!template_session_) {
        std::cerr << "Template model not loaded" << std::endl;
        return {};
    }

    initialize_buffers();

    try {
        // Direct image to tensor conversion (skip intermediate cv::Mat)
        image_to_tensor_direct(template_patch, template_input_buffer_, 128);

        // Create input tensor using pre-allocated buffer
        static const std::vector<int64_t> input_shape = {1, 3, 128, 128};
        Ort::Value input_tensor_ort = create_tensor(template_input_buffer_, input_shape);

        // Use single-element array on stack (avoid vector allocation)
        Ort::Value input_tensors[] = {std::move(input_tensor_ort)};

        // Run inference with cached name pointers and pre-allocated run options
        auto output_tensors = template_session_->Run(
            run_options_,
            template_input_names_ptrs_.data(), input_tensors, 1,
            template_output_names_ptrs_.data(), template_output_names_ptrs_.size());

        if (output_tensors.empty()) {
            std::cerr << "No output from template model" << std::endl;
            return {};
        }

        // Cache the output shape for use in search model
        template_features_shape_ = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Extract features from first output tensor
        return tensor_to_vector(output_tensors[0]);
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference error (template): " << e.what() << std::endl;
        return {};
    }
}

InferenceResult ONNXInference::process_search_region(const std::vector<float>& template_features,
                                                    const cv::Mat& search_region) {
    InferenceResult result;

    if (!search_session_) {
        std::cerr << "Search model not loaded" << std::endl;
        return result;
    }

    if (template_features.empty()) {
        std::cerr << "Empty template features" << std::endl;
        return result;
    }

    initialize_buffers();

    try {
        // Direct image to tensor conversion (skip intermediate cv::Mat ops)
        image_to_tensor_direct(search_region, search_input_buffer_, 256);

        // Create input tensors using pre-allocated buffer
        static const std::vector<int64_t> search_shape = {1, 3, 256, 256};
        Ort::Value search_tensor_ort = create_tensor(search_input_buffer_, search_shape);

        // Use cached template features shape from template model extraction
        if (template_features_shape_.empty()) {
            std::cerr << "Template features shape not available - run extract_template_features first" << std::endl;
            return result;
        }

        Ort::Value template_tensor_ort = create_tensor(template_features, template_features_shape_);

        // Build input tensors - determine order from model input names
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(2);

        for (const auto& name : search_model_info_.input_names) {
            if (name == "template_features" || name == "template") {
                input_tensors.push_back(std::move(template_tensor_ort));
            } else if (name == "search") {
                input_tensors.push_back(std::move(search_tensor_ort));
            }
        }

        // Run inference with cached name pointers and pre-allocated run options
        auto output_tensors = search_session_->Run(
            run_options_,
            search_input_names_ptrs_.data(), input_tensors.data(), input_tensors.size(),
            search_output_names_ptrs_.data(), search_output_names_ptrs_.size());

        if (output_tensors.size() < 2) {
            std::cerr << "Expected 2 outputs, got " << output_tensors.size() << std::endl;
            return result;
        }

        // Extract outputs
        result.classification_map = tensor_to_vector(output_tensors[0]);
        result.regression_map = tensor_to_vector(output_tensors[1]);
        
        // Get output dimensions
        auto cls_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        if (cls_shape.size() >= 2) {
            result.output_size = cv::Size(static_cast<int>(cls_shape[cls_shape.size()-1]),
                                        static_cast<int>(cls_shape[cls_shape.size()-2]));
        }
        
        return result;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference error (search): " << e.what() << std::endl;
        return result;
    }
}

std::vector<std::string> ONNXInference::get_available_providers() const {
    return Ort::GetAvailableProviders();
}

std::vector<std::string> ONNXInference::get_active_providers() const {
    // Return the configured preferred providers
    // (ONNX Runtime C++ API doesn't expose GetProviders on Session)
    return preferred_providers_;
}

void ONNXInference::set_intra_op_threads(int num_threads) {
    intra_op_threads_ = std::max(1, num_threads);
    create_session_options();
}

void ONNXInference::set_inter_op_threads(int num_threads) {
    inter_op_threads_ = std::max(1, num_threads);
    create_session_options();
}

bool ONNXInference::setup_providers() {
    preferred_providers_.clear();
    
    auto available = Ort::GetAvailableProviders();
    
    // Prefer GPU providers for Pi5
    for (const auto& provider : available) {
        if (provider == "OpenCLExecutionProvider") {
            preferred_providers_.push_back(provider);
            std::cout << "✓ VideoCore VII GPU acceleration enabled" << std::endl;
            break;
        }
    }
    
    // Always include CPU provider as fallback
    preferred_providers_.push_back("CPUExecutionProvider");
    
    return true;
}

bool ONNXInference::create_session_options() {
    try {
        session_options_ = std::make_unique<Ort::SessionOptions>();
        
        // Performance settings for Pi5
        session_options_->SetIntraOpNumThreads(intra_op_threads_);
        session_options_->SetInterOpNumThreads(inter_op_threads_);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Memory optimization
        session_options_->EnableMemPattern();
        session_options_->EnableCpuMemArena();
        
        // Setup providers
        setup_providers();
        
        // Note: OpenCL provider is not available in standard ONNX Runtime for ARM
        // The CPUExecutionProvider will be used automatically as fallback
        // GPU acceleration would require a custom ONNX Runtime build with OpenCL support
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create session options: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<Ort::Session> ONNXInference::create_session(const std::string& model_path) {
    try {
        return std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to create session for " << model_path << ": " << e.what() << std::endl;
        return nullptr;
    }
}

ModelInfo ONNXInference::extract_model_info(Ort::Session* session) {
    ModelInfo info;
    
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Extract input information
        auto num_inputs = session->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session->GetInputNameAllocated(i, allocator);
            info.input_names.push_back(std::string(input_name.get()));
            info.input_shapes.push_back(
                session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        
        // Extract output information
        auto num_outputs = session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            info.output_names.push_back(std::string(output_name.get()));
            info.output_shapes.push_back(
                session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to extract model info: " << e.what() << std::endl;
    }
    
    return info;
}

cv::Mat ONNXInference::preprocess_for_inference(const cv::Mat& image, cv::Size target_size) {
    // Resize directly into pre-allocated buffer if possible
    cv::resize(image, preprocess_buffer_, target_size);

    // Convert BGR to RGB if needed
    if (preprocess_buffer_.channels() == 3) {
        cv::cvtColor(preprocess_buffer_, preprocess_buffer_, cv::COLOR_BGR2RGB);
    }

    // Convert to float in-place
    preprocess_buffer_.convertTo(preprocess_buffer_, CV_32F, 1.0 / 255.0);

    return preprocess_buffer_;
}

std::vector<float> ONNXInference::mat_to_tensor(const cv::Mat& mat, bool normalize) {
    std::vector<float> tensor_data;
    
    if (mat.channels() == 3) {
        // Split channels and apply ImageNet normalization if requested
        std::vector<cv::Mat> channels;
        cv::split(mat, channels);
        
        for (int c = 0; c < 3; c++) {
            cv::Mat channel = channels[c];
            
            if (normalize) {
                // Apply ImageNet normalization: (pixel - mean) / std
                channel = (channel - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
            
            // Add channel data to tensor
            if (channel.isContinuous()) {
                float* data = reinterpret_cast<float*>(channel.data);
                tensor_data.insert(tensor_data.end(), data, data + channel.total());
            } else {
                for (int y = 0; y < channel.rows; y++) {
                    float* row_data = channel.ptr<float>(y);
                    tensor_data.insert(tensor_data.end(), row_data, row_data + channel.cols);
                }
            }
        }
    } else {
        // Single channel
        if (mat.isContinuous()) {
            float* data = reinterpret_cast<float*>(mat.data);
            tensor_data.assign(data, data + mat.total());
        } else {
            for (int y = 0; y < mat.rows; y++) {
                const float* row_data = mat.ptr<float>(y);
                tensor_data.insert(tensor_data.end(), row_data, row_data + mat.cols);
            }
        }
    }
    
    return tensor_data;
}

void ONNXInference::mat_to_tensor_fast(const cv::Mat& mat, std::vector<float>& out_buffer, bool normalize) {
    const int rows = mat.rows;
    const int cols = mat.cols;
    const int channel_size = rows * cols;

    if (mat.channels() == 3) {
        // Split channels directly - avoid allocation by reusing channel_buffers_
        cv::split(mat, channel_buffers_);

        float* dst = out_buffer.data();

        for (int c = 0; c < 3; c++) {
            const cv::Mat& channel = channel_buffers_[c];
            const float mean = normalize ? IMAGENET_MEAN[c] : 0.0f;
            const float inv_std = normalize ? (1.0f / IMAGENET_STD[c]) : 1.0f;

            if (channel.isContinuous()) {
                const float* src = reinterpret_cast<const float*>(channel.data);
                for (int i = 0; i < channel_size; ++i) {
                    dst[i] = (src[i] - mean) * inv_std;
                }
            } else {
                int idx = 0;
                for (int y = 0; y < rows; ++y) {
                    const float* row = channel.ptr<float>(y);
                    for (int x = 0; x < cols; ++x) {
                        dst[idx++] = (row[x] - mean) * inv_std;
                    }
                }
            }
            dst += channel_size;
        }
    } else {
        // Single channel - direct copy
        const float* src = reinterpret_cast<const float*>(mat.data);
        std::copy(src, src + mat.total(), out_buffer.data());
    }
}

void ONNXInference::image_to_tensor_direct(const cv::Mat& image, std::vector<float>& out_buffer, int target_size) {
    // Resize image
    cv::resize(image, preprocess_buffer_, cv::Size(target_size, target_size), 0, 0, cv::INTER_LINEAR);

    const int pixels = target_size * target_size;
    float* __restrict dst_r = out_buffer.data();
    float* __restrict dst_g = out_buffer.data() + pixels;
    float* __restrict dst_b = out_buffer.data() + 2 * pixels;
    const uchar* __restrict src = preprocess_buffer_.data;

#if USE_NEON
    // NEON-optimized BGR->RGB conversion with normalization
    // Process 8 pixels at a time (24 bytes BGR -> 8 floats per channel)
    const float32x4_t scale_vec = vdupq_n_f32(1.0f / 255.0f);
    const float32x4_t r_mean_vec = vdupq_n_f32(IMAGENET_MEAN[0]);
    const float32x4_t g_mean_vec = vdupq_n_f32(IMAGENET_MEAN[1]);
    const float32x4_t b_mean_vec = vdupq_n_f32(IMAGENET_MEAN[2]);
    const float32x4_t r_inv_std_vec = vdupq_n_f32(1.0f / IMAGENET_STD[0]);
    const float32x4_t g_inv_std_vec = vdupq_n_f32(1.0f / IMAGENET_STD[1]);
    const float32x4_t b_inv_std_vec = vdupq_n_f32(1.0f / IMAGENET_STD[2]);

    int i = 0;
    const int simd_end = pixels - (pixels % 8);

    for (; i < simd_end; i += 8) {
        // Load 24 bytes (8 BGR pixels)
        uint8x8x3_t bgr = vld3_u8(src + i * 3);

        // Convert to float32 (first 4 pixels)
        uint16x8_t b16 = vmovl_u8(bgr.val[0]);
        uint16x8_t g16 = vmovl_u8(bgr.val[1]);
        uint16x8_t r16 = vmovl_u8(bgr.val[2]);

        float32x4_t b_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16)));
        float32x4_t g_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16)));
        float32x4_t r_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16)));

        float32x4_t b_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b16)));
        float32x4_t g_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g16)));
        float32x4_t r_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r16)));

        // Scale to [0,1]
        b_lo = vmulq_f32(b_lo, scale_vec);
        g_lo = vmulq_f32(g_lo, scale_vec);
        r_lo = vmulq_f32(r_lo, scale_vec);
        b_hi = vmulq_f32(b_hi, scale_vec);
        g_hi = vmulq_f32(g_hi, scale_vec);
        r_hi = vmulq_f32(r_hi, scale_vec);

        // Normalize: (x - mean) * inv_std
        r_lo = vmulq_f32(vsubq_f32(r_lo, r_mean_vec), r_inv_std_vec);
        g_lo = vmulq_f32(vsubq_f32(g_lo, g_mean_vec), g_inv_std_vec);
        b_lo = vmulq_f32(vsubq_f32(b_lo, b_mean_vec), b_inv_std_vec);
        r_hi = vmulq_f32(vsubq_f32(r_hi, r_mean_vec), r_inv_std_vec);
        g_hi = vmulq_f32(vsubq_f32(g_hi, g_mean_vec), g_inv_std_vec);
        b_hi = vmulq_f32(vsubq_f32(b_hi, b_mean_vec), b_inv_std_vec);

        // Store to CHW layout
        vst1q_f32(dst_r + i, r_lo);
        vst1q_f32(dst_r + i + 4, r_hi);
        vst1q_f32(dst_g + i, g_lo);
        vst1q_f32(dst_g + i + 4, g_hi);
        vst1q_f32(dst_b + i, b_lo);
        vst1q_f32(dst_b + i + 4, b_hi);
    }

    // Handle remaining pixels
    constexpr float scale = 1.0f / 255.0f;
    for (; i < pixels; ++i) {
        const int px = i * 3;
        float b = src[px + 0] * scale;
        float g = src[px + 1] * scale;
        float r = src[px + 2] * scale;
        dst_r[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        dst_g[i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        dst_b[i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
#else
    // Scalar fallback
    constexpr float scale = 1.0f / 255.0f;
    for (int i = 0; i < pixels; ++i) {
        const int px = i * 3;
        float b = src[px + 0] * scale;
        float g = src[px + 1] * scale;
        float r = src[px + 2] * scale;
        dst_r[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        dst_g[i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        dst_b[i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
#endif
}

Ort::Value ONNXInference::create_tensor(const std::vector<float>& data,
                                       const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(
        *memory_info_, 
        const_cast<float*>(data.data()), 
        data.size(),
        shape.data(), 
        shape.size());
}

std::vector<float> ONNXInference::tensor_to_vector(const Ort::Value& tensor) {
    const float* tensor_data = tensor.GetTensorData<float>();
    size_t tensor_size = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(tensor_data, tensor_data + tensor_size);
}