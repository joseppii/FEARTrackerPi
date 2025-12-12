#include "onnx_inference.h"
#include <iostream>
#include <algorithm>
#include <cmath>

ONNXInference::ONNXInference()
    : env_(nullptr)
    , template_session_(nullptr)
    , search_session_(nullptr)
    , memory_info_(nullptr)
    , session_options_(nullptr)
    , intra_op_threads_(4)
    , inter_op_threads_(1)
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
    
    try {
        // Preprocess template image
        cv::Mat processed = preprocess_for_inference(template_patch, cv::Size(128, 128));
        std::vector<float> input_tensor = mat_to_tensor(processed, true);
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, 3, 128, 128};
        Ort::Value input_tensor_ort = create_tensor(input_tensor, input_shape);
        
        // Prepare input/output
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor_ort));
        
        std::vector<const char*> input_names;
        for (const auto& name : template_model_info_.input_names) {
            input_names.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names;
        for (const auto& name : template_model_info_.output_names) {
            output_names.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = template_session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());
        
        if (output_tensors.empty()) {
            std::cerr << "No output from template model" << std::endl;
            return {};
        }
        
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
    
    try {
        // Preprocess search image
        cv::Mat processed = preprocess_for_inference(search_region, cv::Size(256, 256));
        std::vector<float> search_tensor = mat_to_tensor(processed, true);
        
        // Create input tensors
        std::vector<int64_t> search_shape = {1, 3, 256, 256};
        Ort::Value search_tensor_ort = create_tensor(search_tensor, search_shape);
        
        // Template features shape (depends on model architecture)
        std::vector<int64_t> template_shape = {1, static_cast<int64_t>(template_features.size())};
        
        // Find correct shape from model info if available
        for (const auto& shape : search_model_info_.input_shapes) {
            if (shape.size() >= 2) {
                size_t feature_size = 1;
                for (size_t i = 1; i < shape.size(); ++i) {
                    feature_size *= shape[i];
                }
                if (feature_size == template_features.size()) {
                    template_shape = shape;
                    break;
                }
            }
        }
        
        Ort::Value template_tensor_ort = create_tensor(template_features, template_shape);
        
        // Prepare inputs
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;
        
        // Order inputs based on model input names
        for (const auto& name : search_model_info_.input_names) {
            if (name == "template_features" || name == "template") {
                input_tensors.push_back(std::move(template_tensor_ort));
                input_names.push_back(name.c_str());
            } else if (name == "search") {
                input_tensors.push_back(std::move(search_tensor_ort));
                input_names.push_back(name.c_str());
            }
        }
        
        std::vector<const char*> output_names;
        for (const auto& name : search_model_info_.output_names) {
            output_names.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = search_session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());
        
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
    std::vector<std::string> active;
    if (template_session_) {
        auto providers_info = template_session_->GetProviders();
        for (const auto& provider : providers_info) {
            active.push_back(provider);
        }
    }
    return active;
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
        
        // Add execution providers
        for (const auto& provider : preferred_providers_) {
            if (provider == "OpenCLExecutionProvider") {
                try {
                    OrtOpenCLProviderOptions opencl_options{};
                    opencl_options.device_type = 1;  // GPU
                    opencl_options.enable_opencl_throttling = 1;
                    session_options_->AppendExecutionProvider_OpenCL(opencl_options);
                } catch (...) {
                    std::cout << "OpenCL provider setup failed, continuing with CPU" << std::endl;
                }
            }
        }
        
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
    cv::Mat processed;
    
    // Resize image
    cv::resize(image, processed, target_size);
    
    // Convert BGR to RGB if needed
    if (image.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    // Convert to float
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    
    return processed;
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
                float* row_data = mat.ptr<float>(y);
                tensor_data.insert(tensor_data.end(), row_data, row_data + mat.cols);
            }
        }
    }
    
    return tensor_data;
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