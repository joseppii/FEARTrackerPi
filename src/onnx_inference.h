#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

struct InferenceResult {
    std::vector<float> classification_map;  // [1, 1, H, W]
    std::vector<float> regression_map;      // [1, 4, H, W]
    cv::Size output_size;                   // H, W of the output maps
    
    InferenceResult() : output_size(0, 0) {}
};

struct ModelInfo {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};

class ONNXInference {
public:
    ONNXInference();
    ~ONNXInference();
    
    // Model loading
    bool load_template_model(const std::string& model_path);
    bool load_search_model(const std::string& model_path);
    bool is_ready() const { return template_session_ && search_session_; }
    
    // Template feature extraction
    std::vector<float> extract_template_features(const cv::Mat& template_patch);
    
    // Search and tracking
    InferenceResult process_search_region(const std::vector<float>& template_features,
                                         const cv::Mat& search_region);
    
    // Model information
    ModelInfo get_template_model_info() const { return template_model_info_; }
    ModelInfo get_search_model_info() const { return search_model_info_; }
    
    // Provider information
    std::vector<std::string> get_available_providers() const;
    std::vector<std::string> get_active_providers() const;
    
    // Performance optimization
    void set_intra_op_threads(int num_threads);
    void set_inter_op_threads(int num_threads);

private:
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> template_session_;
    std::unique_ptr<Ort::Session> search_session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    // Model metadata
    ModelInfo template_model_info_;
    ModelInfo search_model_info_;
    
    // Configuration
    int intra_op_threads_;
    int inter_op_threads_;
    std::vector<std::string> preferred_providers_;

    // Cached template features shape (set after template extraction)
    std::vector<int64_t> template_features_shape_;

    // Pre-allocated buffers for speed (avoid repeated allocations)
    std::vector<float> template_input_buffer_;
    std::vector<float> search_input_buffer_;
    cv::Mat preprocess_buffer_;
    std::vector<cv::Mat> channel_buffers_;
    bool buffers_initialized_ = false;

    // Cached name pointers (avoid rebuilding every call)
    std::vector<const char*> template_input_names_ptrs_;
    std::vector<const char*> template_output_names_ptrs_;
    std::vector<const char*> search_input_names_ptrs_;
    std::vector<const char*> search_output_names_ptrs_;

    // Pre-allocated run options
    Ort::RunOptions run_options_;

    void initialize_buffers();
    void cache_name_pointers();

    // Helper methods
    bool setup_providers();
    bool create_session_options();
    std::unique_ptr<Ort::Session> create_session(const std::string& model_path);
    ModelInfo extract_model_info(Ort::Session* session);
    
    // Image preprocessing
    cv::Mat preprocess_for_inference(const cv::Mat& image, cv::Size target_size);
    std::vector<float> mat_to_tensor(const cv::Mat& mat, bool normalize = true);
    void mat_to_tensor_fast(const cv::Mat& mat, std::vector<float>& out_buffer, bool normalize = true);
    void image_to_tensor_direct(const cv::Mat& image, std::vector<float>& out_buffer, int target_size);
    
    // Tensor utilities
    Ort::Value create_tensor(const std::vector<float>& data, 
                           const std::vector<int64_t>& shape);
    std::vector<float> tensor_to_vector(const Ort::Value& tensor);
    
    // ImageNet normalization constants
    static constexpr float IMAGENET_MEAN[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float IMAGENET_STD[3] = {0.229f, 0.224f, 0.225f};
};