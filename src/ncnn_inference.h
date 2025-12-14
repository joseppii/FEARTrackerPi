#pragma once

#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include "inference_types.h"

class NCNNInference {
public:
    NCNNInference();
    ~NCNNInference();

    // Model loading
    bool load_template_model(const std::string& param_path, const std::string& bin_path);
    bool load_search_model(const std::string& param_path, const std::string& bin_path);
    bool is_ready() const { return template_loaded_ && search_loaded_; }

    // Template feature extraction
    std::vector<float> extract_template_features(const cv::Mat& template_patch);

    // Search and tracking
    InferenceResult process_search_region(const std::vector<float>& template_features,
                                         const cv::Mat& search_region);

    // GPU info
    bool has_gpu() const { return use_gpu_; }
    std::string get_gpu_name() const;

private:
    ncnn::Net template_net_;
    ncnn::Net search_net_;

    bool template_loaded_ = false;
    bool search_loaded_ = false;
    bool use_gpu_ = false;

    // Input/output blob names (discovered at load time)
    std::string template_input_name_;
    std::string template_output_name_;
    std::string search_input_name_;
    std::string search_template_input_name_;
    std::string search_cls_output_name_;
    std::string search_reg_output_name_;

    // Cached template features shape
    int template_features_c_ = 0;
    int template_features_h_ = 0;
    int template_features_w_ = 0;

    // ImageNet normalization constants
    static constexpr float IMAGENET_MEAN[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    static constexpr float IMAGENET_NORM[3] = {1.0f / (0.229f * 255.f), 1.0f / (0.224f * 255.f), 1.0f / (0.225f * 255.f)};

    // Helper methods
    void setup_gpu();
    ncnn::Mat preprocess_image(const cv::Mat& image, int target_size);
};
