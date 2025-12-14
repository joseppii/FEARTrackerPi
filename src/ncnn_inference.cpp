#include "ncnn_inference.h"
#include <iostream>
#include <cstdlib>

NCNNInference::NCNNInference() {
    setup_gpu();
}

NCNNInference::~NCNNInference() {
    template_net_.clear();
    search_net_.clear();
}

void NCNNInference::setup_gpu() {
#if NCNN_VULKAN
    // Check for environment variable to force GPU mode
    // Default is CPU since Pi5's VideoCore VII has too much overhead for small tensors
    const char* force_gpu = std::getenv("NCNN_FORCE_GPU");
    if (force_gpu && std::string(force_gpu) == "1") {
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0) {
            use_gpu_ = true;
            std::cout << "NCNN Vulkan GPU acceleration enabled (forced)" << std::endl;
            std::cout << "  GPU: " << ncnn::get_gpu_info(0).device_name() << std::endl;
            return;
        }
    }

    // Default to CPU - faster for this workload on Pi5
    use_gpu_ = false;
    std::cout << "NCNN using CPU (set NCNN_FORCE_GPU=1 to use Vulkan GPU)" << std::endl;
#else
    use_gpu_ = false;
    std::cout << "NCNN built without Vulkan support, using CPU" << std::endl;
#endif
}

std::string NCNNInference::get_gpu_name() const {
#if NCNN_VULKAN
    if (use_gpu_ && ncnn::get_gpu_count() > 0) {
        return ncnn::get_gpu_info(0).device_name();
    }
#endif
    return "CPU";
}

bool NCNNInference::load_template_model(const std::string& param_path, const std::string& bin_path) {
    template_net_.clear();

#if NCNN_VULKAN
    if (use_gpu_) {
        template_net_.opt.use_vulkan_compute = true;
    }
#endif

    // Optimize for inference
    template_net_.opt.use_fp16_packed = true;
    template_net_.opt.use_fp16_storage = true;
    template_net_.opt.use_fp16_arithmetic = true;
    template_net_.opt.use_packing_layout = true;
    template_net_.opt.num_threads = 4;

    int ret = template_net_.load_param(param_path.c_str());
    if (ret != 0) {
        std::cerr << "Failed to load template model param: " << param_path << std::endl;
        return false;
    }

    ret = template_net_.load_model(bin_path.c_str());
    if (ret != 0) {
        std::cerr << "Failed to load template model bin: " << bin_path << std::endl;
        return false;
    }

    // Get input/output blob names from the model
    const std::vector<const char*>& input_names = template_net_.input_names();
    const std::vector<const char*>& output_names = template_net_.output_names();

    if (input_names.empty() || output_names.empty()) {
        std::cerr << "Template model has no inputs/outputs" << std::endl;
        return false;
    }

    template_input_name_ = input_names[0];
    template_output_name_ = output_names[0];

    std::cout << "Template model loaded: " << param_path << std::endl;
    std::cout << "  Input: " << template_input_name_ << std::endl;
    std::cout << "  Output: " << template_output_name_ << std::endl;

    template_loaded_ = true;
    return true;
}

bool NCNNInference::load_search_model(const std::string& param_path, const std::string& bin_path) {
    search_net_.clear();

#if NCNN_VULKAN
    if (use_gpu_) {
        search_net_.opt.use_vulkan_compute = true;
    }
#endif

    // Optimize for inference
    search_net_.opt.use_fp16_packed = true;
    search_net_.opt.use_fp16_storage = true;
    search_net_.opt.use_fp16_arithmetic = true;
    search_net_.opt.use_packing_layout = true;
    search_net_.opt.num_threads = 4;

    int ret = search_net_.load_param(param_path.c_str());
    if (ret != 0) {
        std::cerr << "Failed to load search model param: " << param_path << std::endl;
        return false;
    }

    ret = search_net_.load_model(bin_path.c_str());
    if (ret != 0) {
        std::cerr << "Failed to load search model bin: " << bin_path << std::endl;
        return false;
    }

    // Get input/output blob names from the model
    const std::vector<const char*>& input_names = search_net_.input_names();
    const std::vector<const char*>& output_names = search_net_.output_names();

    if (input_names.size() < 2 || output_names.size() < 2) {
        std::cerr << "Search model needs 2 inputs and 2 outputs" << std::endl;
        return false;
    }

    // Determine which input is template_features vs search based on name
    for (const char* name : input_names) {
        std::string n(name);
        if (n.find("template") != std::string::npos || n.find("feat") != std::string::npos) {
            search_template_input_name_ = name;
        } else if (n.find("search") != std::string::npos) {
            search_input_name_ = name;
        }
    }

    // Fallback if names don't match expected patterns
    if (search_template_input_name_.empty()) search_template_input_name_ = input_names[0];
    if (search_input_name_.empty()) search_input_name_ = input_names[1];

    // Outputs: classification (out0, 1 channel) and regression (out1, 4 channels)
    // NCNN returns them in order: out1, out0 - so we swap
    search_cls_output_name_ = "out0";  // 1 channel classification
    search_reg_output_name_ = "out1";  // 4 channel regression

    std::cout << "Search model loaded: " << param_path << std::endl;
    std::cout << "  Inputs: " << search_template_input_name_ << ", " << search_input_name_ << std::endl;
    std::cout << "  Outputs: " << search_cls_output_name_ << ", " << search_reg_output_name_ << std::endl;

    search_loaded_ = true;
    return true;
}

ncnn::Mat NCNNInference::preprocess_image(const cv::Mat& image, int target_size) {
    // Resize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_size, target_size));

    // Convert BGR to RGB and create NCNN Mat with normalization
    // ncnn::Mat::from_pixels does BGR->RGB conversion with PIXEL_BGR2RGB
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR2RGB,
                                          target_size, target_size);

    // Apply ImageNet normalization: (pixel - mean) * norm
    in.substract_mean_normalize(IMAGENET_MEAN, IMAGENET_NORM);

    return in;
}

std::vector<float> NCNNInference::extract_template_features(const cv::Mat& template_patch) {
    if (!template_loaded_) {
        std::cerr << "Template model not loaded" << std::endl;
        return {};
    }

    // Preprocess input image (128x128)
    ncnn::Mat in = preprocess_image(template_patch, 128);

    // Create extractor (vulkan is enabled at Net level, not Extractor)
    ncnn::Extractor ex = template_net_.create_extractor();

    // Set input
    ex.input(template_input_name_.c_str(), in);

    // Get output
    ncnn::Mat out;
    ex.extract(template_output_name_.c_str(), out);

    // Cache shape for search model
    template_features_c_ = out.c;
    template_features_h_ = out.h;
    template_features_w_ = out.w;

    // Convert to vector
    std::vector<float> features(out.total());
    memcpy(features.data(), out.data, out.total() * sizeof(float));

    return features;
}

InferenceResult NCNNInference::process_search_region(const std::vector<float>& template_features,
                                                     const cv::Mat& search_region) {
    InferenceResult result;

    if (!search_loaded_) {
        std::cerr << "Search model not loaded" << std::endl;
        return result;
    }

    if (template_features.empty()) {
        std::cerr << "Empty template features" << std::endl;
        return result;
    }

    // Preprocess search image (256x256)
    ncnn::Mat search_in = preprocess_image(search_region, 256);

    // Create template features Mat from cached shape
    ncnn::Mat template_in(template_features_w_, template_features_h_, template_features_c_);
    memcpy(template_in.data, template_features.data(), template_features.size() * sizeof(float));

    // Create extractor (vulkan is enabled at Net level, not Extractor)
    ncnn::Extractor ex = search_net_.create_extractor();

    // Set inputs
    ex.input(search_template_input_name_.c_str(), template_in);
    ex.input(search_input_name_.c_str(), search_in);

    // Get outputs
    ncnn::Mat cls_out, reg_out;
    ex.extract(search_cls_output_name_.c_str(), cls_out);
    ex.extract(search_reg_output_name_.c_str(), reg_out);

    // Convert to vectors
    result.classification_map.resize(cls_out.total());
    memcpy(result.classification_map.data(), cls_out.data, cls_out.total() * sizeof(float));

    result.regression_map.resize(reg_out.total());
    memcpy(result.regression_map.data(), reg_out.data, reg_out.total() * sizeof(float));

    // Store output size
    result.output_size = cv::Size(cls_out.w, cls_out.h);

    return result;
}
