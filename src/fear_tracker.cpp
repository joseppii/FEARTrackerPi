#include "fear_tracker.h"
#if defined(USE_NCNN)
#include "ncnn_inference.h"
#else
#include "onnx_inference.h"
#endif
#include "image_processor.h"
#include "tracking_state.h"
#include "performance_monitor.h"
#include "utils.h"
#include <iostream>
#include <algorithm>

FEARTracker::FEARTracker()
    : inference_engine_(std::make_unique<InferenceEngine>())
    , preprocessor_(std::make_unique<ImageProcessor>())
    , state_(std::make_unique<TrackingState>())
    , monitor_(std::make_unique<PerformanceMonitor>())
    , template_size_(128)
    , instance_size_(256)
    , search_context_(2.0f)
    , template_offset_(0.2f)
    , initialized_(false)
    , tracking_(false)
    , monitoring_enabled_(false)
{
    set_default_configuration();
}

FEARTracker::~FEARTracker() = default;

bool FEARTracker::initialize(const std::string& template_model_path,
                            const std::string& search_model_path) {
    if (!validate_configuration()) {
        std::cerr << "Invalid configuration" << std::endl;
        return false;
    }

#if defined(USE_NCNN)
    // NCNN: paths are .param files, .bin files have same basename
    std::string template_bin = template_model_path;
    std::string search_bin = search_model_path;

    // Replace .param with .bin if needed
    auto replace_ext = [](std::string path, const std::string& new_ext) {
        size_t dot = path.rfind('.');
        if (dot != std::string::npos) {
            path = path.substr(0, dot);
        }
        return path + new_ext;
    };

    template_bin = replace_ext(template_model_path, ".bin");
    search_bin = replace_ext(search_model_path, ".bin");

    std::string template_param = replace_ext(template_model_path, ".param");
    std::string search_param = replace_ext(search_model_path, ".param");

    if (!inference_engine_->load_template_model(template_param, template_bin)) {
        std::cerr << "Failed to load template model: " << template_param << std::endl;
        return false;
    }

    if (!inference_engine_->load_search_model(search_param, search_bin)) {
        std::cerr << "Failed to load search model: " << search_param << std::endl;
        return false;
    }

    // Print GPU info
    std::cout << "NCNN backend: " << (inference_engine_->has_gpu() ? "GPU" : "CPU") << std::endl;
    if (inference_engine_->has_gpu()) {
        std::cout << "  GPU: " << inference_engine_->get_gpu_name() << std::endl;
    }

#else
    // ONNX Runtime
    if (!inference_engine_->load_template_model(template_model_path)) {
        std::cerr << "Failed to load template model: " << template_model_path << std::endl;
        return false;
    }

    if (!inference_engine_->load_search_model(search_model_path)) {
        std::cerr << "Failed to load search model: " << search_model_path << std::endl;
        return false;
    }

    // Configure ONNX Runtime for Pi5 optimization
    inference_engine_->set_intra_op_threads(4);  // Use all 4 Cortex-A76 cores
    inference_engine_->set_inter_op_threads(1);

    // Print provider information
    auto providers = inference_engine_->get_active_providers();
    std::cout << "Active ONNX providers: ";
    for (const auto& provider : providers) {
        std::cout << provider << " ";
    }
    std::cout << std::endl;
#endif

    initialized_ = true;
    return true;
}

bool FEARTracker::start_tracking(const cv::Mat& frame, const cv::Rect& bbox) {
    if (!initialized_) {
        std::cerr << "Tracker not initialized" << std::endl;
        return false;
    }
    
    if (!validate_frame(frame)) {
        std::cerr << "Invalid frame" << std::endl;
        return false;
    }
    
    if (!validate_bbox(bbox, frame.size())) {
        std::cerr << "Invalid bounding box" << std::endl;
        return false;
    }
    
    if (monitoring_enabled_) {
        monitor_->start_operation("initialization");
    }
    
    try {
        // Initialize tracking state
        state_->initialize(frame, bbox);
        
        // Extract template crop
        cv::Mat template_crop;
        cv::Rect template_bbox;
        if (!preprocessor_->extract_template_crop(frame, bbox, template_size_, 
                                                template_offset_, template_crop, template_bbox)) {
            std::cerr << "Failed to extract template crop" << std::endl;
            return false;
        }
        
        // Extract template features using ONNX
        template_features_ = inference_engine_->extract_template_features(template_crop);
        
        if (template_features_.empty()) {
            std::cerr << "Failed to extract template features" << std::endl;
            return false;
        }
        
        tracking_ = true;
        
        if (monitoring_enabled_) {
            monitor_->end_operation("initialization");
        }
        
        std::cout << "Tracking initialized with bbox: [" 
                  << bbox.x << ", " << bbox.y << ", " 
                  << bbox.width << ", " << bbox.height << "]" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception during tracking initialization: " << e.what() << std::endl;
        tracking_ = false;
        return false;
    }
}

TrackingResult FEARTracker::update(const cv::Mat& frame) {
    if (!tracking_) {
        return TrackingResult();
    }
    
    if (!validate_frame(frame)) {
        std::cerr << "Invalid frame for tracking update" << std::endl;
        return TrackingResult();
    }
    
    if (monitoring_enabled_) {
        monitor_->start_operation("tracking_update");
    }
    
    try {
        // Get current tracking state
        cv::Rect current_bbox = state_->get_current_bbox();
        cv::Scalar mean_color = state_->get_mean_color();
        
        // Extract search crop
        cv::Mat search_crop;
        cv::Rect search_region;
        cv::Point2f search_center;
        
        if (!preprocessor_->extract_search_crop(frame, current_bbox, instance_size_, 
                                              search_context_, mean_color, 
                                              search_crop, search_region, search_center)) {
            std::cerr << "Failed to extract search crop" << std::endl;
            return TrackingResult();
        }
        
        // Run inference
        InferenceResult inference_result = inference_engine_->process_search_region(
            template_features_, search_crop);
        
        if (inference_result.classification_map.empty() || 
            inference_result.regression_map.empty()) {
            std::cerr << "Invalid inference result" << std::endl;
            return TrackingResult();
        }
        
        // Postprocess results to get bounding box
        cv::Rect predicted_bbox;
        float confidence;
        
        if (!preprocessor_->postprocess_inference_result(
                inference_result, search_region, search_center, 
                predicted_bbox, confidence)) {
            std::cerr << "Failed to postprocess inference result" << std::endl;
            return TrackingResult();
        }
        
        // Clamp bbox to frame boundaries
        predicted_bbox = clamp_bbox(predicted_bbox, frame.size());
        
        // Update tracking state
        state_->update(frame, predicted_bbox, confidence);
        
        if (monitoring_enabled_) {
            monitor_->end_operation("tracking_update");
            monitor_->record_fps();
        }
        
        return TrackingResult(predicted_bbox, confidence);
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during tracking update: " << e.what() << std::endl;
        return TrackingResult();
    }
}

void FEARTracker::reset_tracking() {
    tracking_ = false;
    template_features_.clear();
    if (state_) {
        state_->reset();
    }
    if (monitor_) {
        monitor_->reset();
    }
}

void FEARTracker::enable_performance_monitoring(bool enable) {
    monitoring_enabled_ = enable;
    if (monitor_) {
        monitor_->set_enabled(enable);
    }
}

std::string FEARTracker::get_performance_stats() const {
    if (monitor_ && monitoring_enabled_) {
        return monitor_->get_summary();
    }
    return "Performance monitoring disabled";
}

cv::Rect FEARTracker::get_current_bbox() const {
    if (state_ && tracking_) {
        return state_->get_current_bbox();
    }
    return cv::Rect();
}

float FEARTracker::get_last_confidence() const {
    if (state_ && tracking_) {
        return state_->get_last_confidence();
    }
    return 0.0f;
}

std::vector<cv::Rect> FEARTracker::get_tracking_history(int num_frames) const {
    if (state_ && tracking_) {
        return state_->get_bbox_history(num_frames);
    }
    return std::vector<cv::Rect>();
}

bool FEARTracker::validate_frame(const cv::Mat& frame) const {
    return !frame.empty() && 
           frame.channels() == 3 && 
           frame.depth() == CV_8U &&
           frame.rows > 32 && 
           frame.cols > 32;
}

bool FEARTracker::validate_bbox(const cv::Rect& bbox, const cv::Size& frame_size) const {
    return bbox.width > 0 && 
           bbox.height > 0 &&
           bbox.x >= 0 && 
           bbox.y >= 0 &&
           bbox.x + bbox.width <= frame_size.width &&
           bbox.y + bbox.height <= frame_size.height &&
           bbox.width >= 8 && 
           bbox.height >= 8;
}

cv::Rect FEARTracker::clamp_bbox(const cv::Rect& bbox, const cv::Size& frame_size) const {
    cv::Rect clamped = bbox;
    
    // Clamp position
    clamped.x = std::max(0, std::min(clamped.x, frame_size.width - 1));
    clamped.y = std::max(0, std::min(clamped.y, frame_size.height - 1));
    
    // Clamp size
    clamped.width = std::max(1, std::min(clamped.width, frame_size.width - clamped.x));
    clamped.height = std::max(1, std::min(clamped.height, frame_size.height - clamped.y));
    
    return clamped;
}

bool FEARTracker::validate_configuration() const {
    return template_size_ > 0 && template_size_ <= 512 &&
           instance_size_ > 0 && instance_size_ <= 1024 &&
           search_context_ > 0.0f && search_context_ <= 10.0f &&
           template_offset_ > 0.0f && template_offset_ <= 10.0f &&
           template_size_ <= instance_size_;
}

void FEARTracker::set_default_configuration() {
    template_size_ = 128;
    instance_size_ = 256;
    search_context_ = 2.0f;
    template_offset_ = 0.2f;
}