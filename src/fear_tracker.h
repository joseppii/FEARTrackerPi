#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Forward declarations for inference backends
#if defined(USE_NCNN)
class NCNNInference;
using InferenceEngine = NCNNInference;
#else
class ONNXInference;
using InferenceEngine = ONNXInference;
#endif

class ImageProcessor;
class TrackingState;
class PerformanceMonitor;

struct TrackingResult {
    cv::Rect bbox;
    float confidence;
    bool valid;
    
    TrackingResult() : bbox(0, 0, 0, 0), confidence(0.0f), valid(false) {}
    TrackingResult(const cv::Rect& b, float c) : bbox(b), confidence(c), valid(true) {}
};

class FEARTracker {
public:
    FEARTracker();
    ~FEARTracker();
    
    // Core functionality
    bool initialize(const std::string& template_model_path,
                   const std::string& search_model_path);
    bool start_tracking(const cv::Mat& frame, const cv::Rect& bbox);
    TrackingResult update(const cv::Mat& frame);
    
    // State management
    bool is_initialized() const { return initialized_; }
    bool is_tracking() const { return tracking_; }
    void reset_tracking();
    
    // Configuration
    void set_template_size(int size) { template_size_ = size; }
    void set_instance_size(int size) { instance_size_ = size; }
    void set_search_context(float context) { search_context_ = context; }
    void set_template_offset(float offset) { template_offset_ = offset; }
    
    // Performance monitoring
    void enable_performance_monitoring(bool enable);
    std::string get_performance_stats() const;
    
    // Getters
    cv::Rect get_current_bbox() const;
    float get_last_confidence() const;
    std::vector<cv::Rect> get_tracking_history(int num_frames = 10) const;

private:
    // Core components
    std::unique_ptr<InferenceEngine> inference_engine_;
    std::unique_ptr<ImageProcessor> preprocessor_;
    std::unique_ptr<TrackingState> state_;
    std::unique_ptr<PerformanceMonitor> monitor_;
    
    // Configuration parameters
    int template_size_;
    int instance_size_;
    float search_context_;
    float template_offset_;
    
    // State flags
    bool initialized_;
    bool tracking_;
    bool monitoring_enabled_;
    
    // Template features cached after initialization
    std::vector<float> template_features_;
    
    // Helper methods
    bool validate_frame(const cv::Mat& frame) const;
    bool validate_bbox(const cv::Rect& bbox, const cv::Size& frame_size) const;
    cv::Rect clamp_bbox(const cv::Rect& bbox, const cv::Size& frame_size) const;
    
    // Configuration validation
    bool validate_configuration() const;
    void set_default_configuration();
};