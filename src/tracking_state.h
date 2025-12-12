#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <chrono>

struct TrackingFrame {
    cv::Rect bbox;
    float confidence;
    std::chrono::steady_clock::time_point timestamp;
    cv::Scalar mean_color;
    
    TrackingFrame() : bbox(0, 0, 0, 0), confidence(0.0f), mean_color(0, 0, 0) {
        timestamp = std::chrono::steady_clock::now();
    }
    
    TrackingFrame(const cv::Rect& b, float c, const cv::Scalar& color) 
        : bbox(b), confidence(c), mean_color(color) {
        timestamp = std::chrono::steady_clock::now();
    }
};

class TrackingState {
public:
    TrackingState();
    ~TrackingState();
    
    // State initialization
    void initialize(const cv::Mat& frame, const cv::Rect& initial_bbox);
    
    // State updates
    void update(const cv::Mat& frame, const cv::Rect& new_bbox, float confidence);
    void reset();
    
    // Getters
    cv::Rect get_current_bbox() const;
    float get_last_confidence() const;
    cv::Scalar get_mean_color() const;
    std::vector<cv::Rect> get_bbox_history(int num_frames = -1) const;
    std::vector<float> get_confidence_history(int num_frames = -1) const;
    
    // State validation
    bool is_initialized() const { return initialized_; }
    bool is_tracking_stable() const;
    float get_tracking_quality() const;
    
    // Configuration
    void set_history_length(int length);
    void set_stability_threshold(float threshold) { stability_threshold_ = threshold; }
    void set_quality_window_size(int size) { quality_window_size_ = size; }
    
    // Statistics
    int get_frame_count() const { return frame_count_; }
    double get_tracking_duration() const;
    cv::Point2f get_velocity() const;
    float get_scale_change_rate() const;

private:
    // Core state
    bool initialized_;
    int frame_count_;
    std::chrono::steady_clock::time_point start_time_;
    
    // Tracking history
    std::deque<TrackingFrame> history_;
    int max_history_length_;
    
    // Quality metrics
    float stability_threshold_;
    int quality_window_size_;
    
    // Cached computations
    mutable cv::Point2f velocity_;
    mutable float scale_change_rate_;
    mutable bool velocity_valid_;
    mutable bool scale_rate_valid_;
    
    // Helper methods
    void update_cached_metrics() const;
    float calculate_bbox_overlap(const cv::Rect& bbox1, const cv::Rect& bbox2) const;
    cv::Scalar calculate_mean_color(const cv::Mat& frame, const cv::Rect& bbox) const;
    float calculate_confidence_trend() const;
    
    // Validation
    bool validate_bbox_transition(const cv::Rect& prev_bbox, const cv::Rect& new_bbox) const;
    void invalidate_cache() const;
};