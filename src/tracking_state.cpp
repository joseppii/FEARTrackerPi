#include "tracking_state.h"
#include <algorithm>
#include <numeric>
#include <cmath>

TrackingState::TrackingState()
    : initialized_(false)
    , frame_count_(0)
    , max_history_length_(50)
    , stability_threshold_(0.6f)
    , quality_window_size_(10)
    , velocity_(0, 0)
    , scale_change_rate_(0.0f)
    , velocity_valid_(false)
    , scale_rate_valid_(false)
{
}

TrackingState::~TrackingState() = default;

void TrackingState::initialize(const cv::Mat& frame, const cv::Rect& initial_bbox) {
    reset();
    
    cv::Scalar mean_color = calculate_mean_color(frame, initial_bbox);
    history_.emplace_back(initial_bbox, 1.0f, mean_color);
    
    initialized_ = true;
    frame_count_ = 1;
    start_time_ = std::chrono::steady_clock::now();
    
    invalidate_cache();
}

void TrackingState::update(const cv::Mat& frame, const cv::Rect& new_bbox, float confidence) {
    if (!initialized_) {
        return;
    }
    
    // Validate bbox transition
    if (!history_.empty() && !validate_bbox_transition(history_.back().bbox, new_bbox)) {
        // Potentially invalid transition, but we'll allow it with lower confidence
        confidence *= 0.5f;
    }
    
    cv::Scalar mean_color = calculate_mean_color(frame, new_bbox);
    history_.emplace_back(new_bbox, confidence, mean_color);
    
    // Maintain history length
    while (static_cast<int>(history_.size()) > max_history_length_) {
        history_.pop_front();
    }
    
    frame_count_++;
    invalidate_cache();
}

void TrackingState::reset() {
    initialized_ = false;
    frame_count_ = 0;
    history_.clear();
    invalidate_cache();
}

cv::Rect TrackingState::get_current_bbox() const {
    if (history_.empty()) {
        return cv::Rect();
    }
    return history_.back().bbox;
}

float TrackingState::get_last_confidence() const {
    if (history_.empty()) {
        return 0.0f;
    }
    return history_.back().confidence;
}

cv::Scalar TrackingState::get_mean_color() const {
    if (history_.empty()) {
        return cv::Scalar(0, 0, 0);
    }
    return history_.back().mean_color;
}

std::vector<cv::Rect> TrackingState::get_bbox_history(int num_frames) const {
    std::vector<cv::Rect> bboxes;
    
    if (num_frames < 0) {
        num_frames = static_cast<int>(history_.size());
    }
    
    int start_idx = std::max(0, static_cast<int>(history_.size()) - num_frames);
    
    for (int i = start_idx; i < static_cast<int>(history_.size()); ++i) {
        bboxes.push_back(history_[i].bbox);
    }
    
    return bboxes;
}

std::vector<float> TrackingState::get_confidence_history(int num_frames) const {
    std::vector<float> confidences;
    
    if (num_frames < 0) {
        num_frames = static_cast<int>(history_.size());
    }
    
    int start_idx = std::max(0, static_cast<int>(history_.size()) - num_frames);
    
    for (int i = start_idx; i < static_cast<int>(history_.size()); ++i) {
        confidences.push_back(history_[i].confidence);
    }
    
    return confidences;
}

bool TrackingState::is_tracking_stable() const {
    if (!initialized_ || history_.size() < 3) {
        return false;
    }
    
    // Check confidence stability
    std::vector<float> recent_confidences = get_confidence_history(quality_window_size_);
    if (recent_confidences.empty()) {
        return false;
    }
    
    float avg_confidence = std::accumulate(recent_confidences.begin(), recent_confidences.end(), 0.0f) 
                          / recent_confidences.size();
    
    return avg_confidence >= stability_threshold_;
}

float TrackingState::get_tracking_quality() const {
    if (!initialized_ || history_.empty()) {
        return 0.0f;
    }
    
    // Combine multiple quality metrics
    float confidence_quality = get_last_confidence();
    float stability_quality = is_tracking_stable() ? 1.0f : 0.5f;
    float trend_quality = std::max(0.0f, calculate_confidence_trend());
    
    // Weighted average
    return 0.5f * confidence_quality + 0.3f * stability_quality + 0.2f * trend_quality;
}

void TrackingState::set_history_length(int length) {
    max_history_length_ = std::max(1, length);
    
    // Trim existing history if needed
    while (static_cast<int>(history_.size()) > max_history_length_) {
        history_.pop_front();
    }
}

double TrackingState::get_tracking_duration() const {
    if (!initialized_) {
        return 0.0;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    return duration.count() / 1000.0;  // Convert to seconds
}

cv::Point2f TrackingState::get_velocity() const {
    if (!velocity_valid_) {
        update_cached_metrics();
    }
    return velocity_;
}

float TrackingState::get_scale_change_rate() const {
    if (!scale_rate_valid_) {
        update_cached_metrics();
    }
    return scale_change_rate_;
}

void TrackingState::update_cached_metrics() const {
    velocity_ = cv::Point2f(0, 0);
    scale_change_rate_ = 0.0f;
    
    if (history_.size() < 2) {
        velocity_valid_ = true;
        scale_rate_valid_ = true;
        return;
    }
    
    // Calculate velocity (pixels per frame)
    const auto& current = history_.back();
    const auto& previous = history_[history_.size() - 2];
    
    cv::Point2f current_center(current.bbox.x + current.bbox.width / 2.0f,
                              current.bbox.y + current.bbox.height / 2.0f);
    cv::Point2f previous_center(previous.bbox.x + previous.bbox.width / 2.0f,
                               previous.bbox.y + previous.bbox.height / 2.0f);
    
    velocity_ = current_center - previous_center;
    
    // Calculate scale change rate
    float current_area = current.bbox.width * current.bbox.height;
    float previous_area = previous.bbox.width * previous.bbox.height;
    
    if (previous_area > 0) {
        scale_change_rate_ = (current_area - previous_area) / previous_area;
    }
    
    velocity_valid_ = true;
    scale_rate_valid_ = true;
}

float TrackingState::calculate_bbox_overlap(const cv::Rect& bbox1, const cv::Rect& bbox2) const {
    cv::Rect intersection = bbox1 & bbox2;
    
    if (intersection.area() == 0) {
        return 0.0f;
    }
    
    cv::Rect union_rect = bbox1 | bbox2;
    return static_cast<float>(intersection.area()) / union_rect.area();
}

cv::Scalar TrackingState::calculate_mean_color(const cv::Mat& frame, const cv::Rect& bbox) const {
    // Clamp bbox to frame boundaries
    cv::Rect clamped_bbox = bbox;
    clamped_bbox.x = std::max(0, std::min(clamped_bbox.x, frame.cols - 1));
    clamped_bbox.y = std::max(0, std::min(clamped_bbox.y, frame.rows - 1));
    clamped_bbox.width = std::max(1, std::min(clamped_bbox.width, frame.cols - clamped_bbox.x));
    clamped_bbox.height = std::max(1, std::min(clamped_bbox.height, frame.rows - clamped_bbox.y));
    
    if (clamped_bbox.width <= 0 || clamped_bbox.height <= 0) {
        return cv::Scalar(0, 0, 0);
    }
    
    cv::Mat roi = frame(clamped_bbox);
    return cv::mean(roi);
}

float TrackingState::calculate_confidence_trend() const {
    if (history_.size() < 2) {
        return 0.0f;
    }
    
    // Calculate trend over recent frames
    int window_size = std::min(quality_window_size_, static_cast<int>(history_.size()));
    std::vector<float> recent_confidences = get_confidence_history(window_size);
    
    if (recent_confidences.size() < 2) {
        return 0.0f;
    }
    
    // Simple linear trend calculation
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int n = recent_confidences.size();
    
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i);
        float y = recent_confidences[i];
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    
    float denominator = n * sum_xx - sum_x * sum_x;
    if (std::abs(denominator) < 1e-6) {
        return 0.0f;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / denominator;
    
    // Normalize slope to [0, 1] range
    return std::max(-1.0f, std::min(1.0f, slope * 10.0f));  // Scale factor for normalization
}

bool TrackingState::validate_bbox_transition(const cv::Rect& prev_bbox, const cv::Rect& new_bbox) const {
    // Check for reasonable movement
    cv::Point2f prev_center(prev_bbox.x + prev_bbox.width / 2.0f,
                           prev_bbox.y + prev_bbox.height / 2.0f);
    cv::Point2f new_center(new_bbox.x + new_bbox.width / 2.0f,
                          new_bbox.y + new_bbox.height / 2.0f);
    
    float movement = cv::norm(new_center - prev_center);
    float max_movement = std::max(prev_bbox.width, prev_bbox.height) * 0.5f;  // 50% of bbox size
    
    if (movement > max_movement) {
        return false;  // Too much movement
    }
    
    // Check for reasonable scale change
    float prev_area = prev_bbox.width * prev_bbox.height;
    float new_area = new_bbox.width * new_bbox.height;
    
    if (prev_area > 0) {
        float scale_ratio = new_area / prev_area;
        if (scale_ratio < 0.5f || scale_ratio > 2.0f) {
            return false;  // Too much scale change
        }
    }
    
    return true;
}

void TrackingState::invalidate_cache() const {
    velocity_valid_ = false;
    scale_rate_valid_ = false;
}