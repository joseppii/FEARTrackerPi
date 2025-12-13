#include "image_processor.h"
#include "onnx_inference.h"
#include <iostream>
#include <algorithm>
#include <cmath>

ImageProcessor::ImageProcessor()
    : score_size_(16)
    , total_stride_(16)
    , smoothing_enabled_(true)
    , smoothing_lr_(0.4f)
    , grids_initialized_(false)
{
}

ImageProcessor::~ImageProcessor() = default;

bool ImageProcessor::extract_template_crop(const cv::Mat& frame,
                                          const cv::Rect& bbox,
                                          int template_size,
                                          float template_offset,
                                          cv::Mat& template_crop,
                                          cv::Rect& template_bbox) const {
    if (frame.empty() || template_size <= 0) {
        return false;
    }

    // Match Python extend_bbox: extends by offset fraction on each side
    // extended = [x - w*offset, y - h*offset, w*(1+2*offset), h*(1+2*offset)]
    int ext_x = static_cast<int>(bbox.x - bbox.width * template_offset);
    int ext_y = static_cast<int>(bbox.y - bbox.height * template_offset);
    int ext_w = static_cast<int>(bbox.width * (1.0f + 2.0f * template_offset));
    int ext_h = static_cast<int>(bbox.height * (1.0f + 2.0f * template_offset));

    template_bbox = cv::Rect(ext_x, ext_y, ext_w, ext_h);

    // Handle padding if crop extends beyond frame
    cv::Scalar mean_color = cv::mean(frame);
    template_crop = apply_padding(frame, template_bbox, frame.size(), mean_color);

    if (template_crop.empty()) {
        return false;
    }

    // Resize to target template size
    cv::resize(template_crop, template_crop, cv::Size(template_size, template_size));

    return true;
}

bool ImageProcessor::extract_search_crop(const cv::Mat& frame,
                                        const cv::Rect& bbox,
                                        int instance_size,
                                        float search_context,
                                        const cv::Scalar& padding_value,
                                        cv::Mat& search_crop,
                                        cv::Rect& search_region,
                                        cv::Point2f& search_center) const {
    if (frame.empty() || instance_size <= 0) {
        return false;
    }

    // Match Python extend_bbox: extends by offset fraction on each side
    // For search_context=2: extended = [x - 2w, y - 2h, 5w, 5h]
    int ext_x = static_cast<int>(bbox.x - bbox.width * search_context);
    int ext_y = static_cast<int>(bbox.y - bbox.height * search_context);
    int ext_w = static_cast<int>(bbox.width * (1.0f + 2.0f * search_context));
    int ext_h = static_cast<int>(bbox.height * (1.0f + 2.0f * search_context));

    search_region = cv::Rect(ext_x, ext_y, ext_w, ext_h);
    search_center = get_bbox_center(bbox);

    // Extract search crop with padding
    search_crop = apply_padding(frame, search_region, frame.size(), padding_value);

    if (search_crop.empty()) {
        return false;
    }

    // Resize to target instance size
    cv::resize(search_crop, search_crop, cv::Size(instance_size, instance_size));

    return true;
}

bool ImageProcessor::postprocess_inference_result(const InferenceResult& result,
                                                 const cv::Rect& search_region,
                                                 const cv::Point2f& search_center,
                                                 cv::Rect& predicted_bbox,
                                                 float& confidence) const {
    if (result.classification_map.empty() || result.regression_map.empty()) {
        return false;
    }
    
    // Initialize grids if needed
    if (!grids_initialized_) {
        initialize_grids();
    }
    
    // Apply sigmoid to classification scores
    std::vector<float> cls_score = result.classification_map;
    for (float& score : cls_score) {
        score = 1.0f / (1.0f + std::exp(-score));
    }
    
    // Apply confidence penalty (simplified)
    apply_confidence_penalty(cls_score, result.regression_map);
    
    // Find best location
    cv::Point best_location = find_best_location(cls_score);
    
    // Get confidence at best location
    int best_idx = best_location.y * score_size_ + best_location.x;
    confidence = cls_score[best_idx];
    
    // Decode bounding box
    predicted_bbox = decode_bbox_regression(result.regression_map, best_location, 
                                          search_center, search_region);
    
    return true;
}

cv::Mat ImageProcessor::apply_padding(const cv::Mat& image, 
                                     const cv::Rect& crop_rect, 
                                     const cv::Size& frame_size,
                                     const cv::Scalar& padding_value) const {
    int pad_top, pad_bottom, pad_left, pad_right;
    cv::Rect adjusted_crop = adjust_crop_bounds(crop_rect, frame_size, 
                                              pad_top, pad_bottom, pad_left, pad_right);
    
    // Extract the valid region
    cv::Mat cropped = image(adjusted_crop);
    
    // Apply padding if needed
    if (pad_top > 0 || pad_bottom > 0 || pad_left > 0 || pad_right > 0) {
        return pad_image(cropped, pad_top, pad_bottom, pad_left, pad_right, padding_value);
    }
    
    return cropped;
}

cv::Rect ImageProcessor::calculate_crop_region(const cv::Rect& bbox, 
                                              float context_factor, 
                                              const cv::Size& frame_size) const {
    cv::Point2f center = get_bbox_center(bbox);
    float max_side = std::max(bbox.width, bbox.height);
    float crop_side = max_side * context_factor;
    
    cv::Rect crop_region(
        static_cast<int>(center.x - crop_side / 2),
        static_cast<int>(center.y - crop_side / 2),
        static_cast<int>(crop_side),
        static_cast<int>(crop_side)
    );
    
    // Clamp to frame boundaries
    crop_region.x = std::max(0, crop_region.x);
    crop_region.y = std::max(0, crop_region.y);
    crop_region.width = std::min(crop_region.width, frame_size.width - crop_region.x);
    crop_region.height = std::min(crop_region.height, frame_size.height - crop_region.y);
    
    return crop_region;
}

cv::Point2f ImageProcessor::get_bbox_center(const cv::Rect& bbox) const {
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
}

void ImageProcessor::initialize_grids() const {
    grid_x_.clear();
    grid_y_.clear();
    grid_x_.reserve(score_size_ * score_size_);
    grid_y_.reserve(score_size_ * score_size_);

    // Match Python: grid = (idx - score_size//2) * total_stride + instance_size//2
    // For score_size=16, total_stride=16, instance_size=256:
    // idx 0: (0-8)*16 + 128 = 0
    // idx 15: (15-8)*16 + 128 = 240
    const int instance_size = 256;
    const int half_score = score_size_ / 2;

    for (int y = 0; y < score_size_; ++y) {
        for (int x = 0; x < score_size_; ++x) {
            grid_x_.push_back((x - half_score) * total_stride_ + instance_size / 2.0f);
            grid_y_.push_back((y - half_score) * total_stride_ + instance_size / 2.0f);
        }
    }

    grids_initialized_ = true;
}

void ImageProcessor::apply_confidence_penalty(std::vector<float>& cls_score, 
                                             const std::vector<float>& regression_map) const {
    // Simple penalty based on bbox size changes
    // In a full implementation, this would include ratio and size penalties
    
    if (cls_score.size() != score_size_ * score_size_) {
        return;
    }
    
    // Apply a simple uniform penalty (this can be enhanced)
    const float penalty_factor = 0.98f;
    
    for (float& score : cls_score) {
        score *= penalty_factor;
    }
}

cv::Point ImageProcessor::find_best_location(const std::vector<float>& cls_score) const {
    if (cls_score.empty()) {
        return cv::Point(-1, -1);
    }
    
    auto max_it = std::max_element(cls_score.begin(), cls_score.end());
    int max_idx = std::distance(cls_score.begin(), max_it);
    
    int best_y = max_idx / score_size_;
    int best_x = max_idx % score_size_;
    
    return cv::Point(best_x, best_y);
}

cv::Rect ImageProcessor::decode_bbox_regression(const std::vector<float>& regression_map,
                                               const cv::Point& best_location,
                                               const cv::Point2f& search_center,
                                               const cv::Rect& search_region) const {
    if (regression_map.size() != 4 * score_size_ * score_size_) {
        std::cerr << "Invalid regression map size" << std::endl;
        return cv::Rect();
    }
    
    // Extract regression values at best location
    int base_idx = best_location.y * score_size_ + best_location.x;
    int stride = score_size_ * score_size_;
    
    float reg_left = regression_map[base_idx];
    float reg_top = regression_map[base_idx + stride];
    float reg_right = regression_map[base_idx + 2 * stride];
    float reg_bottom = regression_map[base_idx + 3 * stride];
    
    // Get grid position
    if (!grids_initialized_) {
        initialize_grids();
    }
    
    float grid_x = grid_x_[base_idx];
    float grid_y = grid_y_[base_idx];
    
    // Decode to crop coordinates
    float x1 = grid_x - reg_left;
    float y1 = grid_y - reg_top;
    float x2 = grid_x + reg_right;
    float y2 = grid_y + reg_bottom;
    
    // Transform from search crop coordinates to original image coordinates
    float scale_x = static_cast<float>(search_region.width) / 256.0f;  // Assuming 256x256 search
    float scale_y = static_cast<float>(search_region.height) / 256.0f;
    
    cv::Rect predicted_bbox;
    predicted_bbox.x = static_cast<int>(x1 * scale_x + search_region.x);
    predicted_bbox.y = static_cast<int>(y1 * scale_y + search_region.y);
    predicted_bbox.width = static_cast<int>((x2 - x1) * scale_x);
    predicted_bbox.height = static_cast<int>((y2 - y1) * scale_y);
    
    return predicted_bbox;
}

cv::Rect ImageProcessor::apply_bbox_smoothing(const cv::Rect& predicted_bbox,
                                             const cv::Rect& previous_bbox,
                                             float confidence) const {
    if (!smoothing_enabled_ || previous_bbox.area() == 0) {
        return predicted_bbox;
    }
    
    // Simple linear interpolation based on confidence
    float lr = smoothing_lr_ * confidence;
    
    cv::Rect smoothed_bbox;
    smoothed_bbox.x = static_cast<int>(lr * predicted_bbox.x + (1 - lr) * previous_bbox.x);
    smoothed_bbox.y = static_cast<int>(lr * predicted_bbox.y + (1 - lr) * previous_bbox.y);
    smoothed_bbox.width = static_cast<int>(lr * predicted_bbox.width + (1 - lr) * previous_bbox.width);
    smoothed_bbox.height = static_cast<int>(lr * predicted_bbox.height + (1 - lr) * previous_bbox.height);
    
    return smoothed_bbox;
}

cv::Mat ImageProcessor::pad_image(const cv::Mat& image, 
                                 int top, int bottom, int left, int right,
                                 const cv::Scalar& value) const {
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, top, bottom, left, right, cv::BORDER_CONSTANT, value);
    return padded;
}

cv::Rect ImageProcessor::adjust_crop_bounds(const cv::Rect& desired_crop,
                                           const cv::Size& frame_size,
                                           int& pad_top, int& pad_bottom,
                                           int& pad_left, int& pad_right) const {
    cv::Rect adjusted_crop = desired_crop;
    
    // Initialize padding
    pad_top = pad_bottom = pad_left = pad_right = 0;
    
    // Check left boundary
    if (adjusted_crop.x < 0) {
        pad_left = -adjusted_crop.x;
        adjusted_crop.width += adjusted_crop.x;
        adjusted_crop.x = 0;
    }
    
    // Check top boundary
    if (adjusted_crop.y < 0) {
        pad_top = -adjusted_crop.y;
        adjusted_crop.height += adjusted_crop.y;
        adjusted_crop.y = 0;
    }
    
    // Check right boundary
    if (adjusted_crop.x + adjusted_crop.width > frame_size.width) {
        pad_right = (adjusted_crop.x + adjusted_crop.width) - frame_size.width;
        adjusted_crop.width = frame_size.width - adjusted_crop.x;
    }
    
    // Check bottom boundary
    if (adjusted_crop.y + adjusted_crop.height > frame_size.height) {
        pad_bottom = (adjusted_crop.y + adjusted_crop.height) - frame_size.height;
        adjusted_crop.height = frame_size.height - adjusted_crop.y;
    }
    
    // Ensure valid crop region
    adjusted_crop.width = std::max(1, adjusted_crop.width);
    adjusted_crop.height = std::max(1, adjusted_crop.height);
    
    return adjusted_crop;
}