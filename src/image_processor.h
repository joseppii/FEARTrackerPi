#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct InferenceResult; // Forward declaration

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();
    
    // Template extraction
    bool extract_template_crop(const cv::Mat& frame, 
                              const cv::Rect& bbox, 
                              int template_size,
                              float template_offset,
                              cv::Mat& template_crop,
                              cv::Rect& template_bbox) const;
    
    // Search region extraction
    bool extract_search_crop(const cv::Mat& frame,
                            const cv::Rect& bbox,
                            int instance_size,
                            float search_context,
                            const cv::Scalar& padding_value,
                            cv::Mat& search_crop,
                            cv::Rect& search_region,
                            cv::Point2f& search_center) const;
    
    // Post-processing
    bool postprocess_inference_result(const InferenceResult& result,
                                    const cv::Rect& search_region,
                                    const cv::Point2f& search_center,
                                    cv::Rect& predicted_bbox,
                                    float& confidence) const;
    
    // Utility functions
    cv::Mat apply_padding(const cv::Mat& image, 
                         const cv::Rect& crop_rect, 
                         const cv::Size& frame_size,
                         const cv::Scalar& padding_value) const;
    
    cv::Rect calculate_crop_region(const cv::Rect& bbox, 
                                  float context_factor, 
                                  const cv::Size& frame_size) const;
    
    cv::Point2f get_bbox_center(const cv::Rect& bbox) const;
    
    // Configuration
    void set_score_size(int score_size) { score_size_ = score_size; }
    void set_total_stride(int total_stride) { total_stride_ = total_stride; }
    void set_smoothing_enabled(bool enabled) { smoothing_enabled_ = enabled; }
    void set_smoothing_lr(float lr) { smoothing_lr_ = lr; }

private:
    // Configuration parameters
    int score_size_;
    int total_stride_;
    bool smoothing_enabled_;
    float smoothing_lr_;
    
    // Grid coordinates for postprocessing
    mutable std::vector<float> grid_x_;
    mutable std::vector<float> grid_y_;
    mutable bool grids_initialized_;
    
    // Helper methods
    void initialize_grids() const;
    void apply_confidence_penalty(std::vector<float>& cls_score, 
                                 const std::vector<float>& regression_map) const;
    
    cv::Point find_best_location(const std::vector<float>& cls_score) const;
    
    cv::Rect decode_bbox_regression(const std::vector<float>& regression_map,
                                   const cv::Point& best_location,
                                   const cv::Point2f& search_center,
                                   const cv::Rect& search_region) const;
    
    cv::Rect apply_bbox_smoothing(const cv::Rect& predicted_bbox,
                                 const cv::Rect& previous_bbox,
                                 float confidence) const;
    
    // Padding utilities
    cv::Mat pad_image(const cv::Mat& image, 
                     int top, int bottom, int left, int right,
                     const cv::Scalar& value) const;
    
    cv::Rect adjust_crop_bounds(const cv::Rect& desired_crop,
                               const cv::Size& frame_size,
                               int& pad_top, int& pad_bottom,
                               int& pad_left, int& pad_right) const;
};