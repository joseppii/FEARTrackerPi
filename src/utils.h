#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FEARUtils {

// String utilities
std::vector<std::string> split(const std::string& str, char delimiter);
std::string trim(const std::string& str);
bool string_to_bbox(const std::string& str, cv::Rect& bbox);
std::string bbox_to_string(const cv::Rect& bbox);

// File and path utilities
bool file_exists(const std::string& path);
bool create_directory(const std::string& path);
std::string get_filename_without_extension(const std::string& path);
std::string get_file_extension(const std::string& path);
std::string join_paths(const std::string& path1, const std::string& path2);

// Image utilities
void draw_bbox(cv::Mat& image, const cv::Rect& bbox, 
               const cv::Scalar& color = cv::Scalar(0, 255, 0), 
               int thickness = 2);
void draw_text(cv::Mat& image, const std::string& text, const cv::Point& position,
               const cv::Scalar& color = cv::Scalar(255, 255, 255),
               double font_scale = 0.6, int thickness = 2);
void draw_tracking_info(cv::Mat& image, const cv::Rect& bbox, float confidence,
                       int frame_number = -1);

// Math utilities
double calculate_iou(const cv::Rect& bbox1, const cv::Rect& bbox2);
cv::Point2f calculate_center(const cv::Rect& bbox);
double calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2);
cv::Rect scale_bbox(const cv::Rect& bbox, float scale_factor);
cv::Rect clamp_bbox(const cv::Rect& bbox, const cv::Size& frame_size);

// Video utilities
bool get_video_info(const std::string& video_path, int& width, int& height, 
                   double& fps, int& frame_count);
cv::VideoWriter create_video_writer(const std::string& output_path, 
                                   const cv::Size& frame_size, double fps);

// System utilities
std::string get_current_timestamp();
double get_system_time_ms();
void print_system_info();

// Configuration utilities
bool parse_args(int argc, char* argv[], std::map<std::string, std::string>& args);
void print_usage(const std::string& program_name);

} // namespace FEARUtils