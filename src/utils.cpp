#include "utils.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <chrono>
#include <iomanip>

namespace FEARUtils {

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string trim(const std::string& str) {
    const std::string whitespace = " \t\r\n";
    size_t start = str.find_first_not_of(whitespace);
    
    if (start == std::string::npos) {
        return "";
    }
    
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

bool string_to_bbox(const std::string& str, cv::Rect& bbox) {
    auto tokens = split(str, ',');
    if (tokens.size() != 4) {
        return false;
    }
    
    try {
        bbox.x = std::stoi(trim(tokens[0]));
        bbox.y = std::stoi(trim(tokens[1]));
        bbox.width = std::stoi(trim(tokens[2]));
        bbox.height = std::stoi(trim(tokens[3]));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string bbox_to_string(const cv::Rect& bbox) {
    return std::to_string(bbox.x) + "," + 
           std::to_string(bbox.y) + "," +
           std::to_string(bbox.width) + "," + 
           std::to_string(bbox.height);
}

bool file_exists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

bool create_directory(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string get_filename_without_extension(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of('.');
    
    size_t start = (last_slash == std::string::npos) ? 0 : last_slash + 1;
    size_t end = (last_dot == std::string::npos || last_dot < start) ? path.length() : last_dot;
    
    return path.substr(start, end - start);
}

std::string get_file_extension(const std::string& path) {
    size_t last_dot = path.find_last_of('.');
    if (last_dot == std::string::npos) {
        return "";
    }
    return path.substr(last_dot + 1);
}

std::string join_paths(const std::string& path1, const std::string& path2) {
    if (path1.empty()) return path2;
    if (path2.empty()) return path1;
    
    bool path1_ends_with_slash = (path1.back() == '/' || path1.back() == '\\');
    bool path2_starts_with_slash = (path2.front() == '/' || path2.front() == '\\');
    
    if (path1_ends_with_slash && path2_starts_with_slash) {
        return path1 + path2.substr(1);
    } else if (!path1_ends_with_slash && !path2_starts_with_slash) {
        return path1 + "/" + path2;
    } else {
        return path1 + path2;
    }
}

void draw_bbox(cv::Mat& image, const cv::Rect& bbox, const cv::Scalar& color, int thickness) {
    cv::rectangle(image, bbox, color, thickness);
}

void draw_text(cv::Mat& image, const std::string& text, const cv::Point& position,
               const cv::Scalar& color, double font_scale, int thickness) {
    // Draw text background for better visibility
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    
    cv::Point bg_top_left = cv::Point(position.x, position.y - text_size.height - baseline);
    cv::Point bg_bottom_right = cv::Point(position.x + text_size.width, position.y + baseline);
    
    cv::rectangle(image, bg_top_left, bg_bottom_right, cv::Scalar(0, 0, 0), cv::FILLED);
    
    // Draw text
    cv::putText(image, text, position, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

void draw_tracking_info(cv::Mat& image, const cv::Rect& bbox, float confidence, int frame_number) {
    // Draw bounding box
    cv::Scalar bbox_color = (confidence > 0.5) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    draw_bbox(image, bbox, bbox_color, 2);
    
    // Prepare text
    std::ostringstream text_stream;
    text_stream << "Conf: " << std::fixed << std::setprecision(3) << confidence;
    
    if (frame_number >= 0) {
        text_stream << " | Frame: " << frame_number;
    }
    
    // Draw confidence text
    cv::Point text_pos(bbox.x, bbox.y - 10);
    if (text_pos.y < 20) {
        text_pos.y = bbox.y + bbox.height + 20;
    }
    
    draw_text(image, text_stream.str(), text_pos, cv::Scalar(255, 255, 255));
    
    // Draw center point
    cv::Point2f center = calculate_center(bbox);
    cv::circle(image, center, 3, bbox_color, cv::FILLED);
}

double calculate_iou(const cv::Rect& bbox1, const cv::Rect& bbox2) {
    cv::Rect intersection = bbox1 & bbox2;
    if (intersection.area() == 0) {
        return 0.0;
    }
    
    cv::Rect union_rect = bbox1 | bbox2;
    return static_cast<double>(intersection.area()) / union_rect.area();
}

cv::Point2f calculate_center(const cv::Rect& bbox) {
    return cv::Point2f(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);
}

double calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return cv::norm(p1 - p2);
}

cv::Rect scale_bbox(const cv::Rect& bbox, float scale_factor) {
    cv::Point2f center = calculate_center(bbox);
    int new_width = static_cast<int>(bbox.width * scale_factor);
    int new_height = static_cast<int>(bbox.height * scale_factor);
    
    return cv::Rect(
        static_cast<int>(center.x - new_width / 2.0f),
        static_cast<int>(center.y - new_height / 2.0f),
        new_width,
        new_height
    );
}

cv::Rect clamp_bbox(const cv::Rect& bbox, const cv::Size& frame_size) {
    cv::Rect clamped = bbox;
    
    // Clamp position
    clamped.x = std::max(0, std::min(clamped.x, frame_size.width - 1));
    clamped.y = std::max(0, std::min(clamped.y, frame_size.height - 1));
    
    // Clamp size
    clamped.width = std::max(1, std::min(clamped.width, frame_size.width - clamped.x));
    clamped.height = std::max(1, std::min(clamped.height, frame_size.height - clamped.y));
    
    return clamped;
}

bool get_video_info(const std::string& video_path, int& width, int& height, 
                   double& fps, int& frame_count) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        return false;
    }
    
    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = cap.get(cv::CAP_PROP_FPS);
    frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    cap.release();
    return true;
}

cv::VideoWriter create_video_writer(const std::string& output_path, 
                                   const cv::Size& frame_size, double fps) {
    // Use MP4V codec for better compatibility
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    return cv::VideoWriter(output_path, fourcc, fps, frame_size);
}

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

double get_system_time_ms() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

void print_system_info() {
    struct utsname info;
    if (uname(&info) == 0) {
        std::cout << "System Information:" << std::endl;
        std::cout << "  OS: " << info.sysname << " " << info.release << std::endl;
        std::cout << "  Architecture: " << info.machine << std::endl;
        std::cout << "  Hostname: " << info.nodename << std::endl;
    }
    
    // Try to get Pi5 specific info
    std::ifstream model_file("/proc/device-tree/model");
    if (model_file.is_open()) {
        std::string model;
        std::getline(model_file, model);
        // Remove null characters
        model.erase(std::find(model.begin(), model.end(), '\0'), model.end());
        std::cout << "  Device Model: " << model << std::endl;
    }
    
    // CPU info
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("Hardware") != std::string::npos) {
                std::cout << "  " << line << std::endl;
                break;
            }
        }
    }
}

bool parse_args(int argc, char* argv[], std::map<std::string, std::string>& args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.rfind("--", 0) == 0) {
            std::string key = arg.substr(2);
            std::string value = "";

            // Check if there's a value
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                value = argv[++i];
            }

            args[key] = value;
        } else if (arg.rfind("-", 0) == 0) {
            std::string key = arg.substr(1);
            std::string value = "";
            
            // Check if there's a value
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                value = argv[++i];
            }
            
            args[key] = value;
        }
    }
    
    return true;
}

void print_usage(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "FEARTracker for Raspberry Pi 5\n\n";
    std::cout << "Input Options (choose one):\n";
    std::cout << "  -i, --input VIDEO        Input video file\n";
    std::cout << "  --camera                 Use CSI camera input\n";
    std::cout << "  --rtp                    Receive H265-over-RTP UDP stream\n";
    std::cout << "\nRTP Options:\n";
    std::cout << "  --rtp-port N             UDP port to listen on (default: 5004)\n";
    std::cout << "  --rtp-latency MS         Jitter buffer latency in ms (default: 50)\n";
    std::cout << "  --rtp-payload N          RTP dynamic payload type (default: 96)\n";
    std::cout << "  --rtp-pipeline STR       Full GStreamer pipeline override\n";
    std::cout << "\nCamera Options:\n";
    std::cout << "  --camera-backend BACKEND Camera backend: auto, libcamera, gstreamer (default: auto)\n";
    std::cout << "  --camera-width W         Camera capture width (default: 1280)\n";
    std::cout << "  --camera-height H        Camera capture height (default: 720)\n";
    std::cout << "  --camera-fps F           Camera target framerate (default: 30)\n";
    std::cout << "  --camera-id ID           Camera index or name (default: 0)\n";
    std::cout << "  --list-cameras           List available cameras and exit\n";
    std::cout << "\nTracking Options:\n";
    std::cout << "  -b, --bbox X,Y,W,H       Initial bounding box (required)\n";
    std::cout << "  --select-bbox            Override -b: drag the bbox interactively on the first frame\n";
    std::cout << "  -t, --template MODEL     Template model path\n";
    std::cout << "  -s, --search MODEL       Search model path\n";
    std::cout << "\nOutput Options:\n";
    std::cout << "  -o, --output VIDEO       Output video file (optional)\n";
    std::cout << "  --save-frames            Save individual frames to output dir\n";
    std::cout << "  --output-dir DIR         Output directory (default: output)\n";
    std::cout << "\nDisplay Options:\n";
    std::cout << "  --display                Show real-time visualization window\n";
    std::cout << "  --benchmark              Enable performance monitoring\n";
    std::cout << "  --fps-limit N            Limit processing to N FPS\n";
    std::cout << "  --threads N              Number of inference threads\n";
    std::cout << "  -h, --help               Show this help message\n";
    std::cout << "\nExamples:\n";
    std::cout << "  # Track in video file\n";
    std::cout << "  " << program_name << " -i video.mp4 -b 100,50,80,120 --benchmark\n\n";
    std::cout << "  # Track with CSI camera (auto-detect backend)\n";
    std::cout << "  " << program_name << " --camera -b 100,50,80,120 --display\n\n";
    std::cout << "  # Track with camera, save output video\n";
    std::cout << "  " << program_name << " --camera --camera-backend libcamera -o output.mp4 -b 100,50,80,120\n\n";
    std::cout << "  # List available cameras\n";
    std::cout << "  " << program_name << " --list-cameras\n\n";
    std::cout << "  # Track an incoming H265-over-RTP stream on UDP port 5004\n";
    std::cout << "  " << program_name << " --rtp --rtp-port 5004 -b 100,50,80,120 --display\n\n";
    std::cout << "  # Pick the bbox interactively on the first frame of an RTP stream\n";
    std::cout << "  " << program_name << " --rtp --rtp-port 5004 -b 0,0,1,1 --select-bbox --display\n";
}

} // namespace FEARUtils