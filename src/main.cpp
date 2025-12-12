#include "fear_tracker.h"
#include "utils.h"
#include "build_info.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <csignal>

// Global flag for graceful shutdown
volatile bool g_shutdown = false;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    g_shutdown = true;
}

struct AppConfig {
    std::string input_video;
    std::string output_video;
    std::string template_model = "models/fear_net_template.onnx";
    std::string search_model = "models/fear_net_search.onnx";
    cv::Rect initial_bbox;
    bool benchmark = false;
    bool display = false;
    bool save_frames = false;
    bool help = false;
    int fps_limit = 0;  // 0 = no limit
    int threads = 4;
    std::string output_dir = "output";
};

bool parse_arguments(int argc, char* argv[], AppConfig& config) {
    std::map<std::string, std::string> args;
    if (!FEARUtils::parse_args(argc, argv, args)) {
        return false;
    }
    
    // Parse arguments
    for (const auto& [key, value] : args) {
        if (key == "i" || key == "input") {
            config.input_video = value;
        } else if (key == "o" || key == "output") {
            config.output_video = value;
        } else if (key == "t" || key == "template") {
            config.template_model = value;
        } else if (key == "s" || key == "search") {
            config.search_model = value;
        } else if (key == "b" || key == "bbox") {
            if (!FEARUtils::string_to_bbox(value, config.initial_bbox)) {
                std::cerr << "Invalid bbox format. Use: x,y,w,h" << std::endl;
                return false;
            }
        } else if (key == "benchmark") {
            config.benchmark = true;
        } else if (key == "display") {
            config.display = true;
        } else if (key == "save-frames") {
            config.save_frames = true;
        } else if (key == "fps-limit") {
            config.fps_limit = std::stoi(value);
        } else if (key == "threads") {
            config.threads = std::stoi(value);
        } else if (key == "output-dir") {
            config.output_dir = value;
        } else if (key == "h" || key == "help") {
            config.help = true;
        } else {
            std::cerr << "Unknown argument: " << key << std::endl;
            return false;
        }
    }
    
    return true;
}

bool validate_config(const AppConfig& config) {
    if (config.help) {
        return true;  // Help doesn't need validation
    }
    
    if (config.input_video.empty()) {
        std::cerr << "Input video is required" << std::endl;
        return false;
    }
    
    if (!FEARUtils::file_exists(config.input_video)) {
        std::cerr << "Input video not found: " << config.input_video << std::endl;
        return false;
    }
    
    if (!FEARUtils::file_exists(config.template_model)) {
        std::cerr << "Template model not found: " << config.template_model << std::endl;
        return false;
    }
    
    if (!FEARUtils::file_exists(config.search_model)) {
        std::cerr << "Search model not found: " << config.search_model << std::endl;
        return false;
    }
    
    if (config.initial_bbox.area() == 0) {
        std::cerr << "Initial bounding box is required" << std::endl;
        return false;
    }
    
    return true;
}

void print_version_info() {
    std::cout << "FEARTracker Pi v" << FEARTRACKER_VERSION_STRING << std::endl;
    std::cout << "Build: " << FEARTRACKER_BUILD_TYPE << " (" 
              << FEARTRACKER_BUILD_DATE << " " << FEARTRACKER_BUILD_TIME << ")" << std::endl;
    
    FEARUtils::print_system_info();
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Parse command line arguments
    AppConfig config;
    if (!parse_arguments(argc, argv, config)) {
        FEARUtils::print_usage(argv[0]);
        return 1;
    }
    
    if (config.help) {
        print_version_info();
        FEARUtils::print_usage(argv[0]);
        return 0;
    }
    
    if (!validate_config(config)) {
        std::cerr << "Configuration validation failed" << std::endl;
        FEARUtils::print_usage(argv[0]);
        return 1;
    }
    
    // Print startup information
    print_version_info();
    std::cout << "Input: " << config.input_video << std::endl;
    std::cout << "Template model: " << config.template_model << std::endl;
    std::cout << "Search model: " << config.search_model << std::endl;
    std::cout << "Initial bbox: " << FEARUtils::bbox_to_string(config.initial_bbox) << std::endl;
    
    try {
        // Initialize tracker
        std::cout << "Initializing FEAR tracker..." << std::endl;
        FEARTracker tracker;
        
        if (!tracker.initialize(config.template_model, config.search_model)) {
            std::cerr << "Failed to initialize tracker" << std::endl;
            return 1;
        }
        
        // Enable performance monitoring if requested
        if (config.benchmark) {
            tracker.enable_performance_monitoring(true);
            std::cout << "Performance monitoring enabled" << std::endl;
        }
        
        // Open input video
        std::cout << "Opening video: " << config.input_video << std::endl;
        cv::VideoCapture cap(config.input_video);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video: " << config.input_video << std::endl;
            return 1;
        }
        
        // Get video properties
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        std::cout << "Video info: " << frame_width << "x" << frame_height 
                  << " @ " << fps << " FPS, " << total_frames << " frames" << std::endl;
        
        // Setup output video writer if needed
        cv::VideoWriter writer;
        if (!config.output_video.empty()) {
            FEARUtils::create_directory(FEARUtils::get_filename_without_extension(config.output_video));
            writer = FEARUtils::create_video_writer(config.output_video, 
                                                   cv::Size(frame_width, frame_height), fps);
            if (!writer.isOpened()) {
                std::cerr << "Failed to create output video: " << config.output_video << std::endl;
                return 1;
            }
            std::cout << "Output: " << config.output_video << std::endl;
        }
        
        // Setup frame saving directory
        std::string frames_dir;
        if (config.save_frames) {
            frames_dir = FEARUtils::join_paths(config.output_dir, "frames");
            FEARUtils::create_directory(frames_dir);
            std::cout << "Saving frames to: " << frames_dir << std::endl;
        }
        
        // Read first frame and initialize tracking
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read first frame" << std::endl;
            return 1;
        }
        
        // Validate initial bbox
        cv::Rect clamped_bbox = FEARUtils::clamp_bbox(config.initial_bbox, frame.size());
        if (clamped_bbox != config.initial_bbox) {
            std::cout << "Initial bbox clamped to frame bounds: " 
                      << FEARUtils::bbox_to_string(clamped_bbox) << std::endl;
        }
        
        // Start tracking
        std::cout << "Starting tracking..." << std::endl;
        if (!tracker.start_tracking(frame, clamped_bbox)) {
            std::cerr << "Failed to start tracking" << std::endl;
            return 1;
        }
        
        // Main tracking loop
        int frame_number = 0;
        auto loop_start = std::chrono::steady_clock::now();
        
        // FPS limiting
        std::chrono::microseconds frame_duration(0);
        if (config.fps_limit > 0) {
            frame_duration = std::chrono::microseconds(1000000 / config.fps_limit);
        }
        
        while (!g_shutdown) {
            auto frame_start = std::chrono::steady_clock::now();
            
            // Read next frame
            if (!cap.read(frame) || frame.empty()) {
                break;
            }
            
            frame_number++;
            
            // Update tracker
            TrackingResult result = tracker.update(frame);
            
            if (!result.valid) {
                std::cerr << "Tracking failed at frame " << frame_number << std::endl;
                break;
            }
            
            // Draw visualization
            cv::Mat vis_frame = frame.clone();
            FEARUtils::draw_tracking_info(vis_frame, result.bbox, result.confidence, frame_number);
            
            // Display frame
            if (config.display) {
                cv::imshow("FEARTracker Pi", vis_frame);
                char key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 27) {  // 'q' or ESC
                    break;
                }
            }
            
            // Save output video
            if (writer.isOpened()) {
                writer.write(vis_frame);
            }
            
            // Save individual frames
            if (config.save_frames) {
                std::string frame_filename = "frame_" + 
                    std::to_string(frame_number) + ".jpg";
                std::string frame_path = FEARUtils::join_paths(frames_dir, frame_filename);
                cv::imwrite(frame_path, vis_frame);
            }
            
            // Progress reporting
            if (frame_number % 50 == 0) {
                double progress = static_cast<double>(frame_number) / total_frames * 100.0;
                std::cout << "Progress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (Frame " << frame_number << "/" << total_frames << ")";
                
                if (config.benchmark) {
                    // Get current performance stats
                    auto elapsed = std::chrono::steady_clock::now() - loop_start;
                    double elapsed_sec = std::chrono::duration<double>(elapsed).count();
                    double current_fps = frame_number / elapsed_sec;
                    
                    std::cout << " - FPS: " << std::fixed << std::setprecision(2) << current_fps;
                    std::cout << " - Conf: " << std::fixed << std::setprecision(3) << result.confidence;
                }
                
                std::cout << std::endl;
            }
            
            // FPS limiting
            if (config.fps_limit > 0) {
                auto frame_end = std::chrono::steady_clock::now();
                auto elapsed = frame_end - frame_start;
                
                if (elapsed < frame_duration) {
                    std::this_thread::sleep_for(frame_duration - elapsed);
                }
            }
        }
        
        // Cleanup
        cap.release();
        if (writer.isOpened()) {
            writer.release();
        }
        if (config.display) {
            cv::destroyAllWindows();
        }
        
        // Final statistics
        auto total_elapsed = std::chrono::steady_clock::now() - loop_start;
        double total_time = std::chrono::duration<double>(total_elapsed).count();
        double avg_fps = frame_number / total_time;
        
        std::cout << std::endl << "Tracking Complete!" << std::endl;
        std::cout << "Processed " << frame_number << " frames in " 
                  << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
        
        // Print performance summary
        if (config.benchmark) {
            std::cout << std::endl << tracker.get_performance_stats() << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}