#include "fear_tracker.h"
#include "utils.h"
#include "video_source.h"
#include "build_info.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <csignal>
#include <algorithm>

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
    bool select_bbox = false;     // override -b with interactive selection on first frame
    bool benchmark = false;
    bool display = false;
    bool save_frames = false;
    bool help = false;
    int fps_limit = 0;  // 0 = no limit
    int threads = 4;
    std::string output_dir = "output";

    // Camera options
    bool use_camera = false;
    std::string camera_backend = "auto";  // "libcamera", "gstreamer", "auto"
    int camera_width = 1280;
    int camera_height = 720;
    int camera_fps = 30;
    std::string camera_id = "0";
    bool list_cameras = false;

    // RTP H265 options
    bool use_rtp = false;
    int rtp_port = 5004;
    int rtp_latency_ms = 50;
    int rtp_payload_type = 96;
    std::string rtp_pipeline;  // optional full pipeline override
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
        } else if (key == "select-bbox") {
            config.select_bbox = true;
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
        } else if (key == "camera") {
            config.use_camera = true;
        } else if (key == "camera-backend") {
            config.camera_backend = value;
        } else if (key == "camera-width") {
            config.camera_width = std::stoi(value);
        } else if (key == "camera-height") {
            config.camera_height = std::stoi(value);
        } else if (key == "camera-fps") {
            config.camera_fps = std::stoi(value);
        } else if (key == "camera-id") {
            config.camera_id = value;
        } else if (key == "list-cameras") {
            config.list_cameras = true;
        } else if (key == "rtp") {
            config.use_rtp = true;
        } else if (key == "rtp-port") {
            config.rtp_port = std::stoi(value);
        } else if (key == "rtp-latency") {
            config.rtp_latency_ms = std::stoi(value);
        } else if (key == "rtp-payload") {
            config.rtp_payload_type = std::stoi(value);
        } else if (key == "rtp-pipeline") {
            config.rtp_pipeline = value;
        } else {
            std::cerr << "Unknown argument: " << key << std::endl;
            return false;
        }
    }
    
    return true;
}

bool validate_config(const AppConfig& config) {
    if (config.help || config.list_cameras) {
        return true;  // Help and list-cameras don't need validation
    }

    // Exactly one input mode must be specified
    const int input_modes = (config.use_camera ? 1 : 0)
                          + (config.use_rtp ? 1 : 0)
                          + (config.input_video.empty() ? 0 : 1);
    if (input_modes == 0) {
        std::cerr << "An input is required: -i <file>, --camera, or --rtp" << std::endl;
        return false;
    }
    if (input_modes > 1) {
        std::cerr << "Specify only one of -i, --camera, --rtp" << std::endl;
        return false;
    }

    // Validate file input
    if (!config.use_camera && !config.use_rtp && !FEARUtils::file_exists(config.input_video)) {
        std::cerr << "Input video not found: " << config.input_video << std::endl;
        return false;
    }

    // Validate RTP options
    if (config.use_rtp) {
        if (config.rtp_port <= 0 || config.rtp_port > 65535) {
            std::cerr << "Invalid --rtp-port: " << config.rtp_port << std::endl;
            return false;
        }
        if (config.rtp_latency_ms < 0) {
            std::cerr << "Invalid --rtp-latency: " << config.rtp_latency_ms << std::endl;
            return false;
        }
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

    // Validate camera backend
    if (config.use_camera) {
        std::string backend = config.camera_backend;
        std::transform(backend.begin(), backend.end(), backend.begin(), ::tolower);
        if (backend != "auto" && backend != "libcamera" && backend != "gstreamer" && backend != "v4l2") {
            std::cerr << "Invalid camera backend: " << config.camera_backend << std::endl;
            std::cerr << "Valid options: auto, libcamera, gstreamer, v4l2" << std::endl;
            return false;
        }
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

    // Handle --list-cameras
    if (config.list_cameras) {
        std::cout << "Available cameras:" << std::endl;
        auto cameras = list_available_cameras();
        if (cameras.empty()) {
            std::cout << "  (no cameras detected)" << std::endl;
        } else {
            for (size_t i = 0; i < cameras.size(); i++) {
                std::cout << "  [" << i << "] " << cameras[i] << std::endl;
            }
        }
        return 0;
    }

    if (!validate_config(config)) {
        std::cerr << "Configuration validation failed" << std::endl;
        FEARUtils::print_usage(argv[0]);
        return 1;
    }
    
    // Print startup information
    print_version_info();
    if (config.use_camera) {
        std::cout << "Input: Camera (backend=" << config.camera_backend
                  << ", " << config.camera_width << "x" << config.camera_height
                  << " @ " << config.camera_fps << " FPS)" << std::endl;
    } else if (config.use_rtp) {
        std::cout << "Input: RTP H265 (udp:" << config.rtp_port
                  << ", payload=" << config.rtp_payload_type
                  << ", latency=" << config.rtp_latency_ms << "ms)" << std::endl;
    } else {
        std::cout << "Input: " << config.input_video << std::endl;
    }
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

        // Create video source (file, camera, or RTP)
        std::unique_ptr<IVideoSource> source;
        if (config.use_camera) {
            VideoSourceConfig cam_config;
            cam_config.width = config.camera_width;
            cam_config.height = config.camera_height;
            cam_config.framerate = config.camera_fps;
            cam_config.camera_id = config.camera_id;

            std::cout << "Opening camera..." << std::endl;
            source = create_video_source("", true, config.camera_backend, cam_config);
        } else if (config.use_rtp) {
            RtpH265Config rtp_config;
            rtp_config.port = config.rtp_port;
            rtp_config.latency_ms = config.rtp_latency_ms;
            rtp_config.payload_type = config.rtp_payload_type;
            rtp_config.custom_pipeline = config.rtp_pipeline;

            std::cout << "Opening RTP H265 stream..." << std::endl;
            source = create_rtp_video_source(rtp_config);
        } else {
            std::cout << "Opening video: " << config.input_video << std::endl;
            source = create_video_source(config.input_video, false, "", VideoSourceConfig());
        }

        if (!source || !source->is_open()) {
            std::cerr << "Failed to open video source" << std::endl;
            return 1;
        }

        // Get video properties
        int frame_width = source->width();
        int frame_height = source->height();
        double fps = source->fps();
        int64_t total_frames = source->frame_count();  // -1 for cameras
        bool is_live = source->is_live();

        if (is_live) {
            std::cout << "Source info: " << source->description() << " - "
                      << frame_width << "x" << frame_height
                      << " @ " << fps << " FPS (live)" << std::endl;
        } else {
            std::cout << "Source info: " << frame_width << "x" << frame_height
                      << " @ " << fps << " FPS, " << total_frames << " frames" << std::endl;
        }
        
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
        if (!source->read(frame) || frame.empty()) {
            std::cerr << "Failed to read first frame" << std::endl;
            return 1;
        }

        // Interactive bbox selection (overrides -b when set). Requires a display
        // surface: in a headless SSH session without X forwarding this will fail.
        if (config.select_bbox) {
            std::cout << "Drag a bounding box on the first frame.\n"
                         "  ENTER or SPACE to confirm, c to cancel.\n";
            const std::string roi_window = "FEARTracker Pi";
            cv::Rect selected = cv::selectROI(roi_window, frame,
                                              /*showCrosshair=*/false,
                                              /*fromCenter=*/false);
            if (selected.width == 0 || selected.height == 0) {
                std::cerr << "Bbox selection cancelled — exiting." << std::endl;
                cv::destroyAllWindows();
                return 1;
            }
            config.initial_bbox = selected;
            std::cout << "Selected bbox: "
                      << FEARUtils::bbox_to_string(config.initial_bbox) << std::endl;
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
            if (!source->read(frame) || frame.empty()) {
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
                if (is_live) {
                    // Live camera - no progress percentage
                    std::cout << "Frame " << frame_number;
                } else {
                    // Video file - show progress percentage
                    double progress = static_cast<double>(frame_number) / total_frames * 100.0;
                    std::cout << "Progress: " << std::fixed << std::setprecision(1)
                             << progress << "% (Frame " << frame_number << "/" << total_frames << ")";
                }

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
        source->close();
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