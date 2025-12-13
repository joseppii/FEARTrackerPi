#include "fear_tracker.h"
#include "utils.h"
#include "build_info.h"
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>

struct BenchmarkConfig {
    std::string input_video = "assets/test.mp4";
    std::string template_model = "models/fear_net_template.onnx";
    std::string search_model = "models/fear_net_search.onnx";
    cv::Rect initial_bbox = cv::Rect(163, 53, 45, 174);
    std::string output_file = "benchmark_results.json";
    int warmup_frames = 10;
    int test_frames = 100;  // 0 = all frames
    bool stress_test = false;
    int stress_duration_min = 5;
};

struct BenchmarkResults {
    double avg_fps;
    double min_fps;
    double max_fps;
    double avg_confidence;
    double min_confidence;
    double max_confidence;
    int total_frames;
    double total_time_sec;
    double initialization_time_ms;
    double avg_inference_time_ms;
    double peak_memory_mb;
    double avg_temperature_c;
    double max_temperature_c;
    std::string system_info;
};

bool parse_bbox(const std::string& bbox_str, cv::Rect& bbox) {
    return FEARUtils::string_to_bbox(bbox_str, bbox);
}

void print_system_specs() {
    std::cout << "System Specifications:" << std::endl;
    std::cout << "=====================" << std::endl;
    
    FEARUtils::print_system_info();
    
    // Additional Pi5-specific info
    std::ifstream cmdline("/proc/cmdline");
    if (cmdline.is_open()) {
        std::string line;
        std::getline(cmdline, line);
        if (line.find("aarch64") != std::string::npos) {
            std::cout << "  ARM64 confirmed" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

BenchmarkResults run_performance_benchmark(const BenchmarkConfig& config) {
    BenchmarkResults results = {};
    
    // Initialize tracker
    FEARTracker tracker;
    tracker.enable_performance_monitoring(true);
    
    auto init_start = std::chrono::steady_clock::now();
    if (!tracker.initialize(config.template_model, config.search_model)) {
        throw std::runtime_error("Failed to initialize tracker");
    }
    auto init_end = std::chrono::steady_clock::now();
    results.initialization_time_ms = std::chrono::duration<double, std::milli>(init_end - init_start).count();
    
    // Open video
    cv::VideoCapture cap(config.input_video);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video: " + config.input_video);
    }
    
    // Read first frame and start tracking
    cv::Mat frame;
    if (!cap.read(frame)) {
        throw std::runtime_error("Failed to read first frame");
    }
    
    if (!tracker.start_tracking(frame, config.initial_bbox)) {
        throw std::runtime_error("Failed to start tracking");
    }
    
    // Warmup phase
    std::cout << "Warmup phase (" << config.warmup_frames << " frames)..." << std::endl;
    for (int i = 0; i < config.warmup_frames && cap.read(frame); i++) {
        tracker.update(frame);
    }
    
    // Benchmark phase
    std::cout << "Benchmark phase..." << std::endl;
    std::vector<double> frame_times;
    std::vector<double> confidences;
    std::vector<double> temperatures;
    
    auto benchmark_start = std::chrono::steady_clock::now();
    int frame_count = 0;
    
    while (cap.read(frame) && !frame.empty()) {
        auto frame_start = std::chrono::steady_clock::now();
        
        TrackingResult result = tracker.update(frame);
        if (!result.valid) {
            std::cerr << "Tracking failed at frame " << frame_count << std::endl;
            break;
        }
        
        auto frame_end = std::chrono::steady_clock::now();
        double frame_time = std::chrono::duration<double>(frame_end - frame_start).count();
        
        frame_times.push_back(1.0 / frame_time);  // Convert to FPS
        confidences.push_back(result.confidence);
        
        // Sample temperature every 10 frames
        if (frame_count % 10 == 0) {
            std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
            if (temp_file.is_open()) {
                int temp_millidegree;
                temp_file >> temp_millidegree;
                temperatures.push_back(temp_millidegree / 1000.0);
            }
        }
        
        frame_count++;
        
        // Progress update
        if (frame_count % 50 == 0) {
            std::cout << "  Processed " << frame_count << " frames..." << std::endl;
        }
        
        // Limit frames if specified
        if (config.test_frames > 0 && frame_count >= config.test_frames) {
            break;
        }
    }
    
    auto benchmark_end = std::chrono::steady_clock::now();
    results.total_time_sec = std::chrono::duration<double>(benchmark_end - benchmark_start).count();
    results.total_frames = frame_count;
    
    // Calculate statistics
    if (!frame_times.empty()) {
        results.avg_fps = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
        results.min_fps = *std::min_element(frame_times.begin(), frame_times.end());
        results.max_fps = *std::max_element(frame_times.begin(), frame_times.end());
    }
    
    if (!confidences.empty()) {
        results.avg_confidence = std::accumulate(confidences.begin(), confidences.end(), 0.0) / confidences.size();
        results.min_confidence = *std::min_element(confidences.begin(), confidences.end());
        results.max_confidence = *std::max_element(confidences.begin(), confidences.end());
    }
    
    if (!temperatures.empty()) {
        results.avg_temperature_c = std::accumulate(temperatures.begin(), temperatures.end(), 0.0) / temperatures.size();
        results.max_temperature_c = *std::max_element(temperatures.begin(), temperatures.end());
    }
    
    // Get inference time from performance monitor
    std::string perf_stats = tracker.get_performance_stats();
    // Parse the stats to extract inference time (simplified)
    results.avg_inference_time_ms = 1000.0 / results.avg_fps;  // Approximation
    
    cap.release();
    
    return results;
}

void run_stress_test(const BenchmarkConfig& config) {
    std::cout << "Starting stress test for " << config.stress_duration_min << " minutes..." << std::endl;
    
    FEARTracker tracker;
    tracker.enable_performance_monitoring(true);
    
    if (!tracker.initialize(config.template_model, config.search_model)) {
        throw std::runtime_error("Failed to initialize tracker for stress test");
    }
    
    cv::VideoCapture cap(config.input_video);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video for stress test");
    }
    
    cv::Mat frame;
    cap.read(frame);
    tracker.start_tracking(frame, config.initial_bbox);
    
    auto start_time = std::chrono::steady_clock::now();
    auto end_time = start_time + std::chrono::minutes(config.stress_duration_min);
    
    int total_frames = 0;
    std::vector<double> fps_samples;
    std::vector<double> temp_samples;
    
    while (std::chrono::steady_clock::now() < end_time) {
        // Loop video if needed
        if (!cap.read(frame) || frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        auto frame_start = std::chrono::steady_clock::now();
        tracker.update(frame);
        auto frame_end = std::chrono::steady_clock::now();
        
        double frame_time = std::chrono::duration<double>(frame_end - frame_start).count();
        fps_samples.push_back(1.0 / frame_time);
        
        // Sample temperature
        if (total_frames % 30 == 0) {
            std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
            if (temp_file.is_open()) {
                int temp_millidegree;
                temp_file >> temp_millidegree;
                temp_samples.push_back(temp_millidegree / 1000.0);
            }
        }
        
        total_frames++;
        
        // Progress update every minute
        if (total_frames % 1800 == 0) {  // Assuming ~30 FPS
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            int elapsed_min = std::chrono::duration_cast<std::chrono::minutes>(elapsed).count();
            double current_temp = temp_samples.empty() ? 0 : temp_samples.back();
            
            std::cout << "  Stress test progress: " << elapsed_min << "/" 
                     << config.stress_duration_min << " minutes, Temp: " 
                     << std::fixed << std::setprecision(1) << current_temp << "°C" << std::endl;
        }
    }
    
    cap.release();
    
    // Print stress test results
    std::cout << "Stress Test Results:" << std::endl;
    std::cout << "  Duration: " << config.stress_duration_min << " minutes" << std::endl;
    std::cout << "  Total frames: " << total_frames << std::endl;
    
    if (!fps_samples.empty()) {
        double avg_fps = std::accumulate(fps_samples.begin(), fps_samples.end(), 0.0) / fps_samples.size();
        double min_fps = *std::min_element(fps_samples.begin(), fps_samples.end());
        std::cout << "  Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
        std::cout << "  Minimum FPS: " << std::fixed << std::setprecision(2) << min_fps << std::endl;
    }
    
    if (!temp_samples.empty()) {
        double avg_temp = std::accumulate(temp_samples.begin(), temp_samples.end(), 0.0) / temp_samples.size();
        double max_temp = *std::max_element(temp_samples.begin(), temp_samples.end());
        std::cout << "  Average temperature: " << std::fixed << std::setprecision(1) << avg_temp << "°C" << std::endl;
        std::cout << "  Maximum temperature: " << std::fixed << std::setprecision(1) << max_temp << "°C" << std::endl;
        
        if (max_temp > 80.0) {
            std::cout << "  WARNING: High temperature detected! Consider improving cooling." << std::endl;
        }
    }
}

void save_results_json(const BenchmarkResults& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    
    file << "{\n";
    file << "  \"timestamp\": \"" << FEARUtils::get_current_timestamp() << "\",\n";
    file << "  \"version\": \"" << FEARTRACKER_VERSION_STRING << "\",\n";
    file << "  \"build_type\": \"" << FEARTRACKER_BUILD_TYPE << "\",\n";
    file << "  \"system_info\": \"" << results.system_info << "\",\n";
    file << "  \"performance\": {\n";
    file << "    \"total_frames\": " << results.total_frames << ",\n";
    file << "    \"total_time_sec\": " << results.total_time_sec << ",\n";
    file << "    \"initialization_time_ms\": " << results.initialization_time_ms << ",\n";
    file << "    \"avg_fps\": " << results.avg_fps << ",\n";
    file << "    \"min_fps\": " << results.min_fps << ",\n";
    file << "    \"max_fps\": " << results.max_fps << ",\n";
    file << "    \"avg_inference_time_ms\": " << results.avg_inference_time_ms << ",\n";
    file << "    \"avg_confidence\": " << results.avg_confidence << ",\n";
    file << "    \"min_confidence\": " << results.min_confidence << ",\n";
    file << "    \"max_confidence\": " << results.max_confidence << "\n";
    file << "  },\n";
    file << "  \"thermal\": {\n";
    file << "    \"avg_temperature_c\": " << results.avg_temperature_c << ",\n";
    file << "    \"max_temperature_c\": " << results.max_temperature_c << "\n";
    file << "  }\n";
    file << "}\n";
    
    file.close();
}

void print_results(const BenchmarkResults& results) {
    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Initialization: " << std::fixed << std::setprecision(2) 
              << results.initialization_time_ms << " ms" << std::endl;
    std::cout << "Total frames: " << results.total_frames << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) 
              << results.total_time_sec << " seconds" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) 
              << results.avg_fps << std::endl;
    std::cout << "FPS range: " << std::fixed << std::setprecision(2) 
              << results.min_fps << " - " << results.max_fps << std::endl;
    std::cout << "Average confidence: " << std::fixed << std::setprecision(3) 
              << results.avg_confidence << std::endl;
    std::cout << "Confidence range: " << std::fixed << std::setprecision(3) 
              << results.min_confidence << " - " << results.max_confidence << std::endl;
    
    if (results.avg_temperature_c > 0) {
        std::cout << "Average temperature: " << std::fixed << std::setprecision(1) 
                  << results.avg_temperature_c << "°C" << std::endl;
        std::cout << "Maximum temperature: " << std::fixed << std::setprecision(1) 
                  << results.max_temperature_c << "°C" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--video" && i + 1 < argc) {
            config.input_video = argv[++i];
        } else if (arg == "--bbox" && i + 1 < argc) {
            if (!parse_bbox(argv[++i], config.initial_bbox)) {
                std::cerr << "Invalid bbox format" << std::endl;
                return 1;
            }
        } else if (arg == "--template" && i + 1 < argc) {
            config.template_model = argv[++i];
        } else if (arg == "--search" && i + 1 < argc) {
            config.search_model = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            config.test_frames = std::stoi(argv[++i]);
        } else if (arg == "--stress-test") {
            config.stress_test = true;
        } else if (arg == "--stress-duration" && i + 1 < argc) {
            config.stress_duration_min = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "FEARTracker Pi Benchmark Tool\n\n";
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --video PATH           Input video (default: assets/test.mp4)\n";
            std::cout << "  --bbox X,Y,W,H         Initial bbox (default: 163,53,45,174)\n";
            std::cout << "  --template PATH        Template model path\n";
            std::cout << "  --search PATH          Search model path\n";
            std::cout << "  --output PATH          Output JSON file\n";
            std::cout << "  --frames N             Limit test to N frames (0=all)\n";
            std::cout << "  --stress-test          Run stress test\n";
            std::cout << "  --stress-duration N    Stress test duration in minutes\n";
            std::cout << "  --help                 Show this help\n";
            return 0;
        }
    }
    
    try {
        // Print system information
        std::cout << "FEARTracker Pi Benchmark v" << FEARTRACKER_VERSION_STRING << std::endl;
        print_system_specs();
        
        // Validate configuration
        if (!FEARUtils::file_exists(config.input_video)) {
            std::cerr << "Input video not found: " << config.input_video << std::endl;
            return 1;
        }
        
        if (!FEARUtils::file_exists(config.template_model)) {
            std::cerr << "Template model not found: " << config.template_model << std::endl;
            return 1;
        }
        
        if (!FEARUtils::file_exists(config.search_model)) {
            std::cerr << "Search model not found: " << config.search_model << std::endl;
            return 1;
        }
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Video: " << config.input_video << std::endl;
        std::cout << "  Initial bbox: " << FEARUtils::bbox_to_string(config.initial_bbox) << std::endl;
        std::cout << "  Test frames: " << (config.test_frames > 0 ? std::to_string(config.test_frames) : "all") << std::endl;
        std::cout << std::endl;
        
        if (config.stress_test) {
            run_stress_test(config);
        } else {
            // Run performance benchmark
            BenchmarkResults results = run_performance_benchmark(config);
            
            // Print results
            print_results(results);
            
            // Save results to file
            save_results_json(results, config.output_file);
            std::cout << "\nResults saved to: " << config.output_file << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}