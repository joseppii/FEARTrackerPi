#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <fstream>

struct OperationStats {
    std::string name;
    uint64_t call_count;
    double total_time_ms;
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    
    OperationStats() : call_count(0), total_time_ms(0), min_time_ms(0), max_time_ms(0), avg_time_ms(0) {}
};

struct SystemMetrics {
    double cpu_usage;
    double memory_usage_mb;
    double temperature_c;
    std::chrono::steady_clock::time_point timestamp;
    
    SystemMetrics() : cpu_usage(0), memory_usage_mb(0), temperature_c(0) {
        timestamp = std::chrono::steady_clock::now();
    }
};

class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // Operation timing
    void start_operation(const std::string& operation_name);
    void end_operation(const std::string& operation_name);
    
    // FPS monitoring
    void record_fps();
    double get_current_fps() const;
    double get_average_fps() const;
    
    // System metrics
    void record_system_metrics();
    SystemMetrics get_current_system_metrics() const;
    
    // Configuration
    void set_enabled(bool enabled) { enabled_ = enabled; }
    void set_system_monitoring_enabled(bool enabled) { system_monitoring_ = enabled; }
    void set_fps_window_size(int size);
    void set_metrics_save_interval(int interval_frames) { save_interval_ = interval_frames; }
    
    // Statistics and reporting
    OperationStats get_operation_stats(const std::string& operation_name) const;
    std::vector<OperationStats> get_all_operation_stats() const;
    std::string get_summary() const;
    void print_summary() const;
    
    // Data export
    void save_to_csv(const std::string& filename) const;
    void save_to_json(const std::string& filename) const;
    
    // Control
    void reset();
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    bool is_enabled() const { return enabled_; }

private:
    // Configuration
    bool enabled_;
    bool system_monitoring_;
    bool paused_;
    int fps_window_size_;
    int save_interval_;
    int frame_counter_;
    
    // Timing data
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> operation_start_times_;
    std::unordered_map<std::string, OperationStats> operation_stats_;
    
    // FPS tracking
    std::vector<std::chrono::steady_clock::time_point> frame_timestamps_;
    std::chrono::steady_clock::time_point last_fps_calculation_;
    mutable double cached_fps_;
    mutable bool fps_cache_valid_;
    
    // System metrics
    std::vector<SystemMetrics> system_metrics_history_;
    std::chrono::steady_clock::time_point session_start_time_;
    
    // Thread safety
    mutable std::mutex stats_mutex_;
    mutable std::mutex fps_mutex_;
    mutable std::mutex metrics_mutex_;
    
    // Helper methods
    void update_operation_stats(const std::string& name, double time_ms);
    double read_cpu_usage() const;
    double read_memory_usage() const;
    double read_temperature() const;
    void cleanup_old_fps_data();
    void cleanup_old_metrics();
    std::string format_duration(double ms) const;
    std::string get_timestamp_string() const;
    
    // Pi5 specific system reading
    bool read_pi5_temperature(double& temp) const;
    bool read_vcgencmd_temperature(double& temp) const;
    
    // Constants
    static constexpr int DEFAULT_FPS_WINDOW_SIZE = 30;
    static constexpr int DEFAULT_SAVE_INTERVAL = 100;
    static constexpr int MAX_METRICS_HISTORY = 1000;
};