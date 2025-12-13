#include "performance_monitor.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/sysinfo.h>

PerformanceMonitor::PerformanceMonitor()
    : enabled_(true)
    , system_monitoring_(false)
    , paused_(false)
    , fps_window_size_(DEFAULT_FPS_WINDOW_SIZE)
    , save_interval_(DEFAULT_SAVE_INTERVAL)
    , frame_counter_(0)
    , cached_fps_(0.0)
    , fps_cache_valid_(false)
{
    session_start_time_ = std::chrono::steady_clock::now();
    frame_timestamps_.reserve(fps_window_size_);
    system_metrics_history_.reserve(MAX_METRICS_HISTORY);
}

PerformanceMonitor::~PerformanceMonitor() = default;

void PerformanceMonitor::start_operation(const std::string& operation_name) {
    if (!enabled_ || paused_) return;
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    operation_start_times_[operation_name] = std::chrono::steady_clock::now();
}

void PerformanceMonitor::end_operation(const std::string& operation_name) {
    if (!enabled_ || paused_) return;
    
    auto end_time = std::chrono::steady_clock::now();
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto start_it = operation_start_times_.find(operation_name);
    if (start_it != operation_start_times_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_it->second);
        double time_ms = duration.count() / 1000.0;
        
        update_operation_stats(operation_name, time_ms);
        operation_start_times_.erase(start_it);
    }
}

void PerformanceMonitor::record_fps() {
    if (!enabled_ || paused_) return;
    
    std::lock_guard<std::mutex> lock(fps_mutex_);
    auto now = std::chrono::steady_clock::now();
    frame_timestamps_.push_back(now);
    
    // Maintain window size
    if (static_cast<int>(frame_timestamps_.size()) > fps_window_size_) {
        frame_timestamps_.erase(frame_timestamps_.begin());
    }
    
    fps_cache_valid_ = false;
    frame_counter_++;
    
    // Record system metrics periodically
    if (system_monitoring_ && (frame_counter_ % 30 == 0)) {  // Every 30 frames
        record_system_metrics();
    }
    
    // Auto-save periodically
    if (save_interval_ > 0 && (frame_counter_ % save_interval_ == 0)) {
        cleanup_old_metrics();
    }
}

double PerformanceMonitor::get_current_fps() const {
    std::lock_guard<std::mutex> lock(fps_mutex_);
    
    if (frame_timestamps_.size() < 2) {
        return 0.0;
    }
    
    if (fps_cache_valid_) {
        return cached_fps_;
    }
    
    // Calculate FPS over recent window
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        frame_timestamps_.back() - frame_timestamps_.front());
    
    if (duration.count() > 0) {
        cached_fps_ = (frame_timestamps_.size() - 1) * 1000.0 / duration.count();
    } else {
        cached_fps_ = 0.0;
    }
    
    fps_cache_valid_ = true;
    return cached_fps_;
}

double PerformanceMonitor::get_average_fps() const {
    std::lock_guard<std::mutex> lock(fps_mutex_);
    
    if (frame_timestamps_.empty()) {
        return 0.0;
    }
    
    auto session_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - session_start_time_);
    
    if (session_duration.count() > 0) {
        return frame_counter_ * 1000.0 / session_duration.count();
    }
    
    return 0.0;
}

void PerformanceMonitor::record_system_metrics() {
    if (!enabled_ || !system_monitoring_) return;
    
    SystemMetrics metrics;
    metrics.cpu_usage = read_cpu_usage();
    metrics.memory_usage_mb = read_memory_usage();
    metrics.temperature_c = read_temperature();
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    system_metrics_history_.push_back(metrics);
    
    // Maintain history size
    if (system_metrics_history_.size() > MAX_METRICS_HISTORY) {
        system_metrics_history_.erase(system_metrics_history_.begin());
    }
}

SystemMetrics PerformanceMonitor::get_current_system_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (!system_metrics_history_.empty()) {
        return system_metrics_history_.back();
    }
    return SystemMetrics();
}

void PerformanceMonitor::set_fps_window_size(int size) {
    std::lock_guard<std::mutex> lock(fps_mutex_);
    fps_window_size_ = std::max(1, size);
    
    // Adjust current data
    if (static_cast<int>(frame_timestamps_.size()) > fps_window_size_) {
        frame_timestamps_.erase(frame_timestamps_.begin(), 
                               frame_timestamps_.end() - fps_window_size_);
    }
    fps_cache_valid_ = false;
}

OperationStats PerformanceMonitor::get_operation_stats(const std::string& operation_name) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto it = operation_stats_.find(operation_name);
    return (it != operation_stats_.end()) ? it->second : OperationStats();
}

std::vector<OperationStats> PerformanceMonitor::get_all_operation_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    std::vector<OperationStats> stats;
    stats.reserve(operation_stats_.size());
    
    for (const auto& pair : operation_stats_) {
        stats.push_back(pair.second);
    }
    
    // Sort by total time (descending)
    std::sort(stats.begin(), stats.end(), 
              [](const OperationStats& a, const OperationStats& b) {
                  return a.total_time_ms > b.total_time_ms;
              });
    
    return stats;
}

std::string PerformanceMonitor::get_summary() const {
    std::ostringstream oss;
    
    oss << "Performance Monitor Summary\n";
    oss << "==========================\n";
    
    // Session info
    auto session_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - session_start_time_);
    
    oss << "Session Duration: " << format_duration(session_duration.count()) << "\n";
    oss << "Total Frames: " << frame_counter_ << "\n";
    oss << "Average FPS: " << std::fixed << std::setprecision(2) << get_average_fps() << "\n";
    oss << "Current FPS: " << std::fixed << std::setprecision(2) << get_current_fps() << "\n\n";
    
    // Operation stats
    auto stats = get_all_operation_stats();
    if (!stats.empty()) {
        oss << "Operation Statistics:\n";
        oss << std::left << std::setw(20) << "Operation" 
            << std::setw(10) << "Calls" 
            << std::setw(12) << "Total(ms)" 
            << std::setw(12) << "Avg(ms)" 
            << std::setw(12) << "Min(ms)" 
            << std::setw(12) << "Max(ms)" << "\n";
        oss << std::string(78, '-') << "\n";
        
        for (const auto& stat : stats) {
            oss << std::left << std::setw(20) << stat.name
                << std::setw(10) << stat.call_count
                << std::setw(12) << std::fixed << std::setprecision(2) << stat.total_time_ms
                << std::setw(12) << std::fixed << std::setprecision(3) << stat.avg_time_ms
                << std::setw(12) << std::fixed << std::setprecision(3) << stat.min_time_ms
                << std::setw(12) << std::fixed << std::setprecision(3) << stat.max_time_ms << "\n";
        }
        oss << "\n";
    }
    
    // System metrics
    if (system_monitoring_) {
        SystemMetrics current = get_current_system_metrics();
        oss << "System Metrics:\n";
        oss << "CPU Usage: " << std::fixed << std::setprecision(1) << current.cpu_usage << "%\n";
        oss << "Memory Usage: " << std::fixed << std::setprecision(1) << current.memory_usage_mb << " MB\n";
        if (current.temperature_c > 0) {
            oss << "Temperature: " << std::fixed << std::setprecision(1) << current.temperature_c << "°C\n";
        }
    }
    
    return oss.str();
}

void PerformanceMonitor::print_summary() const {
    std::cout << get_summary() << std::endl;
}

void PerformanceMonitor::save_to_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write operation stats
    file << "Operation,Calls,Total_ms,Avg_ms,Min_ms,Max_ms\n";
    auto stats = get_all_operation_stats();
    for (const auto& stat : stats) {
        file << stat.name << "," << stat.call_count << "," 
             << stat.total_time_ms << "," << stat.avg_time_ms << ","
             << stat.min_time_ms << "," << stat.max_time_ms << "\n";
    }
    
    // Write system metrics if available
    if (system_monitoring_ && !system_metrics_history_.empty()) {
        file << "\nTimestamp,CPU_Usage,Memory_MB,Temperature_C\n";
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        for (const auto& metric : system_metrics_history_) {
            auto time_point = std::chrono::duration_cast<std::chrono::milliseconds>(
                metric.timestamp - session_start_time_);
            file << time_point.count() << "," << metric.cpu_usage << ","
                 << metric.memory_usage_mb << "," << metric.temperature_c << "\n";
        }
    }
    
    file.close();
}

void PerformanceMonitor::save_to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"session_info\": {\n";
    file << "    \"start_time\": \"" << get_timestamp_string() << "\",\n";
    file << "    \"frame_count\": " << frame_counter_ << ",\n";
    file << "    \"average_fps\": " << get_average_fps() << ",\n";
    file << "    \"current_fps\": " << get_current_fps() << "\n";
    file << "  },\n";
    
    // Operation stats
    file << "  \"operations\": [\n";
    auto stats = get_all_operation_stats();
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto& stat = stats[i];
        file << "    {\n";
        file << "      \"name\": \"" << stat.name << "\",\n";
        file << "      \"calls\": " << stat.call_count << ",\n";
        file << "      \"total_ms\": " << stat.total_time_ms << ",\n";
        file << "      \"avg_ms\": " << stat.avg_time_ms << ",\n";
        file << "      \"min_ms\": " << stat.min_time_ms << ",\n";
        file << "      \"max_ms\": " << stat.max_time_ms << "\n";
        file << "    }" << (i < stats.size() - 1 ? "," : "") << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    
    file.close();
}

void PerformanceMonitor::reset() {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    std::lock_guard<std::mutex> fps_lock(fps_mutex_);
    std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
    
    operation_stats_.clear();
    operation_start_times_.clear();
    frame_timestamps_.clear();
    system_metrics_history_.clear();
    
    frame_counter_ = 0;
    fps_cache_valid_ = false;
    session_start_time_ = std::chrono::steady_clock::now();
}

void PerformanceMonitor::update_operation_stats(const std::string& name, double time_ms) {
    auto& stats = operation_stats_[name];
    stats.name = name;
    stats.call_count++;
    stats.total_time_ms += time_ms;
    stats.avg_time_ms = stats.total_time_ms / stats.call_count;
    
    if (stats.call_count == 1 || time_ms < stats.min_time_ms) {
        stats.min_time_ms = time_ms;
    }
    if (stats.call_count == 1 || time_ms > stats.max_time_ms) {
        stats.max_time_ms = time_ms;
    }
}

double PerformanceMonitor::read_cpu_usage() const {
    // Simple CPU usage reading for Linux
    static long long prev_idle = 0, prev_total = 0;
    
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return 0.0;
    }
    
    std::string line;
    std::getline(file, line);
    
    // Parse CPU line: cpu user nice system idle iowait irq softirq
    std::istringstream iss(line);
    std::string cpu_label;
    long long user, nice, system, idle, iowait, irq, softirq;
    
    iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    
    long long total = user + nice + system + idle + iowait + irq + softirq;
    long long diff_idle = idle - prev_idle;
    long long diff_total = total - prev_total;
    
    double cpu_usage = 0.0;
    if (diff_total != 0) {
        cpu_usage = 100.0 * (diff_total - diff_idle) / diff_total;
    }
    
    prev_idle = idle;
    prev_total = total;
    
    return cpu_usage;
}

double PerformanceMonitor::read_memory_usage() const {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        return 0.0;
    }
    
    double total_mb = info.totalram * info.mem_unit / (1024.0 * 1024.0);
    double free_mb = info.freeram * info.mem_unit / (1024.0 * 1024.0);
    
    return total_mb - free_mb;
}

double PerformanceMonitor::read_temperature() const {
    double temp = 0.0;
    
    // Try Pi5 specific methods
    if (read_pi5_temperature(temp) || read_vcgencmd_temperature(temp)) {
        return temp;
    }
    
    return 0.0;
}

bool PerformanceMonitor::read_pi5_temperature(double& temp) const {
    std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
    if (!file.is_open()) {
        return false;
    }
    
    int temp_millidegree;
    if (file >> temp_millidegree) {
        temp = temp_millidegree / 1000.0;
        return true;
    }
    
    return false;
}

bool PerformanceMonitor::read_vcgencmd_temperature(double& temp) const {
    FILE* pipe = popen("vcgencmd measure_temp", "r");
    if (!pipe) {
        return false;
    }
    
    char buffer[128];
    if (fgets(buffer, sizeof(buffer), pipe)) {
        // Parse: temp=43.2'C
        char* equals = strchr(buffer, '=');
        char* degree = strchr(buffer, '\'');
        if (equals && degree) {
            *degree = '\0';
            temp = atof(equals + 1);
            pclose(pipe);
            return true;
        }
    }
    
    pclose(pipe);
    return false;
}

void PerformanceMonitor::cleanup_old_fps_data() {
    // This method is called periodically to clean up old data
    // Currently, the sliding window handles this automatically
}

void PerformanceMonitor::cleanup_old_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (system_metrics_history_.size() > MAX_METRICS_HISTORY) {
        system_metrics_history_.erase(system_metrics_history_.begin(),
                                     system_metrics_history_.end() - MAX_METRICS_HISTORY);
    }
}

std::string PerformanceMonitor::format_duration(double ms) const {
    if (ms < 1000) {
        return std::to_string(static_cast<int>(ms)) + "ms";
    } else if (ms < 60000) {
        return std::to_string(static_cast<int>(ms / 1000)) + "s";
    } else {
        int minutes = static_cast<int>(ms / 60000);
        int seconds = static_cast<int>((ms - minutes * 60000) / 1000);
        return std::to_string(minutes) + "m " + std::to_string(seconds) + "s";
    }
}

std::string PerformanceMonitor::get_timestamp_string() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}