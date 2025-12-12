#!/usr/bin/env python3

"""
FEARTrackerPi Benchmark Script for Raspberry Pi 5
Performance testing and profiling for all phases
"""

import os
import sys
import time
import cv2
import numpy as np
import psutil
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    phase: str
    avg_fps: float
    min_fps: float
    max_fps: float
    frame_count: int
    total_time: float
    avg_cpu_usage: float
    max_cpu_usage: float
    avg_memory_usage: float
    max_memory_usage: float
    avg_temperature: float
    max_temperature: float
    model_load_time: float
    initialization_time: float
    tracking_accuracy: float
    device_info: Dict[str, Any]
    timestamp: str


class SystemMonitor:
    """System performance monitoring for Pi5"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.temperatures = []
        self.frame_times = []
        self.start_time = None
    
    def start_frame(self):
        self.start_time = time.time()
    
    def end_frame(self):
        if self.start_time:
            frame_time = time.time() - self.start_time
            self.frame_times.append(frame_time)
            
            # Collect system metrics
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # Pi5 temperature monitoring
            self._read_temperature()
    
    def _read_temperature(self):
        """Read CPU temperature from Pi5"""
        try:
            # Primary temperature sensor
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                self.temperatures.append(temp)
        except:
            # Alternative method
            try:
                import subprocess
                result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    temp_str = result.stdout.strip()
                    temp = float(temp_str.split('=')[1].split("'")[0])
                    self.temperatures.append(temp)
            except:
                pass
    
    def get_stats(self) -> Dict[str, float]:
        """Calculate performance statistics"""
        if not self.frame_times:
            return {}
        
        stats = {
            "avg_fps": 1.0 / (sum(self.frame_times) / len(self.frame_times)),
            "min_fps": 1.0 / max(self.frame_times),
            "max_fps": 1.0 / min(self.frame_times),
            "frame_count": len(self.frame_times),
            "total_time": sum(self.frame_times),
        }
        
        if self.cpu_usage:
            stats.update({
                "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage),
                "max_cpu_usage": max(self.cpu_usage),
            })
        
        if self.memory_usage:
            stats.update({
                "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage),
                "max_memory_usage": max(self.memory_usage),
            })
        
        if self.temperatures:
            stats.update({
                "avg_temperature": sum(self.temperatures) / len(self.temperatures),
                "max_temperature": max(self.temperatures),
            })
        
        return stats


class DeviceInfo:
    """Raspberry Pi 5 device information collector"""
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Collect comprehensive Pi5 device information"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "hostname": os.uname().nodename,
        }
        
        # Hardware model
        try:
            with open('/proc/device-tree/model', 'r') as f:
                info["model"] = f.read().strip().replace('\x00', '')
        except:
            info["model"] = "Unknown"
        
        # CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if line.startswith('Hardware'):
                        info["hardware"] = line.split(':')[1].strip()
                    elif line.startswith('Revision'):
                        info["revision"] = line.split(':')[1].strip()
                    elif line.startswith('processor') and 'cpu_count' not in info:
                        info["cpu_count"] = cpuinfo.count('processor')
        except:
            pass
        
        # Memory info
        try:
            mem = psutil.virtual_memory()
            info["total_memory_gb"] = round(mem.total / (1024**3), 2)
            info["available_memory_gb"] = round(mem.available / (1024**3), 2)
        except:
            pass
        
        # GPU memory
        try:
            with open('/boot/config.txt', 'r') as f:
                for line in f:
                    if line.startswith('gpu_mem'):
                        info["gpu_memory_mb"] = line.split('=')[1].strip()
        except:
            pass
        
        # Architecture
        info["architecture"] = os.uname().machine
        
        # Kernel version
        info["kernel"] = os.uname().release
        
        # CPU frequency
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'r') as f:
                freq_khz = int(f.read().strip())
                info["cpu_frequency_mhz"] = freq_khz // 1000
        except:
            pass
        
        # Current temperature
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                info["current_temperature_c"] = round(temp, 1)
        except:
            pass
        
        return info


class FEARBenchmarker:
    """Main benchmarking class for FEARTracker phases"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = results_dir
        self.monitor = SystemMonitor()
        os.makedirs(results_dir, exist_ok=True)
    
    def benchmark_phase1_python(self, video_path: str, bbox: List[float]) -> BenchmarkResult:
        """Benchmark Phase 1: Python ONNX implementation"""
        print("Benchmarking Phase 1: Python ONNX")
        
        # Import demo module
        from demo_pi5 import Pi5InferenceEngine, SimplifiedFEARTracker
        
        self.monitor.reset()
        
        # Model loading time
        load_start = time.time()
        engine = Pi5InferenceEngine()
        template_session = engine.create_session("models/fear_net_template.onnx")
        search_session = engine.create_session("models/fear_net_search.onnx")
        load_time = time.time() - load_start
        
        # Tracker initialization time
        init_start = time.time()
        tracker = SimplifiedFEARTracker(template_session, search_session)
        
        # Load first frame and initialize
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read video")
        
        tracker.initialize(frame, np.array(bbox))
        init_time = time.time() - init_start
        
        # Tracking benchmark
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.monitor.start_frame()
            result = tracker.update(frame)
            self.monitor.end_frame()
            
            frame_count += 1
            if frame_count % 50 == 0:
                progress = frame_count / total_frames * 100
                print(f"  Progress: {progress:.1f}%")
        
        cap.release()
        
        # Calculate results
        stats = self.monitor.get_stats()
        
        return BenchmarkResult(
            phase="Phase1_Python_ONNX",
            avg_fps=stats.get("avg_fps", 0),
            min_fps=stats.get("min_fps", 0),
            max_fps=stats.get("max_fps", 0),
            frame_count=frame_count,
            total_time=stats.get("total_time", 0),
            avg_cpu_usage=stats.get("avg_cpu_usage", 0),
            max_cpu_usage=stats.get("max_cpu_usage", 0),
            avg_memory_usage=stats.get("avg_memory_usage", 0),
            max_memory_usage=stats.get("max_memory_usage", 0),
            avg_temperature=stats.get("avg_temperature", 0),
            max_temperature=stats.get("max_temperature", 0),
            model_load_time=load_time,
            initialization_time=init_time,
            tracking_accuracy=0.0,  # TODO: Implement accuracy calculation
            device_info=DeviceInfo.get_device_info(),
            timestamp=datetime.now().isoformat()
        )
    
    def benchmark_stress_test(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Run stress test to evaluate thermal behavior and sustained performance"""
        print(f"Running stress test for {duration_minutes} minutes...")
        
        # Use a simple test video loop
        test_video = "assets/test.mp4"
        bbox = [163, 53, 45, 174]
        
        if not os.path.exists(test_video):
            print("Test video not found, creating synthetic video...")
            self._create_test_video(test_video)
        
        from demo_pi5 import Pi5InferenceEngine, SimplifiedFEARTracker
        
        # Initialize tracker
        engine = Pi5InferenceEngine()
        template_session = engine.create_session("models/fear_net_template.onnx")
        search_session = engine.create_session("models/fear_net_search.onnx")
        tracker = SimplifiedFEARTracker(template_session, search_session)
        
        # Initialize with first frame
        cap = cv2.VideoCapture(test_video)
        ret, frame = cap.read()
        tracker.initialize(frame, np.array(bbox))
        cap.release()
        
        # Stress test loop
        end_time = time.time() + duration_minutes * 60
        frame_count = 0
        fps_samples = []
        temp_samples = []
        
        while time.time() < end_time:
            # Re-open video for looping
            cap = cv2.VideoCapture(test_video)
            
            while True:
                ret, frame = cap.read()
                if not ret or time.time() >= end_time:
                    break
                
                start = time.time()
                tracker.update(frame)
                frame_time = time.time() - start
                
                fps_samples.append(1.0 / frame_time if frame_time > 0 else 0)
                
                # Sample temperature every 10 frames
                if frame_count % 10 == 0:
                    try:
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                            temp = float(f.read().strip()) / 1000.0
                            temp_samples.append(temp)
                    except:
                        pass
                
                frame_count += 1
            
            cap.release()
            
            # Progress update
            elapsed = time.time() - (end_time - duration_minutes * 60)
            progress = elapsed / (duration_minutes * 60) * 100
            if frame_count % 100 == 0:
                current_fps = np.mean(fps_samples[-50:]) if fps_samples else 0
                current_temp = temp_samples[-1] if temp_samples else 0
                print(f"  Stress test progress: {progress:.1f}%, "
                      f"FPS: {current_fps:.1f}, Temp: {current_temp:.1f}°C")
        
        return {
            "duration_minutes": duration_minutes,
            "total_frames": frame_count,
            "avg_fps": np.mean(fps_samples) if fps_samples else 0,
            "min_fps": np.min(fps_samples) if fps_samples else 0,
            "max_fps": np.max(fps_samples) if fps_samples else 0,
            "avg_temperature": np.mean(temp_samples) if temp_samples else 0,
            "max_temperature": np.max(temp_samples) if temp_samples else 0,
            "temperature_samples": temp_samples,
            "fps_samples": fps_samples[-100:],  # Keep last 100 samples
        }
    
    def _create_test_video(self, output_path: str, duration_sec: int = 30):
        """Create a synthetic test video for benchmarking"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))
        
        frames = duration_sec * 30
        for i in range(frames):
            # Create moving object
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = int(320 + 200 * np.sin(i * 0.1))
            y = int(240 + 100 * np.cos(i * 0.1))
            cv2.rectangle(frame, (x-25, y-25), (x+25, y+25), (0, 255, 0), -1)
            writer.write(frame)
        
        writer.release()
        print(f"Created test video: {output_path}")
    
    def save_results(self, results: List[BenchmarkResult], filename: str = None):
        """Save benchmark results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert results to serializable format
        results_data = [asdict(result) for result in results]
        
        with open(filepath, 'w') as f:
            json.dump({
                "benchmark_info": {
                    "device_info": DeviceInfo.get_device_info(),
                    "results_count": len(results),
                },
                "results": results_data
            }, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print benchmark results summary"""
        print("\n" + "="*60)
        print("FEARTRACKER PI5 BENCHMARK RESULTS")
        print("="*60)
        
        for result in results:
            print(f"\n{result.phase}:")
            print(f"  Average FPS: {result.avg_fps:.2f}")
            print(f"  FPS Range: {result.min_fps:.2f} - {result.max_fps:.2f}")
            print(f"  Frames Processed: {result.frame_count}")
            print(f"  Total Time: {result.total_time:.1f}s")
            print(f"  Model Load Time: {result.model_load_time:.2f}s")
            print(f"  CPU Usage: {result.avg_cpu_usage:.1f}% (max: {result.max_cpu_usage:.1f}%)")
            print(f"  Memory Usage: {result.avg_memory_usage:.1f}% (max: {result.max_memory_usage:.1f}%)")
            if result.avg_temperature > 0:
                print(f"  Temperature: {result.avg_temperature:.1f}°C (max: {result.max_temperature:.1f}°C)")


def main():
    parser = argparse.ArgumentParser(description="FEARTrackerPi Benchmark Suite")
    parser.add_argument("--video", default="assets/test.mp4", help="Test video path")
    parser.add_argument("--bbox", type=str, default="163,53,45,174", 
                       help="Initial bounding box as x,y,w,h")
    parser.add_argument("--phase", choices=["1", "all"], default="1", 
                       help="Which phase to benchmark")
    parser.add_argument("--stress-test", type=int, default=0, metavar="MINUTES",
                       help="Run stress test for specified minutes")
    parser.add_argument("--output-dir", default="benchmark_results", 
                       help="Output directory for results")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("FEARTrackerPi Benchmark Suite")
    print("Raspberry Pi 5 Performance Testing")
    print("=" * 50)
    
    # Device info
    device_info = DeviceInfo.get_device_info()
    print(f"Device: {device_info.get('model', 'Unknown')}")
    print(f"Architecture: {device_info.get('architecture', 'Unknown')}")
    print(f"Memory: {device_info.get('total_memory_gb', 'Unknown')} GB")
    print(f"Temperature: {device_info.get('current_temperature_c', 'Unknown')}°C")
    
    # Parse bbox
    bbox = [float(x) for x in args.bbox.split(",")]
    
    # Initialize benchmarker
    benchmarker = FEARBenchmarker(args.output_dir)
    results = []
    
    # Check if models exist
    if not os.path.exists("models/fear_net_template.onnx"):
        print("❌ ONNX models not found. Please run setup first.")
        return
    
    # Run benchmarks
    if args.phase == "1" or args.phase == "all":
        try:
            result = benchmarker.benchmark_phase1_python(args.video, bbox)
            results.append(result)
            print(f"✓ Phase 1 completed: {result.avg_fps:.2f} FPS average")
        except Exception as e:
            print(f"❌ Phase 1 benchmark failed: {e}")
    
    # Stress test
    if args.stress_test > 0:
        try:
            stress_results = benchmarker.benchmark_stress_test(args.stress_test)
            print(f"✓ Stress test completed: {stress_results['avg_fps']:.2f} FPS sustained")
            print(f"  Temperature range: {stress_results['avg_temperature']:.1f}°C - "
                  f"{stress_results['max_temperature']:.1f}°C")
        except Exception as e:
            print(f"❌ Stress test failed: {e}")
    
    # Print results
    if results:
        benchmarker.print_results(results)
    
    # Save results
    if args.save_results and results:
        benchmarker.save_results(results)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()