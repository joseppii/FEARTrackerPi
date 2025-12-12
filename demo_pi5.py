#!/usr/bin/env python3

"""
FEARTrackerPi Demo for Raspberry Pi 5
Adapted from FEARTrackerAC demo_video_onnx.py for Pi5 compatibility
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
import psutil
from typing import List, Dict, Any, Tuple
from collections import deque

# Try to import original utilities if available, otherwise use simplified versions
try:
    from model_training.utils.utils import get_extended_crop, clamp_bbox, make_grid
    from model_training.dataset.box_coder import TrackerDecodeResult, FEARBoxCoder
    ORIGINAL_UTILS_AVAILABLE = True
except ImportError:
    print("Original FEARTracker utilities not available, using simplified versions")
    ORIGINAL_UTILS_AVAILABLE = False


class Pi5InferenceEngine:
    """ONNX Runtime inference engine optimized for Raspberry Pi 5"""
    
    def __init__(self):
        self.providers = self._get_optimal_providers()
        print(f"Available ONNX providers: {ort.get_available_providers()}")
        print(f"Using providers: {self.providers}")
    
    def _get_optimal_providers(self) -> List[str]:
        """Get optimal execution providers for Pi5"""
        providers = []
        available = ort.get_available_providers()
        
        # Try GPU providers first (VideoCore VII)
        if 'OpenCLExecutionProvider' in available:
            providers.append('OpenCLExecutionProvider')
            print("✓ VideoCore VII GPU acceleration enabled")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        return providers
    
    def create_session(self, model_path: str) -> ort.InferenceSession:
        """Create optimized ONNX session for Pi5"""
        session_options = ort.SessionOptions()
        
        # Pi5 optimization settings
        session_options.intra_op_num_threads = 4  # Use all 4 Cortex-A76 cores
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return ort.InferenceSession(model_path, session_options, providers=self.providers)


class SimplifiedFEARTracker:
    """Simplified FEAR tracker for Pi5 when original utilities are not available"""
    
    def __init__(self, template_session: ort.InferenceSession, search_session: ort.InferenceSession):
        self.template_session = template_session
        self.search_session = search_session
        
        # Simplified config
        self.config = {
            "template_size": 128,
            "instance_size": 256,
            "score_size": 16,
            "total_stride": 16,
            "template_bbox_offset": 2.0,
            "search_context": 2.0,
        }
        
        # Tracking state
        self.bbox = None
        self.template_features = None
        self.mean_color = None
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    
    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize
        image = cv2.resize(image, target_size)
        
        # BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize and reshape
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image.reshape(1, 3, target_size[1], target_size[0])
        
        # ImageNet normalization
        image = (image - self.mean) / self.std
        return image
    
    def _get_crop(self, image: np.ndarray, bbox: np.ndarray, crop_size: int, context: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Simple crop extraction"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        # Calculate crop size with context
        s = max(w, h) * context
        crop_w = crop_h = int(s)
        
        # Calculate crop bounds
        x1 = int(cx - crop_w/2)
        y1 = int(cy - crop_h/2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Handle boundaries
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - image.shape[1])
        pad_bottom = max(0, y2 - image.shape[0])
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        
        # Pad if necessary
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, 
                                    cv2.BORDER_CONSTANT, value=self.mean_color)
        
        mapping = np.array([x1 - pad_left, y1 - pad_top, crop_w, crop_h])
        return crop, mapping
    
    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:
        """Initialize tracker with first frame"""
        self.bbox = bbox.copy()
        self.mean_color = np.mean(image, axis=(0, 1))
        
        # Extract template crop
        template_crop, _ = self._get_crop(image, bbox, self.config["template_size"], 
                                         self.config["template_bbox_offset"])
        
        # Preprocess template
        template_input = self._preprocess_image(template_crop, (128, 128))
        
        # Extract template features
        template_outputs = self.template_session.run(None, {"template": template_input})
        self.template_features = template_outputs[0]
        
        print(f"✓ Tracker initialized with bbox: {bbox}")
    
    def update(self, image: np.ndarray) -> Dict[str, Any]:
        """Update tracker with new frame"""
        # Extract search crop
        search_crop, mapping = self._get_crop(image, self.bbox, self.config["instance_size"], 
                                            self.config["search_context"])
        
        # Preprocess search image
        search_input = self._preprocess_image(search_crop, (256, 256))
        
        # Run search model
        search_outputs = self.search_session.run(
            None,
            {
                "template_features": self.template_features,
                "search": search_input
            }
        )
        
        classification_map = search_outputs[0]  # [1, 1, 16, 16]
        regression_map = search_outputs[1]      # [1, 4, 16, 16]
        
        # Simple postprocessing
        cls_score = 1.0 / (1.0 + np.exp(-classification_map))  # sigmoid
        cls_score = np.squeeze(cls_score)
        
        # Find best location
        max_idx = np.argmax(cls_score)
        r_max, c_max = np.unravel_index(max_idx, cls_score.shape)
        
        # Get regression values
        reg_values = regression_map[0, :, r_max, c_max]
        
        # Calculate position in crop coordinates
        stride = self.config["total_stride"]
        grid_x = c_max * stride + stride // 2
        grid_y = r_max * stride + stride // 2
        
        # Decode bbox (simplified)
        pred_bbox = np.array([
            grid_x - reg_values[0],  # left
            grid_y - reg_values[1],  # top
            reg_values[0] + reg_values[2],  # width
            reg_values[1] + reg_values[3]   # height
        ])
        
        # Transform back to image coordinates
        scale_x = mapping[2] / self.config["instance_size"]
        scale_y = mapping[3] / self.config["instance_size"]
        
        pred_bbox = np.array([
            pred_bbox[0] * scale_x + mapping[0],
            pred_bbox[1] * scale_y + mapping[1],
            pred_bbox[2] * scale_x,
            pred_bbox[3] * scale_y
        ])
        
        # Clamp to image bounds
        pred_bbox[0] = max(0, pred_bbox[0])
        pred_bbox[1] = max(0, pred_bbox[1])
        pred_bbox[2] = min(image.shape[1] - pred_bbox[0], pred_bbox[2])
        pred_bbox[3] = min(image.shape[0] - pred_bbox[1], pred_bbox[3])
        
        self.bbox = pred_bbox
        confidence = cls_score[r_max, c_max]
        
        return {"bbox": pred_bbox, "confidence": confidence}


class Pi5PerformanceMonitor:
    """Performance monitoring for Pi5"""
    
    def __init__(self):
        self.frame_times = []
        self.temperatures = []
        self.memory_usage = []
        self.cpu_usage = []
        self.start_time = None
    
    def start_frame(self):
        self.start_time = time.time()
    
    def end_frame(self):
        if self.start_time is not None:
            frame_time = time.time() - self.start_time
            self.frame_times.append(frame_time)
            
            # Monitor system metrics
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # Try to get Pi5 temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read()) / 1000.0
                    self.temperatures.append(temp)
            except:
                pass
    
    def get_stats(self) -> Dict[str, float]:
        if not self.frame_times:
            return {}
        
        avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        stats = {
            "avg_fps": avg_fps,
            "min_fps": 1.0 / max(self.frame_times),
            "max_fps": 1.0 / min(self.frame_times),
            "avg_cpu": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "avg_memory": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
        }
        
        if self.temperatures:
            stats["avg_temp"] = sum(self.temperatures) / len(self.temperatures)
            stats["max_temp"] = max(self.temperatures)
        
        return stats


def draw_bbox(image: np.ndarray, bbox: np.ndarray, confidence: float = 0.0, 
              color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw bounding box with confidence on image"""
    image = image.copy()
    x, y, w, h = bbox.astype(int)
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Draw confidence text
    if confidence > 0:
        text = f"Conf: {confidence:.3f}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, thickness)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="FEARTrackerPi Demo for Raspberry Pi 5")
    parser.add_argument("--video", default="assets/test.mp4", help="Input video path")
    parser.add_argument("--output", default="outputs/test_tracked.mp4", help="Output video path")
    parser.add_argument("--bbox", type=str, default="163,53,45,174", 
                       help="Initial bounding box as x,y,w,h")
    parser.add_argument("--template-model", default="models/fear_net_template.onnx",
                       help="Template ONNX model path")
    parser.add_argument("--search-model", default="models/fear_net_search.onnx",
                       help="Search ONNX model path")
    parser.add_argument("--benchmark", action="store_true", help="Enable performance monitoring")
    parser.add_argument("--display", action="store_true", help="Display tracking in real-time")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames")
    
    args = parser.parse_args()
    
    print("FEARTrackerPi Demo for Raspberry Pi 5")
    print("=" * 40)
    
    # Parse initial bbox
    bbox = np.array([float(x) for x in args.bbox.split(",")])
    print(f"Initial bbox: {bbox}")
    
    # Check model files
    if not os.path.exists(args.template_model):
        print(f"❌ Template model not found: {args.template_model}")
        return
    if not os.path.exists(args.search_model):
        print(f"❌ Search model not found: {args.search_model}")
        return
    
    print(f"✓ Template model: {args.template_model}")
    print(f"✓ Search model: {args.search_model}")
    
    # Initialize inference engine
    engine = Pi5InferenceEngine()
    
    # Load ONNX models
    print("Loading ONNX models...")
    template_session = engine.create_session(args.template_model)
    search_session = engine.create_session(args.search_model)
    print("✓ Models loaded successfully")
    
    # Initialize tracker
    tracker = SimplifiedFEARTracker(template_session, search_session)
    
    # Initialize performance monitor
    monitor = Pi5PerformanceMonitor() if args.benchmark else None
    
    # Open video
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {args.video}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Setup output
    output_writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
    # Read first frame and initialize tracker
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read first frame")
        return
    
    tracker.initialize(frame, bbox)
    
    frame_count = 0
    print("Starting tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if monitor:
            monitor.start_frame()
        
        # Track
        result = tracker.update(frame)
        tracked_bbox = result["bbox"]
        confidence = result.get("confidence", 0.0)
        
        # Draw visualization
        vis_frame = draw_bbox(frame, tracked_bbox, confidence)
        
        if monitor:
            monitor.end_frame()
        
        # Display frame
        if args.display:
            cv2.imshow('FEARTrackerPi', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save frame
        if output_writer:
            output_writer.write(vis_frame)
        
        if args.save_frames:
            frame_dir = os.path.join(os.path.dirname(args.output), "frames")
            os.makedirs(frame_dir, exist_ok=True)
            cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_count:06d}.jpg"), vis_frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100
            if monitor:
                stats = monitor.get_stats()
                fps_current = stats.get("avg_fps", 0)
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"FPS: {fps_current:.1f}, "
                      f"CPU: {stats.get('avg_cpu', 0):.1f}%, "
                      f"Temp: {stats.get('avg_temp', 0):.1f}°C")
            else:
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    if output_writer:
        output_writer.release()
    if args.display:
        cv2.destroyAllWindows()
    
    print(f"✓ Tracking complete! Processed {frame_count} frames")
    
    # Print performance stats
    if monitor:
        print("\nPerformance Statistics:")
        print("-" * 30)
        stats = monitor.get_stats()
        for key, value in stats.items():
            if key.endswith("fps"):
                print(f"{key}: {value:.2f} FPS")
            elif key.endswith("temp"):
                print(f"{key}: {value:.1f}°C")
            elif key.endswith(("cpu", "memory")):
                print(f"{key}: {value:.1f}%")
            else:
                print(f"{key}: {value:.3f}")
    
    if args.output:
        print(f"✓ Output saved: {args.output}")


if __name__ == "__main__":
    main()