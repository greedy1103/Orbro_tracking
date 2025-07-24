#!/usr/bin/env python3
"""
Main Script - Person Detection, Tracking and Attribute Recognition System
"""

import argparse
import sys
import tkinter as tk

# Import with fallback
try:
    from detector import PersonDetector
    from tracker import PersonTracker
    DETECTOR_AVAILABLE = True
except ImportError as e:
    DETECTOR_AVAILABLE = False
    print(f"⚠️ Detector/Tracker không khả dụng (sẽ dùng mock): {e}")

try:
    from tag_tracker_with_video import VideoTagTracker
    VIDEO_TRACKER_AVAILABLE = True
except ImportError as e:
    VIDEO_TRACKER_AVAILABLE = False
    print(f"⚠️ Video Tag Tracker không khả dụng: {e}")

def launch_video_tracker_standalone(model_path=None):
    """Khởi chạy Video Tag Tracker độc lập"""
    if not VIDEO_TRACKER_AVAILABLE:
        print("❌ Video Tag Tracker không khả dụng!")
        print("Vui lòng cài đặt: apt install python3-tk python3-opencv")
        return
    
    print("🎬 Đang khởi chạy Video Tag Tracker...")
    root = tk.Tk()
    app = VideoTagTracker(root, model_path=model_path)
    root.mainloop()

def main():
    """
    Hàm main cho hệ thống
    """
    parser = argparse.ArgumentParser(description='Person Detection, Tracking and Attribute Recognition System')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt',
                       help='Đường dẫn mô hình YOLOv8 (default: yolov8n.pt) cho VideoTagTracker.')
    
    args = parser.parse_args()
    
    # Mặc định khởi chạy Video Tag Tracker
    launch_video_tracker_standalone(model_path=args.model)

if __name__ == "__main__":
    main()