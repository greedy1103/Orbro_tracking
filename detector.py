"""
Person Detection Module using YOLOv8
Phát hiện người trong video sử dụng mô hình YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import os

class PersonDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Khởi tạo detector với mô hình YOLOv8
        
        Args:
            model_path: Đường dẫn đến mô hình YOLOv8
            confidence_threshold: Ngưỡng tin cậy cho detection
        """
        # Tự động tải model nếu không tồn tại
        if not os.path.exists(model_path):
            print(f"⚠️ Model '{model_path}' không tồn tại. Đang tự động tải xuống...")
            try:
                YOLO(model_path) # Thư viện ultralytics sẽ tự tải model
                print(f"✅ Đã tải thành công model '{model_path}'.")
            except Exception as e:
                print(f"❌ Lỗi nghiêm trọng khi tải model: {e}")
                print("💡 Vui lòng kiểm tra kết nối mạng và thử lại.")
                # Có thể raise exception ở đây để dừng chương trình nếu model là bắt buộc
                raise e

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0  # Class ID cho người trong COCO dataset
        
    def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Phát hiện người trong khung hình
        
        Args:
            frame: Khung hình đầu vào (BGR format)
            
        Returns:
            List các detection với thông tin bbox, confidence
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Chỉ lấy detections của người (class_id = 0)
                    if int(boxes.cls[i]) == self.person_class_id:
                        confidence = float(boxes.conf[i])
                        
                        if confidence >= self.confidence_threshold:
                            # Lấy tọa độ bbox
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': self.person_class_id,
                                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                            }
                            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Vẽ các detection lên khung hình
        
        Args:
            frame: Khung hình gốc
            detections: List các detection
            
        Returns:
            Khung hình đã vẽ detection
        """
        frame_copy = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Vẽ bounding box
            cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Vẽ confidence score
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_copy, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_copy