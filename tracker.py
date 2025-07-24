"""
Person Tracking Module
Theo dõi người và gán ID duy nhất cho mỗi người qua các khung hình
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import euclidean

class PersonTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        """
        Khởi tạo tracker
        
        Args:
            max_disappeared: Số khung hình tối đa một person có thể biến mất
            max_distance: Khoảng cách tối đa để coi là cùng một person
        """
        self.next_id = 1
        self.tracked_persons = {}  # {id: {'bbox', 'center', 'disappeared', 'attributes'}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Cập nhật tracker với detections mới
        
        Args:
            detections: List các detection từ detector
            
        Returns:
            Dict các tracked persons với ID
        """
        # Nếu không có detection nào
        if len(detections) == 0:
            # Tăng disappeared count cho tất cả tracked persons
            to_delete = []
            for person_id in self.tracked_persons:
                self.tracked_persons[person_id]['disappeared'] += 1
                if self.tracked_persons[person_id]['disappeared'] > self.max_disappeared:
                    to_delete.append(person_id)
            
            # Xóa các person đã biến mất quá lâu
            for person_id in to_delete:
                del self.tracked_persons[person_id]
                
            return self.tracked_persons
        
        # Nếu chưa có person nào được track
        if len(self.tracked_persons) == 0:
            for detection in detections:
                self._register_new_person(detection)
        else:
            # Tính toán ma trận khoảng cách
            person_ids = list(self.tracked_persons.keys())
            person_centers = [self.tracked_persons[pid]['center'] for pid in person_ids]
            detection_centers = [det['center'] for det in detections]
            
            distance_matrix = np.zeros((len(person_centers), len(detection_centers)))
            for i, p_center in enumerate(person_centers):
                for j, d_center in enumerate(detection_centers):
                    distance_matrix[i][j] = euclidean(p_center, d_center)
            
            # Gán detections cho tracked persons
            used_detection_indices = set()
            used_person_indices = set()
            
            # Tìm các cặp (person, detection) có khoảng cách nhỏ nhất
            for _ in range(min(len(person_ids), len(detections))):
                min_distance = np.inf
                min_person_idx = -1
                min_detection_idx = -1
                
                for i in range(len(person_ids)):
                    if i in used_person_indices:
                        continue
                    for j in range(len(detections)):
                        if j in used_detection_indices:
                            continue
                        if distance_matrix[i][j] < min_distance:
                            min_distance = distance_matrix[i][j]
                            min_person_idx = i
                            min_detection_idx = j
                
                # Nếu khoảng cách nhỏ hơn threshold, gán detection cho person
                if min_distance < self.max_distance:
                    person_id = person_ids[min_person_idx]
                    detection = detections[min_detection_idx]
                    
                    # Cập nhật thông tin person
                    self.tracked_persons[person_id]['bbox'] = detection['bbox']
                    self.tracked_persons[person_id]['center'] = detection['center']
                    self.tracked_persons[person_id]['confidence'] = detection['confidence']
                    self.tracked_persons[person_id]['disappeared'] = 0
                    
                    used_person_indices.add(min_person_idx)
                    used_detection_indices.add(min_detection_idx)
                else:
                    break
            
            # Tạo person mới cho các detection chưa được gán
            for j in range(len(detections)):
                if j not in used_detection_indices:
                    self._register_new_person(detections[j])
            
            # Tăng disappeared count cho các person không được gán detection
            to_delete = []
            for i, person_id in enumerate(person_ids):
                if i not in used_person_indices:
                    self.tracked_persons[person_id]['disappeared'] += 1
                    if self.tracked_persons[person_id]['disappeared'] > self.max_disappeared:
                        to_delete.append(person_id)
            
            # Xóa các person đã biến mất quá lâu
            for person_id in to_delete:
                del self.tracked_persons[person_id]
        
        return self.tracked_persons
    
    def _register_new_person(self, detection: Dict[str, Any]) -> int:
        """
        Đăng ký person mới
        
        Args:
            detection: Detection information
            
        Returns:
            ID của person mới
        """
        person_id = self.next_id
        self.tracked_persons[person_id] = {
            'bbox': detection['bbox'],
            'center': detection['center'],
            'confidence': detection['confidence'],
            'disappeared': 0,
            'attributes': {}  # Sẽ được cập nhật bởi attribute recognizer
        }
        self.next_id += 1
        return person_id
    
    def draw_tracked_persons(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ các tracked persons lên khung hình
        
        Args:
            frame: Khung hình gốc
            
        Returns:
            Khung hình đã vẽ tracked persons
        """
        frame_copy = frame.copy()
        
        for person_id, person_info in self.tracked_persons.items():
            bbox = person_info['bbox']
            confidence = person_info['confidence']
            
            # Chọn màu dựa trên ID
            color = self._get_color_for_id(person_id)
            
            # Vẽ bounding box
            cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Vẽ ID và confidence
            label = f"ID: {person_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(frame_copy, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Vẽ center point
            center = person_info['center']
            cv2.circle(frame_copy, tuple(center), 5, color, -1)
        
        return frame_copy
    
    def _get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
        """
        Tạo màu duy nhất cho mỗi ID
        
        Args:
            person_id: ID của person
            
        Returns:
            Tuple màu BGR
        """
        # Tạo màu dựa trên ID
        np.random.seed(person_id)
        color = np.random.randint(0, 255, 3)
        return tuple(color.tolist())
    
    def get_person_crop(self, frame: np.ndarray, person_id: int) -> np.ndarray:
        """
        Cắt hình ảnh của person từ khung hình
        
        Args:
            frame: Khung hình gốc
            person_id: ID của person
            
        Returns:
            Hình ảnh đã cắt của person
        """
        if person_id not in self.tracked_persons:
            return None
        
        bbox = self.tracked_persons[person_id]['bbox']
        x1, y1, x2, y2 = bbox
        
        # Đảm bảo tọa độ trong phạm vi khung hình
        height, width = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        return frame[y1:y2, x1:x2]