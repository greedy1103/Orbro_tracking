#!/usr/bin/env python3
"""
Tag Tracker với Video Processing
Ứng dụng theo dõi người dựa trên tag với xử lý video thực
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import random
import time
import numpy as np
from typing import Dict, List, Set
import argparse # Thêm thư viện để đọc tham số dòng lệnh
import os # Thêm thư viện os để xử lý đường dẫn
import json # Thêm thư viện để xử lý JSON

# Import detector và tracker
try:
    from detector import PersonDetector
    from tracker import PersonTracker
    DETECTOR_AVAILABLE = True
    print("✅ Detector và Tracker imported successfully")
except ImportError as e:
    DETECTOR_AVAILABLE = False
    print(f"⚠️ Detector/Tracker không khả dụng (sẽ dùng mock): {e}")

try:
    from openpar_attribute_model import create_openpar_recognizer, OpenPARStyleAttributeModel as OpenPARAttributeModel
    ATTR_MODEL_AVAILABLE = True
    print("✅ OpenPAR Real Model imported successfully")
except ImportError as e:
    ATTR_MODEL_AVAILABLE = False
    print(f"⚠️  OpenPAR Model không khả dụng (sẽ dùng mock): {e}")

class VideoTagTracker:
    def __init__(self, root, model_path=None, checkpoint_path=None):
        self.root = root
        self.root.title("Video Tag Tracker - Theo dõi người với Video")
        self.root.geometry("1400x900")
        
        # --- START: Cấu trúc thư mục chuyên nghiệp ---
        # Tạo và chỉ định thư mục cho video
        self.videos_dir = "videos"
        os.makedirs(self.videos_dir, exist_ok=True)
        # --- END: Cấu trúc thư mục ---
        
        # Variables
        self.selected_tags = set()
        self.is_processing = False
        self.current_frame = None
        self.cap = None
        self.total_detected = 0
        self.total_matched = 0
        self.person_attributes = {} # Cache cho thuộc tính đã nhận dạng
        self.video_output_data = [] # List để lưu kết quả JSON của cả video
        self.video_source_name = None # Tên nguồn video để đặt tên file
        self.processing_thread = None # Tham chiếu đến luồng xử lý
        
        # Initialize detector and tracker
        self.detector = None
        self.tracker = None
        if DETECTOR_AVAILABLE and model_path:
            try:
                self.detector = PersonDetector(model_path=model_path, confidence_threshold=0.6) # Tăng ngưỡng để giảm nhận dạng sai
                self.tracker = PersonTracker(max_disappeared=50, max_distance=75) # Tinh chỉnh tracker
                print("✅ Detector và Tracker initialized")
            except Exception as e:
                print(f"⚠️ Không thể khởi tạo detector/tracker: {e}")
                self.detector = None
                self.tracker = None
        
        # Initialize person attribute recognizer
        if ATTR_MODEL_AVAILABLE:
            try:
                # --- START: Truyền đường dẫn checkpoint từ tham số dòng lệnh ---
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Tạo đường dẫn tuyệt đối cho checkpoint
                abs_checkpoint_path = os.path.join(script_dir, checkpoint_path) if checkpoint_path else None
                
                self.attr_recognizer = create_openpar_recognizer("promptpar", checkpoint_path=abs_checkpoint_path)
                # --- END: Truyền đường dẫn ---
                print("✅ OpenPAR Person Attribute Recognizer loaded")
                    
                if model_path:
                    print(f"📦 Model path provided: {model_path}")
                    # For YOLO detection model, not attribute model
                    
            except Exception as e:
                print(f"❌ Error loading recognizer: {e}")
                self.attr_recognizer = None
        else:
            self.attr_recognizer = None
        
        # --- START: Update tags to English ---
        self.available_tags = {
            'Gender': ['Male', 'Female'],
            'Age': ['Adult', 'Teenager', 'Child', 'Senior'],
            'Body Shape': ['Slim', 'Average', 'Heavy'],
            'Height': ['Short', 'Average', 'Tall'],
            'Upper Body': ['T-shirt', 'Shirt', 'Jacket', 'Sweater', 'Vest', 'Hoodie'],
            'Sleeve Length': ['Long-sleeved', 'Short-sleeved'],
            'Lower Body': ['Trousers', 'Shorts', 'Skirt', 'Jeans'],
            'Footwear': ['Leather Shoes', 'Sneakers', 'Boots'],
            'Accessories': ['Hat', 'Glasses', 'Handbag', 'Backpack', 'Tie', 'Headphones'],
            'Hair': ['Long Hair', 'Short Hair'],
        }
        # --- END: Update tags ---
        
        self.setup_ui()
        self.update_statistics()
        
        # --- START: Xử lý sự kiện đóng cửa sổ ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # --- END: Xử lý sự kiện ---
        
    def setup_ui(self):
        """Setup giao diện người dùng"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel
        left_frame = ttk.LabelFrame(main_frame, text="🏷️ Chọn Tag", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.setup_tag_selection(left_frame)
        
        # Right panel
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Controls
        control_frame = ttk.LabelFrame(right_frame, text="🎮 Điều khiển", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        self.setup_controls(control_frame)
        
        # Stats
        stats_frame = ttk.LabelFrame(right_frame, text="📊 Thống kê", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        self.setup_statistics(stats_frame)
        
        # Video
        video_frame = ttk.LabelFrame(right_frame, text="📹 Video", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        self.setup_video_display(video_frame)
        
    def setup_tag_selection(self, parent):
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tag_tree = ttk.Treeview(tree_frame, selectmode='none', height=20)
        self.tag_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tag_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tag_tree.configure(yscrollcommand=scrollbar.set)
        
        for category, tags in self.available_tags.items():
            category_id = self.tag_tree.insert('', tk.END, text=category, open=True)
            for tag in tags:
                self.tag_tree.insert(category_id, tk.END, text=f"☐ {tag}", tags=('tag',))
        
        self.tag_tree.bind('<Button-1>', self.on_tag_click)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Chọn tất cả", command=self.select_all_tags).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Bỏ chọn", command=self.deselect_all_tags).pack(side=tk.LEFT)
        
    def setup_controls(self, parent):
        btn_frame1 = ttk.Frame(parent)
        btn_frame1.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(btn_frame1, text="🎬 Video File", command=self.open_video_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame1, text="📹 Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame1, text="⏹️ Dừng", command=self.stop_processing).pack(side=tk.LEFT)
        
    def setup_statistics(self, parent):
        stats_grid = ttk.Frame(parent)
        stats_grid.pack(fill=tk.X)
        
        ttk.Label(stats_grid, text="Phát hiện:").grid(row=0, column=0, sticky=tk.W)
        self.total_detected_label = ttk.Label(stats_grid, text="0", foreground="blue")
        self.total_detected_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(stats_grid, text="Khớp tag:").grid(row=1, column=0, sticky=tk.W)
        self.total_matched_label = ttk.Label(stats_grid, text="0", foreground="green")
        self.total_matched_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(stats_grid, text="Trạng thái:").grid(row=2, column=0, sticky=tk.W)
        self.status_label = ttk.Label(stats_grid, text="Sẵn sàng", foreground="orange")
        self.status_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(stats_grid, text="Tags đã chọn:").grid(row=3, column=0, sticky=tk.W)
        self.selected_tags_label = ttk.Label(stats_grid, text="Không có", foreground="purple")
        self.selected_tags_label.grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
    def setup_video_display(self, parent):
        self.video_canvas = tk.Canvas(parent, bg='black', width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.info_text = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
    def on_tag_click(self, event):
        # Lấy item được click trên dòng
        item = self.tag_tree.identify_row(event.y)
        if not item:
            return
            
        # Kiểm tra nếu là tag (không phải category)
        if 'tag' in self.tag_tree.item(item, 'tags'):
            full_text = self.tag_tree.item(item, 'text')
            # Trích xuất tag text một cách an toàn
            tag_text = ' '.join(full_text.split(' ')[1:])
            
            if tag_text in self.selected_tags:
                self.selected_tags.remove(tag_text)
                self.tag_tree.item(item, text=f"☐ {tag_text}")
            else:
                self.selected_tags.add(tag_text)
                self.tag_tree.item(item, text=f"☑ {tag_text}")
            
            self.update_statistics()
            print(f"Selected tags: {self.selected_tags}")  # Debug
    
    def select_all_tags(self):
        self.selected_tags.clear()
        for tags in self.available_tags.values():
            self.selected_tags.update(tags)
        self.refresh_tag_tree_display()
        self.update_statistics()
    
    def deselect_all_tags(self):
        self.selected_tags.clear()
        self.refresh_tag_tree_display()
        self.update_statistics()
    
    def refresh_tag_tree_display(self):
        for category_item in self.tag_tree.get_children(''):
            for tag_item in self.tag_tree.get_children(category_item):
                full_text = self.tag_tree.item(tag_item, 'text')
                # Correctly extract tag_text, handling tags that might contain spaces
                tag_text = ' '.join(full_text.split(' ')[1:])
                
                if tag_text in self.selected_tags:
                    self.tag_tree.item(tag_item, text=f"☑ {tag_text}")
                else:
                    self.tag_tree.item(tag_item, text=f"☐ {tag_text}")
    
    def update_statistics(self):
        self.total_detected_label.config(text=str(self.total_detected))
        self.total_matched_label.config(text=str(self.total_matched))
        
        # Update selected tags display
        if self.selected_tags:
            tags_text = f"{len(self.selected_tags)} tags"
            if len(self.selected_tags) <= 3:
                tags_text = ', '.join(list(self.selected_tags)[:3])
            self.selected_tags_label.config(text=tags_text)
        else:
            self.selected_tags_label.config(text="Không có (theo dõi tất cả)")
    
    def open_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn video file",
            initialdir=self.videos_dir, # Mở thư mục 'videos' mặc định
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.start_video_processing(file_path)
    
    def start_webcam(self):
        self.start_video_processing(0)
    
    def start_video_processing(self, source):
        if self.is_processing:
            self.stop_processing()
        
        # Xóa cache và dữ liệu cũ khi bắt đầu video mới
        self.person_attributes.clear()
        self.video_output_data.clear()
        if isinstance(source, str):
            self.video_source_name = os.path.splitext(os.path.basename(source))[0]
        else:
            self.video_source_name = "webcam"
        
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", f"Không thể mở video: {source}")
                return
            
            self.is_processing = True
            self.status_label.config(text="Đang xử lý...")
            
            self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
            self.processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khởi tạo video: {str(e)}")
    
    def process_video(self):
        frame_count = 0
        
        while self.is_processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.current_frame = frame.copy()
            
            # Real person detection và tracking
            if self.detector and self.tracker:
                # Real detection
                detections = self.detector.detect_persons(frame)
                print(f"DEBUG: YOLO Detections: {len(detections)}") # DEBUG LINE
                persons = self.tracker.update(detections)
                self.total_detected = len(persons)
            else:
                # Fallback to mock
                persons = self.detect_persons_mock(frame)
                self.total_detected = len(persons)
            
            matched_count = 0
            
            # Info header with filtering status
            if self.selected_tags:
                info_lines = [
                    f"Frame {frame_count}: {len(persons)} người phát hiện",
                    f"🏷️ Lọc theo {len(self.selected_tags)} tags: {', '.join(list(self.selected_tags)[:3])}{'...' if len(self.selected_tags) > 3 else ''}",
                    f"📊 Chế độ: Chỉ hiển thị đối tượng khớp tags"
                ]
            else:
                info_lines = [
                    f"Frame {frame_count}: {len(persons)} người phát hiện", 
                    f"🏷️ Chưa chọn tags - theo dõi TẤT CẢ",
                    f"📊 Chế độ: Hiển thị tất cả đối tượng"
                ]
            
            # Process persons based on source (real tracker vs mock)
            if self.detector and self.tracker:
                # Real tracker returns dict with person_id: person_data
                person_items = persons.items()
            else:
                # Mock returns dict with person_id: person_info
                person_items = persons.items()

            # --- START: Logic for structured JSON output ---
            frame_output_data = []
                
            for person_id, person_info in person_items:
                # Get bbox based on data structure
                if self.detector and self.tracker:
                    # Real tracker: bbox is in person_info['bbox'] as [x1,y1,x2,y2]
                    bbox = person_info.get('bbox', [0, 0, 100, 100])
                    x1, y1, x2, y2 = map(int, bbox)
                else:
                    # Mock: bbox is tuple (x1,y1,x2,y2)
                    bbox = person_info['bbox']
                    x1, y1, x2, y2 = bbox
                
                # --- Tối ưu hóa nhận dạng thuộc tính ---
                person_tags = set()

                # 1. Kiểm tra xem người này đã được phân tích chưa
                if person_id in self.person_attributes:
                    person_tags = self.person_attributes[person_id]
                
                # 2. Nếu chưa, thực hiện nhận dạng
                elif self.attr_recognizer:
                    person_img = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                    if person_img is not None and person_img.size > 0:
                        # Giảm ngưỡng để model thuộc tính hoạt động
                        attr_result = self.attr_recognizer.predict(person_img, threshold=0.2)
                        # Lưu kết quả vào cache
                        person_tags = set(attr_result.get('positive_attributes', []))
                        self.person_attributes[person_id] = person_tags
                        print(f"DEBUG: New person P{person_id} analyzed. Tags: {person_tags}") # DEBUG
                
                # Create structured data object for the person
                person_data = {
                    "id": f"P{person_id}",
                    "bbox": [x1, y1, x2, y2],
                    "attributes": sorted(list(person_tags))
                }
                frame_output_data.append(person_data)

                # 3. So khớp tag (để vẽ bounding box)
                person_matched = False
                if self.selected_tags:
                    matched_tags = self.selected_tags.intersection(person_tags)
                    person_matched = len(matched_tags) > 0
                else:
                    person_matched = True
                    matched_tags = person_tags

                person_info['person_tags'] = person_tags
                person_info['matched'] = person_matched

                if person_matched:
                    matched_count += 1
                
                # Draw bbox - only for matched persons or all if no tags selected
                is_matched = person_info.get('matched', False)
                
                if is_matched or not self.selected_tags:  # Show all if no tags selected
                    # Green for matched, red for unmatched, blue for all when no filter
                    if not self.selected_tags:
                        color = (255, 165, 0)  # Orange - all persons when no filter
                        label = f"P{person_id}"
                    elif is_matched:
                        color = (0, 255, 0)  # Green - matched persons
                        label = f"P{person_id} ✓"
                        # Show matched tags on bbox
                        if matched_tags:
                            tag_text = ', '.join(list(matched_tags)[:2])  # Show max 2 tags
                            cv2.putText(frame, tag_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        # This case is now handled by the outer condition, so no red box is drawn when filtering
                        continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # If tags are selected and person doesn't match, we simply don't draw anything
            
            self.total_matched = matched_count
            
            # --- START: Lưu kết quả của frame vào list chung ---
            frame_log = {
                "frame_number": frame_count,
                "persons": frame_output_data
            }
            self.video_output_data.append(frame_log)
            # --- END: Lưu kết quả ---

            # Update UI with structured JSON (chỉ hiển thị frame hiện tại)
            json_output_display = json.dumps(frame_log, indent=4)
            self.root.after(0, self.update_video_display, frame)
            self.root.after(0, self.update_statistics)
            self.root.after(0, self.update_info_display, json_output_display)
            
            time.sleep(0.033)
        
        self.cap.release()
        self.is_processing = False
        self.root.after(0, lambda: self.status_label.config(text="Đã dừng, đang lưu..."))
        self.save_results_to_file()
    
    def detect_persons_mock(self, frame):
        h, w = frame.shape[:2]
        num_persons = random.randint(1, 4)
        persons = {}
        
        for i in range(num_persons):
            x1 = random.randint(0, w//2)
            y1 = random.randint(0, h//2)
            x2 = min(x1 + random.randint(80, 150), w)
            y2 = min(y1 + random.randint(150, 250), h)
            
            persons[i+1] = {
                'bbox': (x1, y1, x2, y2),
                'last_seen': time.time()
            }
        
        return persons
    
    def update_video_display(self, frame):
        if frame is not None:
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = frame.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                resized_frame = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                try:
                    from PIL import Image, ImageTk
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image)
                    
                    self.video_canvas.delete("all")
                    x = (canvas_width - new_w) // 2
                    y = (canvas_height - new_h) // 2
                    self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    self.video_canvas.image = photo
                except ImportError:
                    # Fallback without PIL
                    self.video_canvas.delete("all")
                    self.video_canvas.create_text(canvas_width//2, canvas_height//2, 
                                                text="Video đang chạy\n(cần PIL để hiển thị)", 
                                                fill="white", font=("Arial", 14))
    
    def update_info_display(self, text):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)
    
    def stop_processing(self):
        # The process_video loop will handle saving
        self.is_processing = False
        if self.cap:
            self.cap.release()
        # Status will be updated by the process_video thread upon exit
    
    def on_closing(self):
        """Xử lý khi người dùng nhấn nút X để đóng cửa sổ một cách an toàn."""
        if messagebox.askokcancel("Quit", "Bạn có muốn thoát chương trình không?"):
            self.is_processing = False # Gửi tín hiệu dừng
            
            # Kiểm tra xem luồng xử lý có đang chạy không
            if self.processing_thread and self.processing_thread.is_alive():
                self.status_label.config(text="Đang dừng và lưu file...")
                # Lên lịch kiểm tra lại sau 100ms
                self.root.after(100, self.check_thread_and_close)
            else:
                self.root.destroy()

    def check_thread_and_close(self):
        """Kiểm tra xem luồng xử lý đã dừng chưa, nếu rồi thì đóng cửa sổ."""
        if self.processing_thread.is_alive():
            # Nếu vẫn đang chạy, kiểm tra lại sau 100ms nữa
            self.root.after(100, self.check_thread_and_close)
        else:
            # Nếu đã dừng, đóng cửa sổ
            self.root.destroy()

    def save_results_to_file(self):
        if not self.video_output_data:
            print("Không có dữ liệu để lưu.")
            self.status_label.config(text="Đã dừng (không có dữ liệu)")
            return

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        filename = os.path.join(output_dir, f"{self.video_source_name}_analysis_{timestamp}.json")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.video_output_data, f, indent=4, ensure_ascii=False)
            
            abs_path = os.path.abspath(filename)
            print(f"✅ Kết quả đã được lưu vào: {abs_path}")
            self.status_label.config(text=f"Đã lưu file!")
            messagebox.showinfo("Lưu thành công", f"Kết quả phân tích đã được lưu vào:\n{abs_path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu file: {e}")
            self.status_label.config(text="Lỗi khi lưu file!")
            messagebox.showerror("Lỗi", f"Không thể lưu kết quả:\n{e}")


def main():
    # Thêm bộ phân tích tham số dòng lệnh để nhận --model và --checkpoint
    parser = argparse.ArgumentParser(description="Video Tag Tracker")
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 model (e.g., yolov8n.pt)')
    parser.add_argument('--checkpoint', type=str, default='PETA_Checkpoint.pth',
                        help='Attribute recognition checkpoint file (e.g., PETA_Checkpoint.pth or PA100k_Checkpoint.pth)')
    args = parser.parse_args()
    
    root = tk.Tk()
    # Truyền cả hai đường dẫn model và vào ứng dụng
    app = VideoTagTracker(root, model_path=args.model, checkpoint_path=args.checkpoint)
    
    print("✅ Video Tag Tracker đã khởi chạy!")
    print("📋 Tính năng:")
    print("   • Xử lý video file hoặc webcam")
    print("   • Nhận dạng thuộc tính người theo chuẩn OpenPAR")
    print("   • Lọc và theo dõi thông minh theo tag được chọn")
    
    root.mainloop()

if __name__ == "__main__":
    main()