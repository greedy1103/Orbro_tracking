# Advanced Person Detection, Tracking and Attribute Recognition System
# Hệ thống Phát hiện, Theo dõi và Nhận dạng Thuộc tính Người Nâng cao

Một hệ thống AI toàn diện và tiên tiến để phát hiện, theo dõi và nhận dạng thuộc tính của người trong video. Tích hợp YOLOv8, thuật toán tracking tiên tiến, và mô hình nhận dạng thuộc tính đa lớp dựa trên Air-Clothing-MA với 13 thuộc tính chi tiết.

## 🌟 Tính năng Nâng cao

### 🔥 Tính năng Cốt lõi
- **Phát hiện người**: YOLOv8 với độ chính xác cao và tối ưu hiệu suất
- **Theo dõi người**: Thuật toán tracking tiên tiến với ID duy nhất và xử lý occlusion
- **Nhận dạng thuộc tính cơ bản**: Phân tích nhanh quần áo, màu sắc, phụ kiện cơ bản
- **🏷️ Tag-based Tracker**: Ứng dụng GUI để theo dõi người dựa trên tag/thuộc tính được chọn

### 🧠 Tính năng Nâng cao (Air-Clothing-MA Inspired)
- **Multi-Attribute Classification**: 13 thuộc tính chi tiết theo chuẩn Air-Clothing-MA
  - **Top Clothing**: Màu (14 loại), Pattern (6 loại), Giới tính, Mùa, Loại áo (7 loại), Tay áo (3 loại)
  - **Bottom Clothing**: Màu (14 loại), Pattern (6 loại), Giới tính, Mùa, Độ dài (2 loại), Loại quần (2 loại)
  - **Pose Analysis**: Tư thế cơ thể (3 loại)
- **Confidence-based Filtering**: Lọc kết quả dựa trên độ tin cậy cho từng thuộc tính
- **Person Similarity Analysis**: So sánh độ tương đồng giữa các người
- **Temporal Consistency**: Theo dõi thay đổi thuộc tính theo thời gian
- **Advanced Statistics**: Phân tích chi tiết confidence, clothing combinations
- **Real-time Mode Switching**: Chuyển đổi giữa chế độ basic/advanced (phím A)

### 💡 Tính năng Khác
- **Dual Processing Mode**: Basic (nhanh) và Advanced (chính xác) có thể chuyển đổi
- **GPU Acceleration**: Hỗ trợ CUDA với attention mechanism
- **Advanced JSON Reports**: Báo cáo chi tiết với phân tích thống kê
- **Performance Benchmarking**: So sánh hiệu suất basic vs advanced
- **Interactive Controls**: Điều khiển real-time (Q-quit, S-screenshot, A-toggle)

### 🏷️ Tag-based Tracker (Mới!)
- **Selective Tracking**: Chọn các tag/thuộc tính cụ thể để theo dõi
- **Real-time Filtering**: Hiển thị chỉ những người có thuộc tính được chọn
- **Interactive GUI**: Giao diện thân thiện với tkinter
- **Multi-tag Selection**: Chọn nhiều tag cùng lúc
- **Live Statistics**: Thống kê số người phát hiện và khớp với tag

## 🏗️ Kiến trúc Hệ thống Nâng cao

```
├── detector.py                    # Module phát hiện người (YOLOv8)
├── tracker.py                     # Module theo dõi người (ID tracking)
├── attribute_recognizer.py        # Module nhận dạng thuộc tính cơ bản
├── advanced_attribute_recognizer.py # Module nâng cao (Air-Clothing-MA)
├── simple_tag_tracker_minimal.py  # 🏷️ Minimal Tag Tracker (demo)
├── tag_tracker_with_video.py      # 🎬 Video Tag Tracker (xử lý video thực)
├── person_attribute_model.py      # 🤖 Person Attribute Recognition Model
├── main.py                        # Script chính cơ bản (có tích hợp tag tracker)
├── main_advanced.py               # Script chính nâng cao (có tích hợp tag tracker)
├── run_minimal_tracker.py         # Script khởi chạy minimal tag tracker
├── run_video_tag_tracker.py       # Script khởi chạy video tag tracker
├── quick_test.py                  # Test nhanh hệ thống
├── setup.py                       # Script cài đặt tự động
├── requirements.txt               # Dependencies
├── README.md                      # Tài liệu này
└── INSTALL.md                     # Hướng dẫn cài đặt chi tiết
```

## 📋 Yêu cầu Hệ thống

- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)
- Webcam or video files for input

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd person-analysis-system
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Tải xuống mô hình YOLOv8
Mô hình sẽ được tự động tải xuống khi chạy lần đầu, hoặc bạn có thể tải thủ công:
```bash
# Mô hình nano (nhẹ, nhanh)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Mô hình small (cân bằng)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Mô hình medium (chính xác hơn)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

## 💻 Sử dụng

### 🏷️ Tag-based Tracker (Đã cập nhật!)

#### Minimal Tag Tracker (Demo với dữ liệu giả)
```bash
# Khởi chạy Tag Tracker GUI từ main script
python main.py --tag-tracker

# Hoặc khởi chạy từ main_advanced  
python main_advanced.py --tag-tracker

# Hoặc chạy trực tiếp
python run_minimal_tracker.py
```

#### 🎬 Video Tag Tracker (Mới - với xử lý video thực!)
```bash
# Khởi chạy Video Tag Tracker từ main script
python main.py --video-tracker

# Hoặc khởi chạy từ main_advanced
python main_advanced.py --video-tracker

# Hoặc chạy trực tiếp
python run_video_tag_tracker.py
```

**Tính năng Tag Tracker:**
- ✅ **Xử lý video thực**: Hỗ trợ video file và webcam
- ✅ **Nhận dạng thuộc tính**: Sử dụng person attribute recognition model
- ✅ **Lọc theo tag**: Chọn các tag/thuộc tính cụ thể để theo dõi
- ✅ **Hiển thị real-time**: Video với bounding box và thông tin thuộc tính
- ✅ **Thống kê trực quan**: Số người phát hiện và khớp với tag
- ✅ **Giao diện thân thiện**: GUI với 3 panel chuyên nghiệp

### 🚀 Cách sử dụng Nâng cao (Khuyến nghị)

```bash
# Xử lý với tính năng nâng cao
python main_advanced.py --input your_video.mp4

# Xử lý với mô hình attribute tùy chỉnh
python main_advanced.py --input video.mp4 --advanced-model model.pth

# Chế độ không hiển thị với phân tích chi tiết
python main_advanced.py --input video.mp4 --no-display --output result.mp4

# Chỉ sử dụng tính năng cơ bản (nhanh hơn)
python main_advanced.py --input video.mp4 --no-advanced
```

### 📝 Cách sử dụng Cơ bản

```bash
# Xử lý video đầu vào (chế độ cơ bản)
python main.py --input path/to/your/video.mp4

# Xử lý và lưu kết quả
python main.py --input input.mp4 --output output.mp4

# Sử dụng mô hình YOLOv8 khác
python main.py --input video.mp4 --model yolov8s.pt
```

### Tùy chọn nâng cao

```bash
python main.py \
  --input video.mp4 \
  --output processed_video.mp4 \
  --model yolov8m.pt \
  --confidence 0.6 \
  --max-disappeared 50 \
  --max-distance 80
```

### Tham số chi tiết

- `--input, -i`: Đường dẫn video đầu vào (bắt buộc)
- `--output, -o`: Đường dẫn video đầu ra (tùy chọn)
- `--model, -m`: Mô hình YOLOv8 (default: yolov8n.pt)
- `--confidence, -c`: Ngưỡng tin cậy detection (default: 0.5)
- `--max-disappeared`: Số frame tối đa person có thể biến mất (default: 30)
- `--max-distance`: Khoảng cách tối đa để track person (default: 100)
- `--no-display`: Không hiển thị kết quả real-time
- `--tag-tracker`: Khởi chạy Tag Tracker GUI (demo với dữ liệu giả)
- `--video-tracker`: Khởi chạy Video Tag Tracker (xử lý video thực)

## 🎮 Điều khiển trong quá trình chạy

### Chế độ Cơ bản
- **q**: Thoát chương trình
- **s**: Lưu frame hiện tại thành ảnh

### Chế độ Nâng cao
- **q**: Thoát chương trình
- **s**: Lưu frame hiện tại thành ảnh
- **a**: Chuyển đổi giữa chế độ Advanced/Basic real-time

## 🧪 Demo và Test

### Test Nhanh Hệ thống
```bash
# Test tất cả components
python quick_test.py

# Test chỉ advanced features
python demo_advanced.py --mode model-test

# Benchmark hiệu suất
python demo_advanced.py --mode benchmark
```

### Demo với Video
```bash
# Tạo và xử lý demo video nâng cao
python demo_advanced.py --mode processing-demo

# Tạo demo video tùy chỉnh
python demo_advanced.py --mode create-demo --demo-duration 15

# Demo với webcam (nếu có)
python demo.py --mode webcam
```

## 📊 Kết quả đầu ra

### 1. Video đã xử lý
Nếu chỉ định `--output`, video sẽ được lưu với:
- Bounding boxes màu sắc khác nhau cho mỗi person
- ID và confidence score
- Thông tin thuộc tính (quần áo, màu sắc, phụ kiện)
- Thống kê real-time

### 2. Báo cáo JSON
File `analysis_results_[timestamp].json` chứa:
```json
{
  "video_info": {
    "path": "input.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 1500
  },
  "persons": {
    "1": {
      "first_seen_frame": 15,
      "total_appearances": 1245,
      "attributes": {
        "clothing": {...},
        "colors": [...],
        "accessories": [...],
        "body_attributes": {...},
        "pose": {...}
      }
    }
  },
  "summary": {
    "total_frames_processed": 1500,
    "total_detections": 2340,
    "unique_persons": 5,
    "average_processing_time": 0.045,
    "average_fps": 22.2
  }
}
```

## 🔧 Tùy chỉnh và Mở rộng

### Cải thiện Attribute Recognition

Hiện tại module `attribute_recognizer.py` sử dụng các hàm placeholder. Để cải thiện độ chính xác:

1. **Tích hợp mô hình pre-trained**:
   - Clothing classification models
   - Color recognition models
   - Accessory detection models

2. **Thêm thuộc tính mới**:
   - Age estimation
   - Gender recognition
   - Emotion detection
   - Action recognition

### Ví dụ tích hợp mô hình clothing classification:

```python
# Trong attribute_recognizer.py
def _analyze_clothing(self, person_image: np.ndarray) -> Dict[str, Any]:
    # Thay thế placeholder bằng mô hình thực
    clothing_model = load_clothing_model()
    predictions = clothing_model.predict(person_image)
    return predictions
```

## 🚀 Tối ưu hiệu suất

### 1. GPU Acceleration
```python
# Kiểm tra GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name()}")
```

### 2. Batch Processing
Để xử lý nhiều video:
```bash
for video in *.mp4; do
    python main.py --input "$video" --output "processed_$video" --no-display
done
```

### 3. Memory Optimization
- Giảm resolution input nếu cần
- Tăng interval phân tích attributes (hiện tại: mỗi 10 frames)
- Sử dụng mô hình YOLOv8 nhỏ hơn

## 🐛 Troubleshooting

### Lỗi thường gặp

1. **ModuleNotFoundError: ultralytics**
   ```bash
   pip install ultralytics
   ```

2. **CUDA out of memory**
   - Giảm resolution video
   - Sử dụng YOLOv8n thay vì YOLOv8m/l
   - Xử lý batch size nhỏ hơn

3. **OpenCV không hiển thị video**
   - Cài đặt OpenCV với GUI support:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

4. **Hiệu suất chậm**
   - Sử dụng GPU
   - Giảm confidence threshold
   - Tăng max_disappeared để giảm tracking overhead

## 📈 Roadmap

- [ ] Tích hợp mô hình attribute recognition chuyên nghiệp
- [ ] Hỗ trợ multiple camera streams
- [ ] Web interface để upload và xử lý video
- [ ] Real-time processing từ webcam
- [ ] API REST cho integration
- [ ] Docker containerization
- [ ] Model quantization để tối ưu tốc độ
- [ ] Database integration để lưu trữ kết quả

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📄 License

Dự án này được phân phối dưới MIT License. Xem `LICENSE` file để biết thêm chi tiết.

## 📞 Liên hệ

- Tác giả: [Tên của bạn]
- Email: [email@example.com]
- GitHub: [github.com/username]

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework