# 🚗 Vietnamese Vehicle License Plate Recognition (ALPR)

Hệ thống nhận dạng biển số xe Việt Nam sử dụng Deep Learning — hỗ trợ xử lý ảnh, video và camera thời gian thực, kèm giao diện đồ họa (GUI) trực quan.

---

## Tổng quan

Dự án xây dựng một pipeline **Automatic License Plate Recognition (ALPR)** hoàn chỉnh gồm 2 giai đoạn:

| Giai đoạn | Mô hình | Mô tả |
|-----------|---------|-------|
| **Phát hiện biển số** | YOLOv8n (custom-trained) | Định vị vùng biển số trên ảnh đầu vào |
| **Nhận dạng ký tự** | PaddleOCR | Trích xuất chuỗi ký tự từ vùng biển số đã crop |

Kết quả cuối cùng trả về: bounding box, chuỗi biển số và độ tin cậy cho từng biển số phát hiện được.

---

## Kiến trúc hệ thống

```
Input (Ảnh / Video / Webcam)
        │
        ▼
┌─────────────────────┐
│  YOLOv8n Detection  │  ← plate_yolov8n_320_2024.pt
│  (Phát hiện biển số)│
└────────┬────────────┘
         │  Crop + Padding
         ▼
┌─────────────────────┐
│    PaddleOCR        │
│  (Nhận dạng ký tự)  │
└────────┬────────────┘
         │  Regex cleanup + Format correction
         ▼
   Kết quả biển số
   (text, confidence)
```

---

## Tính năng

- **Xử lý ảnh** — Phát hiện và đọc biển số từ ảnh tĩnh
- **Xử lý video** — Phân tích video frame-by-frame với tuỳ chỉnh skip-frame
- **Webcam real-time** — Nhận dạng biển số trực tiếp từ camera
- **GUI Desktop** — Giao diện Tkinter đầy đủ chức năng: tải ảnh/video, điều chỉnh ngưỡng, xem kết quả dạng bảng, xuất CSV
- **CLI** — Chạy nhanh qua command line với đầy đủ tham số tuỳ chỉnh
- **GPU acceleration** — Tự động detect và sử dụng CUDA nếu có

---

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python |
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | PaddleOCR |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| GUI | Tkinter |
| Inference | CPU / CUDA GPU |

---

## Cấu trúc dự án

```
├── alpr_system.py      # Core engine — detection + OCR pipeline, CLI entry point
├── alpr_gui.py         # Desktop GUI application (Tkinter)
├── requirements.txt    # Dependencies
├── weights/
│   └── plate_yolov8n_320_2024.pt   # Custom-trained YOLOv8n model
└── input_images/       # Ảnh / video mẫu để test
```

---

## Cài đặt

```bash
# Clone repository
git clone https://github.com/<username>/Vietnamese-vehicle-plate-recognition.git
cd Vietnamese-vehicle-plate-recognition

# Cài đặt dependencies
pip install -r requirements.txt
```

> **Yêu cầu:** Python >= 3.8. Nếu có GPU NVIDIA, cài thêm CUDA toolkit để tăng tốc inference.

---

## Sử dụng

### Giao diện đồ họa (GUI)

```bash
python alpr_gui.py
```

Giao diện cho phép:
- Tải ảnh/video → nhấn **Phân tích ảnh** để chạy nhận dạng
- Điều chỉnh ngưỡng phát hiện biển số và ngưỡng OCR bằng thanh trượt
- Xem kết quả dạng bảng (biển số, độ tin cậy)
- Xuất kết quả ra file CSV
- Xử lý video với điều khiển Play/Pause/Stop

### Command Line (CLI)

```bash
# Nhận dạng từ ảnh
python alpr_system.py --input input_images/photo.jpg

# Nhận dạng từ video
python alpr_system.py --input input_images/video.mp4 --output result.mp4

# Webcam real-time
python alpr_system.py --input 0

# Tuỳ chỉnh tham số
python alpr_system.py --input image.jpg --device cuda:0 --plate_conf 0.3 --ocr_threshold 0.85
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--input` | *(bắt buộc)* | Đường dẫn ảnh/video hoặc index camera |
| `--output` | `None` | Đường dẫn lưu kết quả |
| `--device` | `auto` | Thiết bị chạy: `auto`, `cpu`, `cuda:0` |
| `--plate_conf` | `0.25` | Ngưỡng confidence phát hiện biển số |
| `--ocr_threshold` | `0.9` | Ngưỡng confidence OCR |
| `--no_display` | `False` | Chế độ headless (không hiển thị cửa sổ) |

---

## Ví dụ kết quả

```
Detection Results:
Processing time: 0.142s
Plates found: 1
Plate 1: '29A 12345' (conf: 0.956)
```

---

## Tác giả

Dự án thực tập — Nhận dạng biển số xe Việt Nam sử dụng Deep Learning.
