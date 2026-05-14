# 🚀 AIrender — Studio Sáng Tạo Ảnh AI Local

**AIrender** là một giải pháp mã nguồn mở cho phép bạn tạo ra những tác phẩm nghệ thuật kỹ thuật số đỉnh cao bằng trí tuệ nhân tạo, chạy hoàn toàn trên máy tính cá nhân của bạn. Không cần đăng ký, không cần trả phí hàng tháng, và hoàn toàn riêng tư.

![AIrender Header](https://raw.githubusercontent.com/Tanphuc1906/AIrender/main/frontend/screenshot.png) *(Lưu ý: Thay link ảnh thực tế nếu bạn có)*

---

## ✨ AIrender là gì?

AIrender là một giao diện web (Web UI) hiện đại, tối giản và mạnh mẽ được xây dựng để tương tác với các mô hình Generative AI (như Stable Diffusion). Nó biến những dòng mô tả văn bản (prompts) của bạn thành hình ảnh chất lượng cao chỉ trong vài giây.

### 🌟 Công dụng chính:
- **Sáng tạo nghệ thuật**: Tạo minh họa, concept art, hoặc ảnh chân dung nghệ thuật.
- **Thiết kế đồ họa**: Tạo nhanh các asset, texture, hoặc ý tưởng thiết kế.
- **Quyền riêng tư tuyệt đối**: Mọi dữ liệu và hình ảnh đều nằm trên máy bạn, không gửi lên cloud.
- **Tùy biến không giới hạn**: Hỗ trợ các model chuyên biệt (Checkpoints) và phong cách riêng (LoRA).

---

## 🛠️ Tính năng nổi bật

- **Giao diện Glassmorphism**: Thiết kế hiện đại, mượt mà và trực quan.
- **Hỗ trợ đa Model**: Chạy được SD 1.5, SDXL, và các file `.safetensors` phổ biến.
- **Hệ thống LoRA**: Dễ dàng thêm các "gia vị" phong cách vào ảnh.
- **Gallery tích hợp**: Xem lại và quản lý lịch sử các ảnh đã tạo.
- **Tối ưu hóa hiệu suất**: Hỗ trợ cả GPU (NVIDIA CUDA) và CPU.

---

## 📁 Cấu trúc dự án

```text
AIrender/
├── 🧠 backend/       # Engine xử lý AI (FastAPI + Diffusers)
├── 🎨 frontend/      # Giao diện người dùng (HTML/CSS/JS)
├── 📁 models/        # Nơi chứa các model AI (.safetensors)
├── 📁 loras/         # Nơi chứa các file LoRA tùy chỉnh
└── 🖼️ outputs/       # Thư mục lưu trữ tác phẩm hoàn thiện
```

---

## 🚀 Hướng dẫn cài đặt nhanh

### 1. Chuẩn bị Model
Tải model (ví dụ từ Civitai) và đặt vào thư mục `models/`.
Xem thêm chi tiết tại [models/README.md](./models/README.md).

### 2. Chạy ứng dụng
**Trên Windows:**
Chạy file `start.bat`

**Trên Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

### 3. Trải nghiệm
Truy cập: `http://localhost:8000` trên trình duyệt của bạn.

---

## 📋 Yêu cầu hệ thống

| Linh kiện | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| **GPU** | 4GB VRAM (NVIDIA) | 8GB+ VRAM |
| **RAM** | 8GB | 16GB+ |
| **Ổ cứng** | 10GB trống | SSD (tốc độ cao) |

---

## 🤝 Đóng góp
Nếu bạn yêu thích dự án, hãy tặng cho AIrender một ⭐️ trên GitHub nhé!

---
*Phát triển bởi Tanphuc1906*
