# 💾 Thư mục Checkpoints

Đặt file checkpoint (`.safetensors`, `.ckpt`) hoặc thư mục model vào đây.

## Checkpoint vs Base Model?

| | `/models` | `/checkpoints` |
|---|---|---|
| **Mục đích** | Model chính (mặc định load) | Model phụ / thay thế |
| **Format** | Bất kỳ | `.safetensors`, `.ckpt`, HF folder |
| **Load tự động** | ✅ Khi khởi động | ❌ Phải chọn thủ công |

> Cả hai thư mục đều hoạt động giống nhau, `/checkpoints` để **tổ chức** nhiều model.

## Cấu trúc thư mục

```
checkpoints/
  dreamshaper_8.safetensors       ← Checkpoint đơn file
  realistic_vision_v6/            ← HuggingFace folder format
    model_index.json
    unet/
    vae/
    ...
```

## Chuyển đổi model khi đang chạy

Dùng API (không cần restart server):

```bash
# Qua curl
curl -X POST http://localhost:8000/api/load-model \
  -H "Content-Type: application/json" \
  -d '{"model_name": "dreamshaper_8.safetensors"}'
```

Hoặc chọn trong **Settings → Switch Model** trên giao diện web.

## Model phổ biến để tải về

| Model | Style | Format | Link |
|-------|-------|--------|------|
| DreamShaper 8 | Realistic/Fantasy | `.safetensors` | [CivitAI](https://civitai.com/models/4384) |
| Realistic Vision V6 | Hyper-realistic | `.safetensors` | [CivitAI](https://civitai.com/models/4201) |
| AbsoluteReality | Chân dung thực | `.safetensors` | [CivitAI](https://civitai.com/models/81458) |
| Anything V5 | Anime | `.safetensors` | [CivitAI](https://civitai.com/models/9409) |
| SDXL Base 1.0 | General (1024px) | HF folder | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |

## Lưu ý

- File `.safetensors` **an toàn và nhanh hơn** `.ckpt` (khuyến dùng)
- SDXL model cần **8GB+ VRAM**
- SD 1.5 model cần **4GB VRAM** (hoặc CPU với 16GB RAM)
