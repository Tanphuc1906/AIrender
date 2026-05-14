# 🎨 Thư mục LoRA

Đặt file LoRA (`.safetensors`) vào thư mục này để dùng trong AI Image Studio.

## LoRA là gì?

LoRA (Low-Rank Adaptation) là các file nhỏ giúp **tinh chỉnh phong cách** cho AI mà không cần thay base model.  
Ví dụ: LoRA vẽ anime, LoRA chân dung thực, LoRA concept art, v.v.

## Cách dùng

1. Tải file `.safetensors` từ [CivitAI](https://civitai.com) hoặc [HuggingFace](https://huggingface.co)
2. Đặt vào thư mục này
3. Tạo file metadata `.json` cùng tên (tùy chọn, xem mẫu bên dưới)
4. Chọn LoRA trong giao diện web khi generate

## Cấu trúc thư mục

```
loras/
  my_lora.safetensors       ← file LoRA
  my_lora.json              ← metadata (tùy chọn)
```

## Format file metadata `.json`

```json
{
  "description": "Mô tả ngắn về LoRA này",
  "base_model": "SD 1.5",
  "trigger_words": ["word1", "word2"],
  "recommended_scale": 0.8
}
```

| Field | Mô tả | Default |
|-------|-------|---------|
| `description` | Mô tả LoRA | `""` |
| `base_model` | Model gốc tương thích | `"unknown"` |
| `trigger_words` | Từ khoá cần thêm vào prompt | `[]` |
| `recommended_scale` | Cường độ khuyến nghị (0.0–2.0) | `0.8` |
| `nsfw` | Đánh dấu là NSFW — bị ẩn khi **Safe mode** | `false` |

## LoRA gợi ý từ CivitAI

| LoRA | Phong cách | Link |
|------|-----------|------|
| Anime Lineart | Anime nét vẽ tay | [civitai](https://civitai.com) |
| Detail Tweaker | Tăng chi tiết | [civitai](https://civitai.com) |
| LCM LoRA | Sinh ảnh siêu nhanh | [HF](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) |
| epiNoiseoffset | Cải thiện độ tối sáng | [civitai](https://civitai.com) |

## Lưu ý

- LoRA phải **tương thích** với base model đang dùng (SD 1.5 LoRA ≠ SDXL LoRA)
- Có thể dùng **nhiều LoRA cùng lúc**, mỗi cái có scale riêng
- Scale khuyến nghị: **0.5 – 1.0** (cao hơn = ảnh hưởng mạnh hơn)
