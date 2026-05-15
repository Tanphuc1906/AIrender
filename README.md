# CharacterForge AI

**CharacterForge AI** is a local AI character generation studio built with **FastAPI**, **PyTorch**, **HuggingFace Diffusers**, and **vanilla JavaScript**.

The project focuses on local-first image generation, Stable Diffusion/SDXL model management, LoRA-based character workflows, prompt presets, gallery metadata, content access control, and CUDA/VRAM-optimized inference.

> Portfolio project by **SD / @phucvb588**

---

## Product Overview

CharacterForge AI turns a local Stable Diffusion setup into a small product-style web application.  
Instead of being only a basic prompt-to-image demo, it includes model metadata, LoRA management, character presets, gallery history, access control, and inference performance options.

---

## Features

### Image Generation

- Text-to-image generation
- Stable Diffusion 1.x / 2.x / SDXL support
- Single-file checkpoints: `.safetensors`, `.ckpt`, `.bin`
- HuggingFace Diffusers directory format
- Async generation job queue
- Progress polling
- Gallery output with metadata

### Model Management

- Base model discovery from `/models`
- Checkpoint discovery from `/checkpoints`
- LoRA discovery from `/loras`
- Runtime model switching
- Runtime LoRA loading and scale control
- Sidecar JSON metadata support

### Character Studio Features

- Character/style preset API
- Prompt template-ready backend
- Recommended steps, CFG, resolution, and LoRA metadata
- Metadata-driven model cards
- Gallery metadata for reproducible outputs

### Access Control

- SFW/NSFW model classification
- Password-based NSFW unlock
- NSFW models hidden by default
- Backend-enforced model loading protection

### Performance

- CUDA GPU acceleration
- FP16 on CUDA
- TF32 optimization
- DPMSolverMultistepScheduler + Karras sigmas
- VRAM modes: `max_speed`, `balanced`, `low_vram`, `ultra_low_vram`
- RAM guard using `psutil`
- Generation metrics saved with output metadata

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| AI Runtime | PyTorch, HuggingFace Diffusers |
| Model Format | Safetensors, CKPT, Diffusers folders |
| Frontend | HTML, CSS, Vanilla JavaScript |
| API Docs | FastAPI Swagger UI |
| Optimization | CUDA, FP16, TF32, DPMSolver scheduler |
| Metadata | JSON sidecar files |

---

## Project Structure

```text
AI/
├── backend/
│   ├── main.py
│   ├── pipeline.py
│   ├── config.py
│   └── debug_paths.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── models/
├── checkpoints/
├── loras/
├── outputs/
├── presets/
│   └── character_presets.json
├── requirements.txt
├── start.bat
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bat
cd /d E:\AI
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

For NVIDIA GPU:

```bat
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA:

```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

---

## Run

```bat
cd /d E:\AI
venv\Scripts\activate
start.bat
```

Open:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

---

## Model Folders

Place base models in:

```text
/models
```

Place checkpoints in:

```text
/checkpoints
```

Place LoRA files in:

```text
/loras
```

Supported model files:

```text
.safetensors
.ckpt
.bin
```

---

## Metadata Sidecar Files

You can place a `.json` file next to a model or LoRA.

Example:

```text
models/my_model.safetensors
models/my_model.json
```

Example metadata:

```json
{
  "nsfw": false,
  "base_model": "SDXL",
  "description": "Portfolio-safe cinematic model",
  "recommended_steps": 14,
  "recommended_cfg": 6.5,
  "recommended_resolution": "768x768",
  "tags": ["cinematic", "portrait", "portfolio"]
}
```

LoRA metadata example:

```json
{
  "nsfw": false,
  "base_model": "SDXL",
  "description": "Character LoRA for consistent portrait generation",
  "trigger_words": ["character_name"],
  "recommended_scale": 0.8,
  "tags": ["character", "portrait"]
}
```

---

## NSFW / SFW Access Control

NSFW models are hidden by default.

The password can be configured with:

```env
NSFW_PASSWORD=change-me
```

Default development password:

```text
1234
```

Backend endpoints:

```text
GET  /api/auth/nsfw-status
POST /api/auth/nsfw
POST /api/auth/nsfw-logout
```

---

## Main API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/product` | Product and portfolio information |
| GET | `/api/health` | Server and model status |
| GET | `/api/models` | List base models |
| GET | `/api/checkpoints` | List checkpoints |
| GET | `/api/loras` | List LoRAs |
| GET | `/api/presets` | List character/style presets |
| POST | `/api/generate` | Start image generation job |
| GET | `/api/jobs/{job_id}` | Get job status |
| GET | `/api/jobs` | List recent jobs |
| GET | `/api/gallery` | List generated images |
| POST | `/api/load-model` | Runtime model switching |
| POST | `/api/loras/apply` | Apply LoRA |
| POST | `/api/loras/clear` | Clear LoRA |

---

## Example Generate Request

```json
{
  "prompt": "anime character portrait, cinematic lighting, detailed eyes",
  "negative_prompt": "blurry, low quality, watermark, text",
  "width": 768,
  "height": 768,
  "steps": 14,
  "guidance_scale": 6.5,
  "seed": -1,
  "num_images": 1,
  "lora_names": [],
  "lora_scales": [],
  "clip_skip": 1,
  "performance_mode": "fast",
  "vram_mode": "balanced",
  "ram_limit_gb": 0
}
```

---

## Performance Notes

For faster generation:

```text
Resolution: 512x512 or 768x768
Steps: 10–14
CFG: 5.5–6.5
Performance mode: fast
VRAM mode: max_speed or balanced
```

For limited VRAM:

```text
VRAM mode: low_vram
Resolution: 512x512
Num images: 1
```

---

## Portfolio Highlights

This project demonstrates:

- Full-stack AI application development
- FastAPI backend design
- PyTorch / Diffusers inference integration
- Stable Diffusion and SDXL model loading
- LoRA workflow integration
- Async job tracking and progress polling
- Metadata-driven model and LoRA management
- Password-based content access control
- CUDA/VRAM performance optimization
- Gallery and reproducible metadata output

---

## CV Description

**CharacterForge AI — Local AI Character Generation Studio**

Built a local AI character generation web application using FastAPI, PyTorch, HuggingFace Diffusers, and JavaScript. The app supports SDXL checkpoints, LoRA-based character/style workflows, runtime model switching, asynchronous generation jobs, gallery metadata, password-based content access control, and CUDA/VRAM-optimized inference.

---

## GitHub Notes

Do not upload large model files to GitHub.

Recommended `.gitignore`:

```gitignore
venv/
env/
__pycache__/
*.pyc
.env

models/
checkpoints/
loras/
outputs/

*.safetensors
*.ckpt
*.bin
*.pt
*.pth

.vscode/
.DS_Store
```

---

## License

Personal portfolio project. Add a license before public release if needed.
