"""
CharacterForge AI - Backend Server

FastAPI + HuggingFace Diffusers local AI image generation backend.

Portfolio-oriented features:
- Product information endpoint
- Model / checkpoint / LoRA metadata
- Character preset API
- Async generation jobs
- Gallery metadata
- Password-based NSFW/SFW access control
- CUDA / VRAM performance metadata
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import ImagePipeline
from .config import settings

BASE_DIR = Path(__file__).parent.parent
frontend_dir = BASE_DIR / "frontend"
outputs_dir = BASE_DIR / "outputs"
models_dir = BASE_DIR / "models"
checkpoints_dir = BASE_DIR / "checkpoints"
loras_dir = BASE_DIR / "loras"
presets_dir = BASE_DIR / "presets"

for directory in [outputs_dir, models_dir, checkpoints_dir, loras_dir, presets_dir]:
    directory.mkdir(exist_ok=True)

NSFW_PASSWORD = os.getenv("NSFW_PASSWORD", "1234")
NSFW_COOKIE_NAME = "nsfw_token"
NSFW_COOKIE_VALUE = "unlocked"

NSFW_KEYWORDS = [
    "nsfw", "porn", "pornmaster", "hentai", "nude", "nudity",
    "sex", "sexy", "xxx", "18plus", "18+", "adult",
]

pipeline: Optional[ImagePipeline] = None
jobs: Dict[str, Dict] = {}


def get_file_size_mb(item: Path) -> float:
    if item.is_dir():
        total = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
    else:
        total = item.stat().st_size
    return round(total / 1e6, 1)


def is_model_file(item: Path) -> bool:
    if item.name.startswith(".") or item.name.lower().endswith(".md"):
        return False
    return item.is_dir() or item.suffix.lower() in [".ckpt", ".safetensors", ".bin"]


def read_sidecar_metadata(item: Path) -> dict:
    meta_file = item.with_suffix(".json")
    if not meta_file.exists():
        return {}
    try:
        with open(meta_file, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"   [WARN] Failed to read metadata for {item.name}: {e}")
        return {}


def detect_nsfw(item: Path, meta: Optional[dict] = None) -> bool:
    meta = meta or {}
    if "nsfw" in meta:
        return bool(meta.get("nsfw"))
    name = item.name.lower()
    return any(keyword in name for keyword in NSFW_KEYWORDS)


def has_nsfw_access(request: Request) -> bool:
    return request.cookies.get(NSFW_COOKIE_NAME) == NSFW_COOKIE_VALUE


def model_file_info(item: Path, source: str) -> dict:
    meta = read_sidecar_metadata(item)
    is_nsfw = detect_nsfw(item, meta)
    return {
        "name": item.name,
        "filename": item.name,
        "path": str(item),
        "type": "hf_directory" if item.is_dir() else item.suffix.lstrip(".").lower(),
        "size_mb": get_file_size_mb(item),
        "source": source,
        "category": "nsfw" if is_nsfw else "sfw",
        "nsfw": is_nsfw,
        "base_model": meta.get("base_model", "unknown"),
        "description": meta.get("description", ""),
        "recommended_steps": meta.get("recommended_steps", 20),
        "recommended_cfg": meta.get("recommended_cfg", 7.0),
        "recommended_resolution": meta.get("recommended_resolution", "768x768"),
        "tags": meta.get("tags", []),
    }


def find_model_by_name(model_name: str) -> Optional[Path]:
    for search_dir in [models_dir, checkpoints_dir]:
        if not search_dir.exists():
            continue
        for item in search_dir.iterdir():
            if not is_model_file(item):
                continue
            if item.name == model_name or item.stem == model_name:
                return item

    model_name_lower = model_name.lower()
    for search_dir in [models_dir, checkpoints_dir]:
        if not search_dir.exists():
            continue
        for item in search_dir.iterdir():
            if not is_model_file(item):
                continue
            if model_name_lower in item.name.lower():
                return item
    return None


def filter_by_nsfw_access(items: list[dict], request: Request) -> list[dict]:
    if has_nsfw_access(request):
        return items
    return [item for item in items if not item.get("nsfw", False)]


def load_json_file(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback


def default_character_presets() -> list[dict]:
    return [
        {
            "id": "portfolio-cinematic",
            "name": "Portfolio Cinematic Portrait",
            "category": "style",
            "prompt_template": "{subject}, cinematic portrait, detailed eyes, dramatic rim light, high detail, professional composition",
            "negative_prompt": "blurry, low quality, watermark, text, deformed, bad anatomy",
            "recommended_steps": 14,
            "recommended_cfg": 6.5,
            "recommended_resolution": "768x768",
            "recommended_loras": [],
            "description": "A general-purpose cinematic preset for portfolio screenshots."
        },
        {
            "id": "character-consistency",
            "name": "Character Consistency",
            "category": "character",
            "prompt_template": "{subject}, consistent character design, reference sheet style, clean linework, detailed outfit, neutral background",
            "negative_prompt": "blurry, inconsistent face, low quality, extra limbs, text, watermark",
            "recommended_steps": 18,
            "recommended_cfg": 7.0,
            "recommended_resolution": "768x768",
            "recommended_loras": [],
            "description": "A preset focused on stable character identity and reusable outputs."
        }
    ]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1500, description="Text prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    width: int = Field(default=512, ge=256, le=1536, description="Image width")
    height: int = Field(default=512, ge=256, le=1536, description="Image height")
    steps: int = Field(default=20, ge=1, le=100, description="Inference steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="CFG scale")
    seed: int = Field(default=-1, description="Random seed (-1 = random)")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images")
    lora_names: List[str] = Field(default_factory=list, description="LoRA filenames to apply")
    lora_scales: List[float] = Field(default_factory=list, description="Scale per LoRA")
    clip_skip: int = Field(default=1, ge=1, le=4, description="CLIP Skip")
    performance_mode: str = Field(default="fast", description="fast | balanced | quality")
    vram_mode: str = Field(default="balanced", description="max_speed | balanced | low_vram | ultra_low_vram")
    ram_limit_gb: float = Field(default=0, ge=0, le=256, description="Soft RAM guard in GB. 0 = disabled")


class LoadModelRequest(BaseModel):
    model_name: str = Field(default="", description="Model filename or folder name")


class ApplyLoraRequest(BaseModel):
    lora_names: List[str] = Field(..., description="LoRA filenames to apply")
    lora_scales: List[float] = Field(default_factory=list, description="Scale per LoRA")


class NSFWLoginRequest(BaseModel):
    password: str = Field(..., description="NSFW unlock password")


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    images: List[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    metadata: Optional[Dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("[CharacterForge AI] Starting server...")
    try:
        pipeline = ImagePipeline(settings)
        await pipeline.load()
        print("[CharacterForge AI] Model loaded OK!")
    except Exception as e:
        print(f"[CharacterForge AI] Model not loaded: {e}")
        print("  -> Place your model in /models or /checkpoints and restart.")
    yield
    if pipeline:
        pipeline.unload()
    print("[CharacterForge AI] Server shutting down.")


app = FastAPI(
    title="CharacterForge AI",
    description="Local AI character generation studio powered by FastAPI, PyTorch, and HuggingFace Diffusers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/api/product")
async def product_info():
    return {
        "name": "CharacterForge AI",
        "tagline": "Local AI Character Generation Studio",
        "author": "SD / @phucvb588",
        "description": (
            "A local-first AI image generation app focused on character creation, LoRA workflows, "
            "metadata-driven model management, gallery history, content access control, "
            "and CUDA/VRAM optimized inference."
        ),
        "tech_stack": [
            "Python 3.11", "FastAPI", "PyTorch CUDA", "HuggingFace Diffusers",
            "Transformers", "Safetensors", "Vanilla JavaScript", "HTML/CSS"
        ],
        "features": [
            "Stable Diffusion / SDXL checkpoint loading",
            "LoRA management with runtime scale control",
            "Character and style preset API",
            "Async generation queue with progress polling",
            "Gallery with metadata output",
            "SFW/NSFW content access control",
            "CUDA acceleration and VRAM modes",
        ],
    }


@app.get("/api/auth/nsfw-status")
async def nsfw_status(request: Request):
    return {"unlocked": has_nsfw_access(request)}


@app.post("/api/auth/nsfw")
async def unlock_nsfw(payload: NSFWLoginRequest, response: Response):
    if payload.password != NSFW_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong NSFW password")
    response.set_cookie(
        key=NSFW_COOKIE_NAME,
        value=NSFW_COOKIE_VALUE,
        httponly=False,
        samesite="lax",
        max_age=60 * 60 * 12,
    )
    return {"status": "ok", "unlocked": True, "message": "NSFW models unlocked"}


@app.post("/api/auth/nsfw-logout")
async def lock_nsfw(response: Response):
    response.delete_cookie(NSFW_COOKIE_NAME)
    return {"status": "ok", "unlocked": False, "message": "NSFW models locked"}


@app.get("/api/health")
async def health():
    global pipeline
    model_info = pipeline.get_info() if pipeline and pipeline.is_loaded else None
    return {
        "status": "ok",
        "app": "CharacterForge AI",
        "model_loaded": pipeline.is_loaded if pipeline else False,
        "model_info": model_info,
        "server_time": datetime.now().isoformat(),
    }


@app.get("/api/models")
async def list_models(request: Request):
    models = []
    if models_dir.exists():
        for item in models_dir.iterdir():
            if is_model_file(item):
                models.append(model_file_info(item, "models"))
    return filter_by_nsfw_access(models, request)


@app.get("/api/checkpoints")
async def list_checkpoints(request: Request):
    checkpoints = []
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if is_model_file(item):
                checkpoints.append(model_file_info(item, "checkpoints"))
    return filter_by_nsfw_access(checkpoints, request)


@app.get("/api/loras")
async def list_loras():
    loras = []
    if loras_dir.exists():
        for item in loras_dir.glob("*.safetensors"):
            meta = read_sidecar_metadata(item)
            loras.append({
                "name": item.name,
                "filename": item.name,
                "stem": item.stem,
                "path": str(item),
                "type": "safetensors",
                "size_mb": round(item.stat().st_size / 1e6, 1),
                "description": meta.get("description", ""),
                "base_model": meta.get("base_model", "unknown"),
                "trigger_words": meta.get("trigger_words", []),
                "recommended_scale": meta.get("recommended_scale", 0.8),
                "tags": meta.get("tags", []),
                "nsfw": detect_nsfw(item, meta),
            })
    return loras


@app.get("/api/presets")
async def list_presets():
    preset_file = presets_dir / "character_presets.json"
    return load_json_file(preset_file, default_character_presets())


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    global pipeline
    if not pipeline or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check /models directory.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "images": [],
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "request": request.model_dump(),
        "metadata": None,
    }
    background_tasks.add_task(run_generation, job_id, request)
    return GenerateResponse(job_id=job_id, status="queued", message="Generation job queued successfully.")


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**{k: v for k, v in jobs[job_id].items() if k != "request"})


@app.get("/api/jobs")
async def list_jobs(limit: int = 20):
    sorted_jobs = sorted(jobs.values(), key=lambda x: x["created_at"], reverse=True)
    return [{k: v for k, v in job.items() if k != "request"} for job in sorted_jobs[:limit]]


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs.pop(job_id)
    for img_url in job.get("images", []):
        img_path = outputs_dir / Path(img_url).name
        if img_path.exists():
            img_path.unlink()
    return {"message": "Job deleted"}


@app.get("/api/gallery")
async def gallery(limit: int = 50):
    images = []
    if outputs_dir.exists():
        png_files = sorted(outputs_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
        for f in png_files[:limit]:
            meta_file = f.with_suffix(".json")
            meta = {}
            try:
                if meta_file.exists():
                    with open(meta_file, encoding="utf-8") as mf:
                        meta = json.load(mf)
            except Exception as e:
                print(f"   [WARN] Failed to read metadata for {f.name}: {e}")
            images.append({
                "filename": f.name,
                "url": f"/outputs/{f.name}",
                "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "metadata": meta,
            })
    return images


@app.post("/api/load-model")
async def load_model(request: LoadModelRequest, http_request: Request):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialised")

    selected_model = find_model_by_name(request.model_name)
    if selected_model:
        meta = read_sidecar_metadata(selected_model)
        if detect_nsfw(selected_model, meta) and not has_nsfw_access(http_request):
            raise HTTPException(status_code=403, detail="NSFW model locked. Please enter NSFW password first.")

    try:
        await pipeline.load(model_name=request.model_name)
        return {"status": "ok", "model_info": pipeline.get_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/loras/apply")
async def apply_loras(request: ApplyLoraRequest):
    global pipeline
    if not pipeline or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Base model not loaded")
    try:
        await pipeline.apply_loras(request.lora_names, request.lora_scales)
        return {"status": "ok", "active_loras": pipeline._active_loras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/loras/clear")
async def clear_loras():
    global pipeline
    if not pipeline or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Base model not loaded")
    await pipeline.clear_loras()
    return {"status": "ok", "active_loras": []}


async def run_generation(job_id: str, request: GenerateRequest):
    global pipeline
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = 0

    try:
        start_time = time.time()

        def progress_callback(step: int, total: int):
            jobs[job_id]["progress"] = int((step / total) * 100)

        images = await pipeline.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            num_images=request.num_images,
            lora_names=request.lora_names,
            lora_scales=request.lora_scales,
            clip_skip=request.clip_skip,
            performance_mode=request.performance_mode,
            vram_mode=request.vram_mode,
            ram_limit_gb=request.ram_limit_gb,
            progress_callback=progress_callback,
        )

        elapsed = round(time.time() - start_time, 2)
        image_urls = []
        last_generation = pipeline.get_last_generation_info() if pipeline else {}

        for i, image in enumerate(images):
            filename = f"{job_id}_{i}.png"
            filepath = outputs_dir / filename
            image.save(str(filepath), "PNG")

            meta = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "lora_names": request.lora_names,
                "lora_scales": request.lora_scales,
                "clip_skip": request.clip_skip,
                "performance_mode": request.performance_mode,
                "vram_mode": request.vram_mode,
                "model": pipeline.get_info().get("name", "") if pipeline else "",
                "elapsed_seconds": elapsed,
                "pipeline_metrics": last_generation,
                "generated_at": datetime.now().isoformat(),
            }
            with open(filepath.with_suffix(".json"), "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
            image_urls.append(f"/outputs/{filename}")

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "images": image_urls,
            "completed_at": datetime.now().isoformat(),
            "metadata": {"elapsed_seconds": elapsed, "pipeline_metrics": last_generation},
        })
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e), "completed_at": datetime.now().isoformat()})
        print(f"Generation error for job {job_id}: {e}")

@app.get("/api/system-info")
async def system_info():
    info = {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "torch_version": None,
        "cuda_version": None,
        "ram": None,
        "vram": None,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)

            try:
                free_vram, total_vram = torch.cuda.mem_get_info()
                info["vram"] = {
                    "free_gb": round(free_vram / (1024 ** 3), 2),
                    "total_gb": round(total_vram / (1024 ** 3), 2),
                    "used_gb": round((total_vram - free_vram) / (1024 ** 3), 2),
                }
            except Exception:
                info["vram"] = None

    except Exception as e:
        info["torch_error"] = str(e)

    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram"] = {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "percent": mem.percent,
        }
    except Exception as e:
        info["ram_error"] = str(e)

    return info



if outputs_dir.exists():
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
