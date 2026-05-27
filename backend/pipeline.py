"""
CharacterForge AI - Image Generation Pipeline

Local AI character generation pipeline built on HuggingFace Diffusers.

Features:
- Stable Diffusion 1.x / 2.x / SDXL support
- Single-file checkpoint loading: .safetensors / .ckpt / .bin
- HuggingFace directory format support
- LoRA loading and runtime switching
- CUDA, FP16, TF32, DPMSolver scheduler optimization
- VRAM modes and RAM guard
- SDXL/Pony-safe clip_skip handling
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .config import Settings

if TYPE_CHECKING:
    from PIL import Image

BASE_DIR = Path(__file__).parent.parent


class ImagePipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipe: Optional[Any] = None
        self.is_loaded: bool = False
        self._model_info: dict = {}
        self._active_loras: list = []
        self._device: str = "cpu"
        self._last_generation_info: dict = {}

    async def load(self, model_name: str = "") -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync, model_name)

    async def apply_loras(self, lora_names: list, scales: list | None = None) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._apply_loras_sync, lora_names, scales or [])

    async def clear_loras(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._clear_loras_sync)

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = -1,
        num_images: int = 1,
        lora_names: Optional[list] = None,
        lora_scales: Optional[list] = None,
        clip_skip: int = 1,
        performance_mode: str = "fast",
        vram_mode: str = "balanced",
        ram_limit_gb: float = 0,
        progress_callback: Optional[Callable] = None,
    ) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
            num_images,
            lora_names or [],
            lora_scales or [],
            clip_skip,
            performance_mode,
            vram_mode,
            ram_limit_gb,
            progress_callback,
        )

    def _load_sync(self, model_name: str = "") -> None:
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        if self.pipe is not None:
            self.unload()

        model_path = self._find_model(model_name)
        if not model_path:
            raise FileNotFoundError(
                "No model found in /models or /checkpoints directory. "
                "Supported: .safetensors, .ckpt, .bin files or HuggingFace folders."
            )

        if torch.cuda.is_available():
            self._device = "cuda"
            dtype = torch.float16
            print("   CUDA available: True")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = "cpu"
            dtype = torch.float32
            print("   CUDA available: False")
            print("   WARNING: Running on CPU. Install CUDA-enabled PyTorch for GPU acceleration.")

        print(f"Loading model: {model_path.name}")
        print(f"   Device: {self._device} | Dtype: {dtype}")
        if model_path.is_file():
            print(f"   Size  : {round(model_path.stat().st_size / 1e9, 2)} GB")

        pipe_cls = self._detect_pipeline_class(model_path)
        is_sdxl = "XL" in pipe_cls.__name__
        is_single_file = model_path.suffix.lower() in (".safetensors", ".ckpt", ".bin")

        name_lower = model_path.name.lower()
        is_vpred = any(kw in name_lower for kw in ["vpred", "v-pred", "v_pred", "v prediction"])

        load_kwargs: dict = {"torch_dtype": dtype}
        if not is_sdxl:
            load_kwargs["safety_checker"] = None
            if is_vpred and is_single_file:
                print("   [SETUP] V-Prediction detected. Fetching SD 2.1 config and applying upcast_attention.")
                load_kwargs["config"] = "stabilityai/stable-diffusion-2-1"
                load_kwargs["upcast_attention"] = True
        else:
            if is_single_file:
                # Use local SDXL config if available to prevent diffusers NoneType guessing bug
                local_sdxl_yaml = BASE_DIR / "sd_xl_base.yaml"
                if local_sdxl_yaml.exists():
                    load_kwargs["original_config_file"] = str(local_sdxl_yaml)
                    print("   [SETUP] Using local sd_xl_base.yaml for SDXL")

        try:
            if is_single_file:
                self.pipe = pipe_cls.from_single_file(str(model_path), **load_kwargs)
            else:
                load_kwargs["use_safetensors"] = True
                self.pipe = pipe_cls.from_pretrained(str(model_path), **load_kwargs)
        except Exception as exc:
            print(f"   First load attempt failed: {exc}")
            print("   Retrying with fallback parameters...")
            load_kwargs.pop("safety_checker", None)
            
            # Diffusers bug fallback
            if is_single_file and is_sdxl and "NoneType" in str(exc):
                print("   [FIX] Supplying SDXL base config to bypass diffusers NoneType bug.")
                load_kwargs.pop("original_config_file", None)
                load_kwargs["config"] = "stabilityai/stable-diffusion-xl-base-1.0"

            if is_single_file:
                self.pipe = pipe_cls.from_single_file(str(model_path), **load_kwargs)
            else:
                self.pipe = pipe_cls.from_pretrained(str(model_path), **load_kwargs)

        self.pipe = self.pipe.to(self._device)
        self._apply_static_performance_optimizations(model_path.name)

        self._active_loras = []
        self.is_loaded = True
        self._model_info = {
            "name": model_path.name,
            "path": str(model_path),
            "device": self._device,
            "dtype": str(dtype),
            "type": pipe_cls.__name__,
            "is_sdxl": is_sdxl,
            "active_loras": [],
            "metadata": self._read_sidecar_metadata(model_path),
        }
        print(f"Base model ready: {model_path.name}")

    def _apply_static_performance_optimizations(self, model_name: str = "") -> None:
        if self.pipe is None:
            return

        try:
            import torch
            if self._device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
                try:
                    if hasattr(self.pipe, "unet") and self.pipe.unet is not None:
                        self.pipe.unet.to(memory_format=torch.channels_last)
                    if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
                        self.pipe.vae.to(memory_format=torch.channels_last)
                    print("   [PERF] channels_last enabled")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            from diffusers import DPMSolverMultistepScheduler
            
            scheduler_kwargs = {"use_karras_sigmas": True}
            
            # Check for v-prediction model
            name_lower = model_name.lower()
            is_vpred = any(kw in name_lower for kw in ["vpred", "v-pred", "v_pred", "v prediction"])
            if is_vpred:
                print(f"   [SETUP] Detected v-prediction model: {model_name}")
                scheduler_kwargs["prediction_type"] = "v_prediction"
                scheduler_kwargs["timestep_spacing"] = "trailing"
                
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                **scheduler_kwargs
            )
            print("   [PERF] Scheduler: DPMSolverMultistepScheduler + Karras")
        except Exception as exc:
            print(f"   [WARN] Could not switch scheduler: {exc}")

        if self._device == "cuda" and getattr(self.settings, "enable_xformers", False):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("   [PERF] xFormers enabled")
            except Exception:
                print("   [PERF] xFormers not available")

    def _apply_runtime_performance_mode(self, performance_mode: str, vram_mode: str) -> None:
        if self.pipe is None:
            return

        vram_mode = (vram_mode or "balanced").lower()

        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        try:
            if vram_mode == "max_speed":
                try:
                    self.pipe.disable_attention_slicing()
                    print("   [VRAM] max_speed: attention slicing disabled")
                except Exception:
                    pass
            elif vram_mode == "balanced":
                try:
                    self.pipe.enable_attention_slicing("auto")
                    print("   [VRAM] balanced: attention slicing auto")
                except Exception:
                    pass
            elif vram_mode == "low_vram":
                try:
                    self.pipe.enable_attention_slicing("max")
                    self.pipe.enable_vae_slicing()
                    self.pipe.enable_vae_tiling()
                    print("   [VRAM] low_vram: slicing/tiling enabled")
                except Exception:
                    pass
            elif vram_mode == "ultra_low_vram":
                try:
                    self.pipe.enable_attention_slicing("max")
                    self.pipe.enable_vae_slicing()
                    self.pipe.enable_vae_tiling()
                    self.pipe.enable_sequential_cpu_offload()
                    print("   [VRAM] ultra_low_vram: Sequential CPU offload enabled (best for low RAM + VRAM)")
                except Exception as exc:
                    print(f"   [WARN] ultra_low_vram failed: {exc}")
        except Exception as exc:
            print(f"   [WARN] Runtime VRAM mode failed: {exc}")

    def _check_ram_limit(self, ram_limit_gb: float) -> None:
        if not ram_limit_gb or ram_limit_gb <= 0:
            return
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < ram_limit_gb:
                raise RuntimeError(
                    f"Not enough free RAM. Available: {available_gb:.1f} GB, "
                    f"required: {ram_limit_gb:.1f} GB."
                )
            print(f"   [RAM] Available: {available_gb:.1f} GB | Required: {ram_limit_gb:.1f} GB")
        except ImportError:
            print("   [WARN] psutil not installed, RAM limit check skipped.")
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"   [WARN] RAM check failed: {exc}")

    def _apply_loras_sync(self, lora_names: list, scales: list) -> None:
        if not self.pipe:
            raise RuntimeError("Base model not loaded.")

        self._clear_loras_sync()

        if not lora_names:
            return

        scales = scales or [1.0] * len(lora_names)
        loaded = []

        for i, name in enumerate(lora_names):
            lora_path = self._find_lora(name)
            if not lora_path:
                print(f"   LoRA not found: {name}")
                continue

            scale = scales[i] if i < len(scales) else 1.0
            print(f"   Loading LoRA: {lora_path.name} | scale={scale}")

            try:
                adapter_name = lora_path.stem.replace(" ", "_")
                self.pipe.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                    adapter_name=adapter_name,
                )
                loaded.append({"name": name, "file": lora_path.name, "scale": scale, "adapter": adapter_name})
            except Exception as exc:
                print(f"   Failed to load LoRA {name}: {exc}")

        if loaded:
            adapter_names = [lo["adapter"] for lo in loaded]
            adapter_scales = [lo["scale"] for lo in loaded]
            try:
                self.pipe.set_adapters(adapter_names, adapter_weights=adapter_scales)
            except Exception as exc:
                print(f"   set_adapters failed: {exc}")
                print("   Trying fuse_lora instead...")
                try:
                    self.pipe.fuse_lora(lora_scale=adapter_scales[0] if adapter_scales else 1.0)
                except Exception as fuse_exc:
                    print(f"   fuse_lora failed: {fuse_exc}")

        self._active_loras = [lo["name"] for lo in loaded]
        if self._model_info:
            self._model_info["active_loras"] = self._active_loras
        print(f"   Active LoRAs: {self._active_loras}")

    def _clear_loras_sync(self) -> None:
        if not self.pipe:
            return
        try:
            self.pipe.unload_lora_weights()
        except Exception:
            pass
        self._active_loras = []
        if self._model_info:
            self._model_info["active_loras"] = []
        print("   LoRAs cleared")

    def _find_model(self, model_name: str = "") -> Optional[Path]:
        search_dirs = [BASE_DIR / "models", BASE_DIR / "checkpoints"]
        candidates: list[Path] = []

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for item in search_dir.iterdir():
                if item.name.startswith(".") or item.name.lower().endswith(".md"):
                    continue
                if item.is_dir() or item.suffix.lower() in (".safetensors", ".ckpt", ".bin"):
                    candidates.append(item)

        if not candidates:
            return None

        if model_name:
            for candidate in candidates:
                if candidate.name == model_name or candidate.stem == model_name:
                    return candidate
            model_name_lower = model_name.lower()
            for candidate in candidates:
                if model_name_lower in candidate.name.lower():
                    return candidate

        for candidate in candidates:
            if candidate.is_dir() and (candidate / "model_index.json").exists():
                return candidate
        for candidate in candidates:
            if candidate.suffix.lower() == ".safetensors":
                return candidate
        for candidate in candidates:
            if candidate.suffix.lower() == ".ckpt":
                return candidate
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def _find_lora(self, name: str) -> Optional[Path]:
        loras_dir = BASE_DIR / "loras"
        if not loras_dir.exists():
            return None
        for item in loras_dir.glob("*.safetensors"):
            if item.name == name or item.stem == name:
                return item
        name_lower = name.lower()
        for item in loras_dir.glob("*.safetensors"):
            if name_lower in item.name.lower():
                return item
        return None

    def _read_sidecar_metadata(self, item: Path) -> dict:
        meta_file = item.with_suffix(".json")
        if not meta_file.exists():
            return {}
        try:
            with open(meta_file, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _detect_pipeline_class(self, model_path: Path):
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        path_str = str(model_path).lower()
        sdxl_keywords = ["xl", "sdxl", "pony", "autismmix", "illustrious", "animagine", "noobai", "waiarceusx"]

        if any(keyword in path_str for keyword in sdxl_keywords):
            print(f"   Detected: SDXL pipeline | keyword match in '{model_path.name}'")
            return StableDiffusionXLPipeline

        if model_path.is_dir():
            index_file = model_path / "model_index.json"
            if index_file.exists():
                with open(index_file, encoding="utf-8") as file:
                    data = json.load(file)
                if "XL" in data.get("_class_name", ""):
                    print("   Detected: SDXL pipeline | from model_index.json")
                    return StableDiffusionXLPipeline

        print("   Detected: Stable Diffusion 1.x/2.x pipeline")
        return StableDiffusionPipeline

    def _generate_sync(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        num_images: int,
        lora_names: list,
        lora_scales: list,
        clip_skip: int,
        performance_mode: str,
        vram_mode: str,
        ram_limit_gb: float,
        progress_callback: Optional[Callable],
    ) -> list:
        import torch

        if self.pipe is None:
            raise RuntimeError("Base model not loaded.")

        start_time = time.time()
        performance_mode = (performance_mode or "fast").lower()
        vram_mode = (vram_mode or "balanced").lower()

        self._check_ram_limit(ram_limit_gb)
        self._apply_runtime_performance_mode(performance_mode, vram_mode)

        if lora_names:
            self._apply_loras_sync(lora_names, lora_scales)

        if performance_mode == "fast":
            num_inference_steps = min(num_inference_steps, 14)
            guidance_scale = min(guidance_scale, 6.5)
            width = min(width, 768)
            height = min(height, 768)
        elif performance_mode == "turbo":
            num_inference_steps = min(num_inference_steps, 14)
            guidance_scale = min(guidance_scale, 6.5)
            width = min(width, 768)
            height = min(height, 768)
            if not getattr(self.pipe, "_is_compiled", False):
                try:
                    import torch
                    print("   [PERF] Compiling UNet for turbo mode (this may take a while on first run)...")
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                    self.pipe._is_compiled = True
                    print("   [PERF] Compilation initiated.")
                except Exception as exc:
                    print(f"   [WARN] Turbo compilation failed: {exc}")
        elif performance_mode == "balanced":
            num_inference_steps = min(num_inference_steps, 22)
            guidance_scale = min(guidance_scale, 8.0)
        elif performance_mode == "quality":
            pass
        else:
            performance_mode = "fast"
            num_inference_steps = min(num_inference_steps, 14)
            guidance_scale = min(guidance_scale, 6.5)
            width = min(width, 768)
            height = min(height, 768)

        width = int(width)
        height = int(height)
        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)

        actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=self._device).manual_seed(actual_seed)

        step_counter = [0]

        def on_step(pipe: object, step: int, timestep: int, callback_kwargs: dict) -> dict:
            step_counter[0] += 1
            if progress_callback:
                progress_callback(step_counter[0], num_inference_steps)
            return callback_kwargs

        is_sdxl = bool(self._model_info.get("is_sdxl"))

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "generator": generator,
            "callback_on_step_end": on_step,
        }

        if not is_sdxl and clip_skip > 1:
            call_kwargs["clip_skip"] = clip_skip

        print(
            f"   [GEN] mode={performance_mode} | vram={vram_mode} | "
            f"{width}x{height} | steps={num_inference_steps} | cfg={guidance_scale} | seed={actual_seed}"
        )

        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        with torch.inference_mode():
            if self._device == "cuda":
                with torch.autocast("cuda"):
                    result = self.pipe(**call_kwargs)  # type: ignore[operator]
            else:
                result = self.pipe(**call_kwargs)  # type: ignore[operator]

        elapsed = round(time.time() - start_time, 2)
        self._last_generation_info = {
            "elapsed_seconds": elapsed,
            "device": self._device,
            "model": self._model_info.get("name", ""),
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": actual_seed,
            "performance_mode": performance_mode,
            "vram_mode": vram_mode,
            "num_images": num_images,
            "active_loras": self._active_loras,
        }
        print(f"   [GEN] done in {elapsed}s")

        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return result.images

    def get_info(self) -> dict:
        info = dict(self._model_info)
        info["last_generation"] = self._last_generation_info
        return info

    def get_last_generation_info(self) -> dict:
        return self._last_generation_info

    def unload(self) -> None:
        if self.pipe:
            try:
                self._clear_loras_sync()
            except Exception:
                pass
            del self.pipe
            self.pipe = None

        self.is_loaded = False
        self._active_loras = []
        self._model_info = {}
        self._last_generation_info = {}

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        print("Pipeline unloaded")
