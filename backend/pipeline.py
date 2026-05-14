"""
Image Generation Pipeline — Optimized v2
Supports:
  - Stable Diffusion 1.x / 2.x / SDXL
  - .safetensors / .ckpt checkpoints (single-file)
  - HuggingFace directory format
  - LoRA (.safetensors) on top of any base model
  - Runtime model/LoRA switching without restart

Speed optimizations v2:
  - torch.compile() UNet (PyTorch >= 2.0)
  - VRAM mode caching (no re-apply every generate)
  - Real CUDA memory fraction limit (vram_limit_gb)
  - Reduced fast-mode steps (12 instead of 14)
  - CUDA TF32 / autocast / inference_mode
  - channels_last memory format
  - DPMSolverMultistep + Karras scheduler
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

# Fix CUDA memory fragmentation (PyTorch recommendation for OOM errors)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import Settings

if TYPE_CHECKING:
    import torch
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
        self._current_vram_mode: str = ""   # cache — skip re-apply if unchanged
        self._compiled: bool = False         # torch.compile flag

    # -------------------------------------------------------------------------
    # Public async wrappers
    # -------------------------------------------------------------------------

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
        vram_limit_gb: float = 0,
        ram_limit_gb: float = 0,
        progress_callback: Optional[Callable] = None,
    ) -> list:
        """
        Generate images.

        performance_mode: fast | balanced | quality
        vram_mode: max_speed | balanced | low_vram | ultra_low_vram
        vram_limit_gb: hard VRAM cap in GB (0 = no limit)
        ram_limit_gb: min free RAM guard in GB (0 = no limit)
        """
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
            vram_limit_gb,
            ram_limit_gb,
            progress_callback,
        )

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_sync(self, model_name: str = "") -> None:
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

        if self.pipe is not None:
            self.unload()

        model_path = self._find_model(model_name)
        if not model_path:
            raise FileNotFoundError(
                "No model found in /models or /checkpoints directory.\n"
                "Supported: .safetensors, .ckpt files or HuggingFace folders."
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        print(f"Loading model: {model_path.name}")
        print(f"   Device: {self._device} | Dtype: {dtype}")

        if model_path.is_file():
            print(f"   Size  : {round(model_path.stat().st_size / 1e9, 2)} GB")

        pipe_cls = self._detect_pipeline_class(model_path)
        is_sdxl = "XL" in pipe_cls.__name__
        is_single_file = model_path.suffix.lower() in (".safetensors", ".ckpt")

        load_kwargs: dict = {"torch_dtype": dtype}

        if not is_sdxl:
            load_kwargs["safety_checker"] = None

        try:
            if is_single_file:
                self.pipe = pipe_cls.from_single_file(str(model_path), **load_kwargs)
            else:
                load_kwargs["use_safetensors"] = True
                self.pipe = pipe_cls.from_pretrained(str(model_path), **load_kwargs)
        except Exception as exc:
            print(f"   First load attempt failed: {exc}")
            print("   Retrying without safety_checker...")
            load_kwargs.pop("safety_checker", None)
            if is_single_file:
                self.pipe = pipe_cls.from_single_file(str(model_path), **load_kwargs)
            else:
                self.pipe = pipe_cls.from_pretrained(str(model_path), **load_kwargs)

        self.pipe = self.pipe.to(self._device)
        self._current_vram_mode = ""   # reset cache after reload
        self._compiled = False

        self._apply_memory_optimizations()
        self._apply_static_performance_optimizations()
        self._try_compile_unet()

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
            "compiled": self._compiled,
        }

        print(f"Base model ready: {model_path.name}")

    def _apply_memory_optimizations(self) -> None:
        if self.pipe is None:
            return

        if self._device == "cuda":
            # Always enable attention slicing as baseline
            try:
                self.pipe.enable_attention_slicing(1)
                print("   [MEM] Attention slicing enabled (slice_size=1)")
            except Exception:
                pass

            # Enable VAE slicing by default — avoids OOM on high-res
            try:
                self.pipe.enable_vae_slicing()
                print("   [MEM] VAE slicing enabled")
            except Exception:
                pass

            # Enable VAE tiling by default — greatly reduces peak VRAM
            try:
                self.pipe.enable_vae_tiling()
                print("   [MEM] VAE tiling enabled")
            except Exception:
                pass

            if getattr(self.settings, "enable_xformers", False):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("   xformers enabled")
                except Exception:
                    print("   xformers not available, skipping")

    def _apply_static_performance_optimizations(self) -> None:
        """One-time optimizations after model load."""
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

                # channels_last on UNet only — VAE can cause fragmentation
                try:
                    if hasattr(self.pipe, "unet") and self.pipe.unet is not None:
                        self.pipe.unet.to(memory_format=torch.channels_last)
                    print("   [PERF] channels_last enabled (UNet only)")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
            )
            print("   [PERF] Scheduler: DPMSolverMultistepScheduler + Karras")
        except Exception as exc:
            print(f"   [WARN] Could not switch scheduler: {exc}")

    def _try_compile_unet(self) -> None:
        """
        torch.compile() the UNet for faster inference (PyTorch >= 2.0).
        Uses 'default' mode instead of 'reduce-overhead' to avoid peak VRAM
        spikes caused by CUDA graph capture on 8GB GPUs.
        Warmup happens on first generation only.
        """
        if self._device != "cuda":
            return

        try:
            import torch
            if not hasattr(torch, "compile"):
                print("   [PERF] torch.compile not available (PyTorch < 2.0), skipping")
                return

            # Check available VRAM — skip compile if < 6 GB free to avoid OOM
            mem = torch.cuda.mem_get_info(0)
            free_gb = mem[0] / (1024 ** 3)
            if free_gb < 2.0:
                print(f"   [PERF] Skipping torch.compile — only {free_gb:.1f} GB VRAM free (need >= 2 GB)")
                return

            if hasattr(self.pipe, "unet") and self.pipe.unet is not None:
                print("   [PERF] Compiling UNet with torch.compile (mode=default)...")
                self.pipe.unet = torch.compile(
                    self.pipe.unet,
                    mode="default",       # safer than reduce-overhead for 8GB GPUs
                    fullgraph=False,
                )
                self._compiled = True
                print("   [PERF] UNet compiled OK — first generation will be slower (warmup)")
        except Exception as exc:
            print(f"   [WARN] torch.compile failed: {exc}")

    # -------------------------------------------------------------------------
    # Runtime performance / RAM helpers
    # -------------------------------------------------------------------------

    def _apply_vram_limit(self, vram_limit_gb: float) -> None:
        """
        Hard VRAM limit via CUDA memory fraction.
        0 = no limit.
        """
        if not vram_limit_gb or vram_limit_gb <= 0:
            return

        try:
            import torch
            if not torch.cuda.is_available():
                return

            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            fraction = min(vram_limit_gb / total_vram, 1.0)

            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            print(f"   [VRAM] CUDA memory fraction set: {fraction:.2f} ({vram_limit_gb:.1f}/{total_vram:.1f} GB)")
        except Exception as exc:
            print(f"   [WARN] VRAM limit failed: {exc}")

    def _apply_runtime_vram_mode(self, vram_mode: str) -> None:
        """Apply VRAM mode only if changed (cached)."""
        if self.pipe is None:
            return

        vram_mode = (vram_mode or "balanced").lower()

        if vram_mode == self._current_vram_mode:
            return   # already applied, skip overhead

        print(f"   [VRAM] Switching mode: {self._current_vram_mode!r} → {vram_mode!r}")

        try:
            if vram_mode == "max_speed":
                try:
                    self.pipe.disable_attention_slicing()
                    print("   [VRAM] max_speed: attention slicing disabled")
                except Exception:
                    pass

            elif vram_mode == "balanced":
                try:
                    self.pipe.enable_attention_slicing(1)
                except Exception:
                    pass
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass
                try:
                    self.pipe.enable_vae_tiling()
                except Exception:
                    pass
                # Use sequential CPU offload in balanced mode for 8GB GPUs
                # This moves sub-models to CPU when not in use
                try:
                    self.pipe.enable_sequential_cpu_offload()
                    print("   [VRAM] balanced: sequential CPU offload + VAE slicing/tiling")
                except Exception as exc:
                    print(f"   [VRAM] balanced: CPU offload unavailable ({exc}), using attention slicing")

            elif vram_mode == "low_vram":
                try:
                    self.pipe.enable_attention_slicing("max")
                except Exception:
                    pass
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass
                try:
                    self.pipe.enable_vae_tiling()
                except Exception:
                    pass
                print("   [VRAM] low_vram: attention slicing max + VAE slicing/tiling")

            elif vram_mode == "ultra_low_vram":
                try:
                    self.pipe.enable_attention_slicing("max")
                except Exception:
                    pass
                try:
                    self.pipe.enable_vae_slicing()
                except Exception:
                    pass
                try:
                    self.pipe.enable_vae_tiling()
                except Exception:
                    pass
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("   [VRAM] ultra_low_vram: model CPU offload enabled")
                except Exception as exc:
                    print(f"   [WARN] CPU offload failed: {exc}")

        except Exception as exc:
            print(f"   [WARN] VRAM mode apply failed: {exc}")

        self._current_vram_mode = vram_mode

    def _apply_cuda_tf32(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    def _check_ram_limit(self, ram_limit_gb: float) -> None:
        """
        Soft RAM guard. Blocks generation if free RAM < ram_limit_gb.
        Use 0 to disable.
        """
        if not ram_limit_gb or ram_limit_gb <= 0:
            return

        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)

            if available_gb < ram_limit_gb:
                raise RuntimeError(
                    f"Not enough free RAM. Available: {available_gb:.1f} GB, required: {ram_limit_gb:.1f} GB."
                )
            print(f"   [RAM] Available: {available_gb:.1f} GB | Required: {ram_limit_gb:.1f} GB — OK")
        except ImportError:
            print("   [WARN] psutil not installed, RAM limit check skipped.")
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"   [WARN] RAM check failed: {exc}")

    # -------------------------------------------------------------------------
    # LoRA management
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

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
                if candidate.name == model_name:
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

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

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
        vram_limit_gb: float,
        ram_limit_gb: float,
        progress_callback: Optional[Callable],
    ) -> list:
        import torch

        if self.pipe is None:
            raise RuntimeError("Base model not loaded.")

        performance_mode = (performance_mode or "fast").lower()
        vram_mode = (vram_mode or "balanced").lower()

        # 1. RAM guard
        self._check_ram_limit(ram_limit_gb)

        # 2. Hard VRAM limit (CUDA memory fraction)
        self._apply_vram_limit(vram_limit_gb)

        # 3. CUDA TF32 (idempotent)
        self._apply_cuda_tf32()

        # 4. VRAM mode — only re-apply if changed (cached)
        self._apply_runtime_vram_mode(vram_mode)

        # 5. Ensure scheduler is fast
        try:
            from diffusers import DPMSolverMultistepScheduler
            if self.pipe.scheduler.__class__.__name__ != "DPMSolverMultistepScheduler":
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config,
                    use_karras_sigmas=True,
                )
        except Exception:
            pass

        # 6. Apply LoRAs if provided
        if lora_names:
            self._apply_loras_sync(lora_names, lora_scales)

        # 7. Performance mode presets
        if performance_mode == "fast":
            # Fast: aggressively reduce cost
            num_inference_steps = min(num_inference_steps, 12)
            guidance_scale = min(guidance_scale, 6.0)
            width = min(width, 768)
            height = min(height, 768)

        elif performance_mode == "balanced":
            num_inference_steps = min(num_inference_steps, 22)
            guidance_scale = min(guidance_scale, 8.0)

        elif performance_mode == "quality":
            pass  # keep user settings

        else:
            performance_mode = "fast"
            num_inference_steps = min(num_inference_steps, 12)
            guidance_scale = min(guidance_scale, 6.0)
            width = min(width, 768)
            height = min(height, 768)

        # 8. Safe dimensions (multiples of 8)
        width = max(256, (int(width) // 8) * 8)
        height = max(256, (int(height) // 8) * 8)

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
            f"{width}x{height} | steps={num_inference_steps} | "
            f"cfg={guidance_scale} | seed={actual_seed} | compiled={self._compiled}"
        )

        # 9. Aggressive memory cleanup before generation
        gc.collect()
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        # Log VRAM state before inference for diagnostics
        if self._device == "cuda":
            try:
                mem = torch.cuda.mem_get_info(0)
                free_gb = mem[0] / (1024 ** 3)
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                used_gb = total_gb - free_gb
                print(f"   [VRAM] Pre-inference: {used_gb:.2f}/{total_gb:.2f} GB used, {free_gb:.2f} GB free")
            except Exception:
                pass

        # 10. Inference
        with torch.inference_mode():
            if self._device == "cuda":
                with torch.autocast("cuda"):
                    result = self.pipe(**call_kwargs)  # type: ignore[operator]
            else:
                result = self.pipe(**call_kwargs)  # type: ignore[operator]

        # 11. Clear cache after
        gc.collect()
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return result.images

    # -------------------------------------------------------------------------
    # System info
    # -------------------------------------------------------------------------

    def get_system_info(self) -> dict:
        info: dict = {
            "device": self._device,
            "compiled": self._compiled,
            "vram_total_gb": None,
            "vram_used_gb": None,
            "vram_free_gb": None,
            "ram_total_gb": None,
            "ram_available_gb": None,
            "gpu_name": None,
        }

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["vram_total_gb"] = round(props.total_memory / 1e9, 2)
                mem = torch.cuda.mem_get_info(0)
                info["vram_free_gb"] = round(mem[0] / 1e9, 2)
                info["vram_used_gb"] = round((props.total_memory - mem[0]) / 1e9, 2)
        except Exception:
            pass

        try:
            import psutil
            vm = psutil.virtual_memory()
            info["ram_total_gb"] = round(vm.total / 1e9, 2)
            info["ram_available_gb"] = round(vm.available / 1e9, 2)
        except Exception:
            pass

        return info

    # -------------------------------------------------------------------------
    # Info / Cleanup
    # -------------------------------------------------------------------------

    def get_info(self) -> dict:
        return self._model_info

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
        self._current_vram_mode = ""
        self._compiled = False

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        print("Pipeline unloaded")