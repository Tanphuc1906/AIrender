"""
Configuration settings for AI Text-to-Image Server
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class Settings:
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Paths
    models_dir: str = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
    outputs_dir: str = os.getenv("OUTPUTS_DIR", str(BASE_DIR / "outputs"))
    frontend_dir: str = os.getenv("FRONTEND_DIR", str(BASE_DIR / "frontend"))

    # Generation defaults
    default_width: int = int(os.getenv("DEFAULT_WIDTH", "512"))
    default_height: int = int(os.getenv("DEFAULT_HEIGHT", "512"))
    default_steps: int = int(os.getenv("DEFAULT_STEPS", "20"))
    default_guidance: float = float(os.getenv("DEFAULT_GUIDANCE", "7.5"))
    max_jobs_in_memory: int = int(os.getenv("MAX_JOBS", "100"))

    # Model loading
    model_name: str = os.getenv("MODEL_NAME", "")        # Optional: specific model name in /models
    device: str = os.getenv("DEVICE", "auto")            # auto | cuda | cpu
    enable_xformers: bool = os.getenv("ENABLE_XFORMERS", "true").lower() == "true"
    enable_attention_slicing: bool = True


settings = Settings()
