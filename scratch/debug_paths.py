from pathlib import Path
import os

base_dir = Path(__file__).parent.parent
print(f"Base dir: {base_dir.resolve()}")

models_dir = base_dir / "models"
ckpt_dir = base_dir / "checkpoints"
loras_dir = base_dir / "loras"

print(f"Models dir: {models_dir} exists: {models_dir.exists()}")
print(f"Checkpoints dir: {ckpt_dir} exists: {ckpt_dir.exists()}")
print(f"Loras dir: {loras_dir} exists: {loras_dir.exists()}")

if ckpt_dir.exists():
    print(f"Checkpoints content: {[f.name for f in ckpt_dir.iterdir()]}")

if loras_dir.exists():
    print(f"Loras content: {[f.name for f in loras_dir.iterdir()]}")
