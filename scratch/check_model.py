import json
from pathlib import Path
from safetensors import safe_open

model_path = Path("e:/AI/models/pornmasterAnima_preview3V1.safetensors")
if not model_path.exists():
    model_path = Path("e:/AI/checkpoints/pornmasterAnima_preview3V1.safetensors")

if model_path.exists():
    f = safe_open(model_path, framework="pt", device="cpu")
    meta = f.metadata()
    print("Metadata:")
    print(json.dumps(meta, indent=2) if meta else "No metadata")
else:
    print("Model file not found")
