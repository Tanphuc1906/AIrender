import sys
from safetensors import safe_open

try:
    with safe_open("e:/AI/models/pornmasterAnima_preview3V1.safetensors", framework="pt", device="cpu") as f:
        keys = f.keys()
        is_sdxl = any("conditioner.embedders.1" in k for k in keys) or any("text_model.encoder.layers.31" in k for k in keys)
        has_sdxl_unet = any("add_embedding" in k for k in keys)
        print("Total keys:", len(keys))
        print("Seems to be SDXL:", is_sdxl or has_sdxl_unet)
        print("Sample keys:")
        for k in list(keys)[:10]:
            print(" -", k)
except Exception as e:
    print("Error:", e)
