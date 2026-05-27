from diffusers import StableDiffusion3Pipeline
import torch

try:
    print("Testing StableDiffusion3Pipeline...")
    pipe = StableDiffusion3Pipeline.from_single_file(
        "e:/AI/models/pornmasterAnima_preview3V1.safetensors",
        torch_dtype=torch.float16,
        text_encoder_3=None # often omitted in pruned models
    )
    print("Loaded SD3!")
except Exception as e:
    print("SD3 Error:", e)
