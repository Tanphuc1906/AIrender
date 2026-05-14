                                                                                                                                                from pathlib import Path
import os

base_dir = Path(__file__).parent.parent
print(f"Base Dir: {base_dir.absolute()}")

search_dirs = [
    base_dir / "models",
    base_dir / "checkpoints",
    base_dir / "loras",
]

for sd in search_dirs:
    print(f"Checking: {sd.absolute()}")
    if sd.exists():
        print(f"  - Exists! Contents: {[f.name for f in sd.iterdir()]}")
    else:
        print("  - Does NOT exist.")
