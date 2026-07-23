import os
import shutil
from huggingface_hub import hf_hub_download

print("[SYSTEM]: Fetching verified public Fire & Smoke model weights...")

weights_dir = "runs/detect/train/weights"
os.makedirs(weights_dir, exist_ok=True)
target_path = os.path.join(weights_dir, "best.pt")

try:
    # Public D-Fire fine-tuned model (Classes: 0 = smoke, 1 = fire)
    weight_file = hf_hub_download(
        repo_id="rabahdev/fire-smoke-yolov8n", 
        filename="best.pt"
    )
    print("[SYSTEM]: Primary HuggingFace model fetched successfully!")
except Exception as e:
    print(f"[SYSTEM]: Trying fallback model repository...")
    weight_file = hf_hub_download(
        repo_id="mfranzon/fire-smoke-yolov8",
        filename="fire_smoke_yolov8.pt"
    )

# Deploy weights across local run directories
target_dirs = [
    "runs/detect/train/weights",
    "runs/detect/train-6/weights"
]

for d in target_dirs:
    os.makedirs(d, exist_ok=True)
    shutil.copy(weight_file, os.path.join(d, "best.pt"))

print(f"[SYSTEM]: Success! Real-world weights installed to {target_path}")
