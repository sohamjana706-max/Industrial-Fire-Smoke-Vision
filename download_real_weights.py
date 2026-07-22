import os
import urllib.request

print("[SYSTEM]: Fetching high-accuracy real-world Fire & Smoke weights from HuggingFace...")
weights_dir = "runs/detect/train/weights"
os.makedirs(weights_dir, exist_ok=True)
target_path = os.path.join(weights_dir, "best.pt")

url = "https://huggingface.co/arnabdhar/YOLOv8-Fire-Extraction/resolve/main/best.pt"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
    out_file.write(response.read())

print(f"[SYSTEM]: Success! Real-world weights deployed to {target_path}")
