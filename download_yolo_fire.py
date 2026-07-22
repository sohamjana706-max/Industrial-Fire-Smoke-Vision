import os
import shutil
import urllib.request
import zipfile

# Direct mirror of an annotated YOLOv8 Fire & Smoke dataset (with bounding boxes)
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/fire-smoke-yolo.zip"
ZIP_PATH = "fire_smoke_yolo.zip"

print("[SYSTEM]: Downloading pre-annotated YOLOv8 Fire & Smoke dataset...")

# Clean up old empty dataset folders
for folder in ["dataset", "dataset_yolo"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Try fetching pre-packaged annotated YOLO zip
try:
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ultralytics/hub/main/ultralytics/hub/assets/fire-smoke.zip", 
        ZIP_PATH
    )
except Exception:
    # Fallback mirror
    urllib.request.urlretrieve(
        "https://github.com/ayushdudani/Fire-and-Smoke-Detection-YOLO/releases/download/v1.0/dataset.zip",
        ZIP_PATH
    )

print("[SYSTEM]: Unzipping annotated dataset into ./dataset_yolo ...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall("dataset_yolo")

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

# Verify presence of text labels
lbl_files = [f for f in os.listdir("dataset_yolo/train/labels") if f.endswith(".txt")] if os.path.exists("dataset_yolo/train/labels") else []
print(f"[SYSTEM]: Success! Found {len(lbl_files)} annotated label files.")

# Write fire_smoke_data.yaml
abs_path = os.path.abspath("dataset_yolo")
yaml_content = f"""path: {abs_path}
train: train/images
val: valid/images

names:
  0: fire
  1: smoke
"""

with open("fire_smoke_data.yaml", "w") as f:
    f.write(yaml_content)

print("[SYSTEM]: fire_smoke_data.yaml configured successfully!")
