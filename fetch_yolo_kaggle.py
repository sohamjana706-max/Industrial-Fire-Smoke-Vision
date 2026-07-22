import os
import shutil
import glob
import kagglehub

print("[SYSTEM]: Fetching verified YOLOv8 Fire & Smoke dataset via KaggleHub...")

# Download YOLOv8-formatted dataset from Kaggle
path = kagglehub.dataset_download("o2132303/fire-and-smoke-detection-yolov8")

print(f"[SYSTEM]: Dataset downloaded to cache: {path}")

# Clean and recreate local target directory
target_dir = "dataset_yolo"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# Search for train images directory inside downloaded folder
train_img_paths = glob.glob(f"{path}/**/train/images", recursive=True) + \
                  glob.glob(f"{path}/**/images/train", recursive=True) + \
                  glob.glob(f"{path}/**/train", recursive=True)

found_root = None
for p in train_img_paths:
    if os.path.isdir(p):
        found_root = os.path.dirname(p) if "images" in p else p
        break

if not found_root:
    found_root = path

print(f"[SYSTEM]: Copying annotated dataset from {found_root} to ./{target_dir} ...")
shutil.copytree(found_root, target_dir, dirs_exist_ok=True)

# Count label files to confirm annotations exist
labels_found = glob.glob(f"{target_dir}/**/*.txt", recursive=True)
print(f"[SYSTEM]: Success! Found {len(labels_found)} annotation (.txt) files.")

# Automatically configure fire_smoke_data.yaml
abs_path = os.path.abspath(target_dir)

# Check subfolder structure
train_rel = "train/images" if os.path.exists(f"{target_dir}/train/images") else "train"
val_rel = "valid/images" if os.path.exists(f"{target_dir}/valid/images") else ("val/images" if os.path.exists(f"{target_dir}/val/images") else "valid")

yaml_content = f"""path: {abs_path}
train: {train_rel}
val: {val_rel}

names:
  0: fire
  1: smoke
"""

with open("fire_smoke_data.yaml", "w") as f:
    f.write(yaml_content)

print("[SYSTEM]: fire_smoke_data.yaml updated and ready for training!")
