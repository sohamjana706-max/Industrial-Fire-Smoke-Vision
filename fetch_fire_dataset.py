import os
import shutil
import glob
import kagglehub

print("[SYSTEM]: Fetching verified Fire & Smoke dataset from Kaggle...")

# Download Fire and Smoke dataset via KaggleHub
path = kagglehub.dataset_download("ashutosh69/fire-and-smoke-dataset")

print(f"[SYSTEM]: Raw dataset downloaded to: {path}")

# Local target directories
target_dir = "dataset"
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

os.makedirs(f"{target_dir}/train/images", exist_ok=True)
os.makedirs(f"{target_dir}/train/labels", exist_ok=True)
os.makedirs(f"{target_dir}/valid/images", exist_ok=True)
os.makedirs(f"{target_dir}/valid/labels", exist_ok=True)

# Copy downloaded dataset files into local target directory structure
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)

# Create / Update fire_smoke_data.yaml
yaml_content = f"""path: {os.path.abspath('dataset')}
train: train/images
val: valid/images

names:
  0: fire
  1: smoke
"""

with open("fire_smoke_data.yaml", "w") as f:
    f.write(yaml_content)

print("[SYSTEM]: Dataset setup complete & fire_smoke_data.yaml configured successfully!")
