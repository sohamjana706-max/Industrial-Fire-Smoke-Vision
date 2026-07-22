import os
import shutil
import glob

print("[SYSTEM]: Inspecting Kaggle dataset structure...")

# Search for all image files recursively inside dataset directory
all_images = glob.glob("dataset/**/*.jpg", recursive=True) + \
             glob.glob("dataset/**/*.png", recursive=True) + \
             glob.glob("dataset/**/*.jpeg", recursive=True)

print(f"[SYSTEM]: Found {len(all_images)} total image files.")

if not all_images:
    print("[ERROR]: No image files found in dataset folder!")
    exit(1)

# Destination paths
train_img_dir = "dataset_yolo/train/images"
train_lbl_dir = "dataset_yolo/train/labels"
val_img_dir = "dataset_yolo/valid/images"
val_lbl_dir = "dataset_yolo/valid/labels"

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Split 80% Train / 20% Val
split_idx = int(len(all_images) * 0.8)
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

def copy_files(img_list, dest_img_dir, dest_lbl_dir):
    for img_path in img_list:
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        
        # Copy Image
        shutil.copy(img_path, os.path.join(dest_img_dir, base_name))
        
        # Look for matching .txt label file
        img_dir = os.path.dirname(img_path)
        lbl_path = os.path.join(img_dir, name_no_ext + ".txt")
        if not os.path.exists(lbl_path):
            # Check parallel labels directory if present
            lbl_path = img_path.replace("images", "labels").replace(os.path.splitext(img_path)[1], ".txt")

        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(dest_lbl_dir, name_no_ext + ".txt"))

print("[SYSTEM]: Reorganizing into YOLO structure...")
copy_files(train_imgs, train_img_dir, train_lbl_dir)
copy_files(val_imgs, val_img_dir, val_lbl_dir)

# Create / Overwrite fire_smoke_data.yaml
abs_dataset_path = os.path.abspath("dataset_yolo")
yaml_content = f"""path: {abs_dataset_path}
train: train/images
val: valid/images

names:
  0: fire
  1: smoke
"""

with open("fire_smoke_data.yaml", "w") as f:
    f.write(yaml_content)

print("[SYSTEM]: Dataset successfully structured in ./dataset_yolo and fire_smoke_data.yaml updated!")
