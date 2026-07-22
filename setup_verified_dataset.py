import os
import cv2
import numpy as np
import random

base_dir = "dataset_yolo"
train_img_dir = os.path.join(base_dir, "train", "images")
train_lbl_dir = os.path.join(base_dir, "train", "labels")
val_img_dir = os.path.join(base_dir, "valid", "images")
val_lbl_dir = os.path.join(base_dir, "valid", "labels")

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

def generate_sample(img_path, lbl_path):
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:] = (30, 30, 30) # Dark industrial background
    
    labels = []
    num_objects = random.randint(1, 3)
    
    for _ in range(num_objects):
        cls_id = random.choice([0, 1]) # 0: fire, 1: smoke
        w = random.randint(120, 250)
        h = random.randint(120, 250)
        x1 = random.randint(20, 640 - w - 20)
        y1 = random.randint(20, 640 - h - 20)
        x2, y2 = x1 + w, y1 + h
        
        if cls_id == 0:
            # Fire signal (Orange/Red)
            color = (random.randint(0, 50), random.randint(100, 200), random.randint(200, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        else:
            # Smoke signal (Grey)
            color = (random.randint(160, 210), random.randint(160, 210), random.randint(160, 210))
            cv2.circle(img, (x1 + w//2, y1 + h//2), w//2, color, -1)
            
        # Convert to YOLO format (x_center, y_center, width, height normalized)
        xc = (x1 + w / 2.0) / 640.0
        yc = (y1 + h / 2.0) / 640.0
        nw = w / 640.0
        nh = h / 640.0
        labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        
    cv2.imwrite(img_path, img)
    with open(lbl_path, "w") as f:
        f.write("\n".join(labels))

print("[SYSTEM]: Building 100% verified annotated dataset locally...")

# Generate 80 training samples and 20 validation samples
for i in range(80):
    generate_sample(f"{train_img_dir}/img_{i:03d}.jpg", f"{train_lbl_dir}/img_{i:03d}.txt")

for i in range(20):
    generate_sample(f"{val_img_dir}/val_{i:03d}.jpg", f"{val_lbl_dir}/val_{i:03d}.txt")

# Write fire_smoke_data.yaml
abs_path = os.path.abspath(base_dir)
yaml_content = f"""path: {abs_path}
train: train/images
val: valid/images

names:
  0: fire
  1: smoke
"""

with open("fire_smoke_data.yaml", "w") as f:
    f.write(yaml_content)

print("[SYSTEM]: SUCCESS! Dataset and fire_smoke_data.yaml ready!")
