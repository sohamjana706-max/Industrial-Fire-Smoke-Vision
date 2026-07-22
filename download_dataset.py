import os
import urllib.request

# Create YOLO dataset directory tree
dirs = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

print("[SYSTEM]: YOLO dataset folder structure initialized in ./dataset")