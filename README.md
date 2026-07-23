# Industrial Fire & Smoke Detection Suite 🛡️🔥

A high-throughput, low-latency Computer Vision safety system engineered for industrial facility monitoring using **YOLOv8** and custom tactical HUD overlays.

## Features
- **Real-Time Hazard Detection:** Identifies fire and smoke zones in real-time video feeds.
- **Tactical Telemetry HUD:** Live overlay detailing FPS, processing latency (ms), and active hazard zone counts.
- **Apple Silicon Hardware Acceleration:** Accelerated via Metal Performance Shaders (`mps`).
- **Real-World Fine-Tuned Model:** Leverages high-precision pre-trained weights for industrial fire and smoke detection.

## Tech Stack
- **Vision Engine:** PyTorch, Ultralytics YOLOv8
- **Processing:** OpenCV, NumPy
- **Hardware Acceleration:** Apple Silicon MPS Execution

## Quickstart
```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python ultralytics huggingface_hub

# Download real-world model weights
python3 download_real_weights.py

# Launch live HUD engine
python3 run_live_hud.py
