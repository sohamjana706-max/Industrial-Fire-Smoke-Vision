import cv2
import time
import glob
import os
from ultralytics import YOLO

# Dynamic Auto-Discovery of trained weights
weight_files = glob.glob("runs/detect/**/weights/best.pt", recursive=True)

if weight_files:
    # Pick the most recently created best.pt
    model_path = max(weight_files, key=os.path.getmtime)
    print(f"[SYSTEM]: Loading custom fine-tuned weights from: {model_path}")
else:
    model_path = "yolov8n.pt"
    print(f"[SYSTEM]: Pre-trained weights not found. Falling back to base model: {model_path}")

model = YOLO(model_path)
cap = cv2.VideoCapture(0)

print("[SYSTEM]: Industrial Safety Tactical HUD Engaged. Press 'q' to exit.")

prev_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    t_start = time.time()

    # Run inference on Apple Silicon
    results = model(frame, conf=0.25, verbose=False)[0]
    latency_ms = (time.time() - t_start) * 1000.0

    fire_count = 0
    smoke_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id].upper()

        if "FIRE" in label:
            fire_count += 1
            color = (0, 0, 255)
        else:
            smoke_count += 1
            color = (200, 200, 200)

        # Tactical Bounding Box & Corner Accents
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        corner_len = 10
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

        tag_text = f"{label} {int(conf * 100)}%"
        cv2.putText(frame, tag_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Frame Rate Metrics
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    # Top Telemetry Header
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (15, 15, 15), -1)
    status_text = f"FPS: {fps:.1f} | LATENCY: {latency_ms:.1f}ms | FIRE ZONES: {fire_count} | SMOKE ZONES: {smoke_count}"
    cv2.putText(frame, status_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

    cv2.imshow("Industrial Fire & Smoke Detection Engine", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
