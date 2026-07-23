import cv2
import time
import glob
import os
import threading
import requests
from ultralytics import YOLO

# Dynamic weight discovery
weight_files = glob.glob("runs/detect/**/weights/best.pt", recursive=True)
model_path = max(weight_files, key=os.path.getmtime) if weight_files else "yolov8n.pt"
print(f"[SYSTEM]: Loading weights: {model_path}")

model = YOLO(model_path)

ALERT_API_URL = "http://127.0.0.1:8000/api/v1/alert"
COOLDOWN_SECONDS = 15  # Strict 15-second cooldown between alerts
last_alert_time = 0

def send_alert_async(hazard_type, confidence_str, frame):
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'snapshot': ('snapshot.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'hazard_type': hazard_type, 'confidence': confidence_str}
        
        resp = requests.post(ALERT_API_URL, data=data, files=files, timeout=3)
        print(f"[VISION ENGINE]: Webhook dispatched -> HTTP {resp.status_code}")
    except Exception as e:
        print(f"[VISION ENGINE ALERT ERROR]: Alert Gateway unreachable: {e}")

cap = cv2.VideoCapture(0)
prev_time = time.time()

print("[SYSTEM]: Tactical HUD + Webhook Alert Engine Active. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img = frame.shape[:2]
    frame_area = h_img * w_img

    t_start = time.time()
    results = model(frame, conf=0.35, verbose=False)[0]
    latency_ms = (time.time() - t_start) * 1000.0

    fire_count, smoke_count = 0, 0
    highest_hazard = None
    highest_conf = 0.0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)

        # Ignore false-positive full-screen background boxes (>75% of screen)
        if box_area > 0.75 * frame_area:
            continue

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = str(model.names[cls_id]).upper()

        if "SMOKE" in label:
            smoke_count += 1
            color = (200, 200, 200)
        else:
            fire_count += 1
            color = (0, 0, 255)

        if conf > highest_conf:
            highest_conf = conf
            highest_hazard = label

        # Draw HUD Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        corner_len = 12
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)

        tag_text = f"{label} {int(conf * 100)}%"
        cv2.putText(frame, tag_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Cooldown Check with Main-Thread Lock
    curr_time = time.time()
    time_since_last = curr_time - last_alert_time

    if (fire_count > 0 or smoke_count > 0) and (time_since_last > COOLDOWN_SECONDS):
        # LOCK IMMEDIATELY in main thread BEFORE starting background task
        last_alert_time = curr_time
        
        threading.Thread(
            target=send_alert_async,
            args=(highest_hazard, f"{int(highest_conf * 100)}%", frame.copy()),
            daemon=True
        ).start()

    # Telemetry Bar
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    cooldown_rem = max(0, int(COOLDOWN_SECONDS - time_since_last))
    status_text = f"FPS: {fps:.1f} | LATENCY: {latency_ms:.1f}ms | FIRE: {fire_count} | SMOKE: {smoke_count} | ALERT CD: {cooldown_rem}s"

    cv2.rectangle(frame, (0, 0), (w_img, 35), (15, 15, 15), -1)
    cv2.putText(frame, status_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

    cv2.imshow("Industrial Fire & Smoke Engine (Alert Enabled)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
