import cv2
import time
import numpy as np
from ultralytics import YOLO

class IndustrialSafetyHUD:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35):
        # Load your custom fine-tuned YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        
    def draw_tactical_hud(self, frame, fire_boxes, smoke_boxes, fps, latency_ms):
        h, w, _ = frame.shape
        
        # -------------------------------------------------------------
        # 1. TOP TELEMETRY BAR
        # -------------------------------------------------------------
        # Dark banner background at top
        cv2.rectangle(frame, (0, 0), (w, 35), (15, 15, 15), -1)
        cv2.line(frame, (0, 35), (w, 35), (50, 50, 200), 2)
        
        # Determine global risk status
        fire_count = len(fire_boxes)
        smoke_count = len(smoke_boxes)
        
        if fire_count >= 2 or (fire_count >= 1 and smoke_count >= 1):
            risk_level = "CRITICAL"
            risk_color = (0, 0, 255) # Red
        elif fire_count == 1 or smoke_count >= 1:
            risk_level = "WARNING"
            risk_color = (0, 165, 255) # Orange
        else:
            risk_level = "NOMINAL"
            risk_color = (0, 255, 0) # Green

        # Render Header Metrics
        header_text = (
            f"INDUSTRIAL FIRE & SMOKE DETECTION AI    |   "
            f"FPS: {fps:.1f}   LAT: {latency_ms:.1f}ms   "
            f"FRAME: {self.frame_count:05d}"
        )
        cv2.putText(frame, header_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        # Render Risk Indicator at top center/right
        cv2.putText(frame, f"RISK: {risk_level}", (w - 200, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color, 2)

        # -------------------------------------------------------------
        # 2. HAZARD TELEMETRY SIDEBAR PANEL (Top-Right Overlay)
        # -------------------------------------------------------------
        panel_w, panel_h = 240, 200
        px1, py1 = w - panel_w - 15, 50
        px2, py2 = w - 15, py1 + panel_h
        
        # Semi-transparent overlay box for telemetry dashboard
        overlay = frame.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Sidebar Border
        cv2.rectangle(frame, (px1, py1), (px2, py2), (60, 60, 60), 1)
        cv2.line(frame, (px1, py1), (px2, py1), risk_color, 3)

        # Max confidence extraction
        max_fire_conf = max([b['conf'] for b in fire_boxes], default=0.0) * 100
        max_smoke_conf = max([b['conf'] for b in smoke_boxes], default=0.0) * 100

        # Draw Telemetry Labels
        cv2.putText(frame, "HAZARD STATUS", (px1 + 10, py1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.putText(frame, f"FIRE CONF:", (px1 + 10, py1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{max_fire_conf:.1f}%", (px1 + 140, py1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if max_fire_conf > 0 else (180, 180, 180), 1)

        cv2.putText(frame, f"SMOKE CONF:", (px1 + 10, py1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{max_smoke_conf:.1f}%", (px1 + 140, py1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200) if max_smoke_conf > 0 else (180, 180, 180), 1)

        cv2.putText(frame, f"RISK LEVEL:", (px1 + 10, py1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{risk_level}", (px1 + 140, py1 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_color, 2)

        cv2.putText(frame, f"FIRE ZONES:", (px1 + 10, py1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{fire_count}", (px1 + 140, py1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frame, f"SMOKE ZONES:", (px1 + 10, py1 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{smoke_count}", (px1 + 140, py1 + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # -------------------------------------------------------------
        # 3. BOUNDING BOXES WITH INDUSTRIAL HIGHLIGHTS
        # -------------------------------------------------------------
        all_hazards = [("FIRE", b, (0, 0, 255)) for b in fire_boxes] + [("SMOKE", b, (200, 200, 200)) for b in smoke_boxes]

        for label, bbox, color in all_hazards:
            x1, y1, x2, y2 = bbox['coords']
            conf = bbox['conf']
            
            # Thick Red/Grey Outer Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Corner Crosshairs / Reticles for Tactical Aesthetic
            corner_len = 12
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
            
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Label Tag
            tag_text = f"{label} {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, tag_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def run_pipeline(self, video_path=0):
        cap = cv2.VideoCapture(video_path)
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            t_start = time.time()

            # Run inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            latency_ms = (time.time() - t_start) * 1000.0

            fire_boxes = []
            smoke_boxes = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id].lower()
                conf = float(box.conf[0])
                coords = map(int, box.xyxy[0])

                item = {"coords": list(coords), "conf": conf}
                
                # Check target classes (supports custom fine-tuned model or standard baseline)
                if "fire" in class_name:
                    fire_boxes.append(item)
                elif "smoke" in class_name:
                    smoke_boxes.append(item)

            # Measure real-time FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Render Tactical Overlay
            annotated_frame = self.draw_tactical_hud(frame, fire_boxes, smoke_boxes, fps, latency_ms)

            cv2.imshow("Industrial Safety Vision HUD", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = IndustrialSafetyHUD(model_path="yolov8n.pt", conf_threshold=0.30)