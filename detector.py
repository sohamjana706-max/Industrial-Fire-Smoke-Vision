# import cv2
# import time
# import requests
# from ultralytics import YOLO

# class IndustrialSafetyDetector:
#     def __init__(self, model_path="yolov8n.pt", conf_threshold=0.45):
#         # Load custom fine-tuned weights or pre-trained baseline model
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold
#         self.consecutive_hazards = 0
#         self.ALERT_TRIGGER_COUNT = 5  # Frames required to confirm hazard (prevents false alarms)

#     def trigger_industrial_alert(self, hazard_class, confidence, frame):
#         """Dispatches event to central dashboard or safety webhook."""
#         print(f"[HAZARD CRITICAL]: Confirmed {hazard_class.upper()} with confidence {confidence:.2f}!")
#         # Example: Trigger Webhook / Industrial Alert API
#         # requests.post("https://api.factory-safety.com/alerts", json={"event": hazard_class, "conf": float(confidence)})

#     def process_stream(self, stream_source=0):
#         cap = cv2.VideoCapture(stream_source)
#         if not cap.isOpened():
#             print(f"[ERROR]: Unable to open video source: {stream_source}")
#             return

#         print("[SYSTEM]: Industrial Fire & Smoke Detection Pipeline Engaged. Press 'q' to quit.")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run inference on frame
#             results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
#             hazard_detected_this_frame = False

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 class_name = self.model.names[cls_id]
#                 conf = float(box.conf[0])

#                 # Filter for Target Classes (e.g., 'fire', 'smoke')
#                 # Note: Adjust class names based on your fine-tuned model labels
#                 if class_name in ["fire", "smoke"]:
#                     hazard_detected_this_frame = True
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])

#                     # Render Visual Annotations on HUD
#                     color = (0, 0, 255) if class_name == "fire" else (128, 128, 128)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(
#                         frame,
#            import cv2
# import time
# import requests
# from ultralytics import YOLO

# class IndustrialSafetyDetector:
#     def __init__(self, model_path="yolov8n.pt", conf_threshold=0.45):
#         # Load custom fine-tuned weights or pre-trained baseline model
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold
#         self.consecutive_hazards = 0
#         self.ALERT_TRIGGER_COUNT = 5  # Frames required to confirm hazard (prevents false alarms)

#     def trigger_industrial_alert(self, hazard_class, confidence, frame):
#         """Dispatches event to central dashboard or safety webhook."""
#         print(f"[HAZARD CRITICAL]: Confirmed {hazard_class.upper()} with confidence {confidence:.2f}!")
#         # Example: Trigger Webhook / Industrial Alert API
#         # requests.post("https://api.factory-safety.com/alerts", json={"event": hazard_class, "conf": float(confidence)})

#     def process_stream(self, stream_source=0):
#         cap = cv2.VideoCapture(stream_source)
#         if not cap.isOpened():
#             print(f"[ERROR]: Unable to open video source: {stream_source}")
#             return

#         print("[SYSTEM]: Industrial Fire & Smoke Detection Pipeline Engaged. Press 'q' to quit.")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run inference on frame
#             results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
#             hazard_detected_this_frame = False

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 class_name = self.model.names[cls_id]
#                 conf = float(box.conf[0])

#                 # Filter for Target Classes (e.g., 'fire', 'smoke')
#                 # Note: Adjust class names based on your fine-tuned model labels
#                 if class_name in ["fire", "smoke"]:
#                     hazard_detected_this_frame = True
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])

#                     # Render Visual Annotations on HUD
#                     color = (0, 0, 255) if class_name == "fire" else (128, 128, 128)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(
#                         frame,
#                         f"{class_name.upper()} {conf:.2f}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         color,
#                         2
#                     )

#             # Verification Logic (Temporal Window Filter)
#             if hazard_detected_this_frame:
#                 self.consecutive_hazards += 1
#                 if self.consecutive_hazards == self.ALERT_TRIGGER_COUNT:
#                     self.trigger_industrial_alert(class_name, conf, frame)
#             else:
#                 self.consecutive_hazards = max(0, self.consecutive_hazards - 1)

#             # Display Output Stream
#             cv2.imshow("Industrial Safety Vision Suite", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detector = IndustrialSafetyDetector(model_path="yolov8n.pt", conf_threshold=0.40)
#     detector.process_stream(0)  # Pass 0 for local webcam, or RTSP stream URL for IP Camera             f"{class_name.upper()} {conf:.2f}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         color,
#                         2
#                     )

#             # Verification Logic (Temporal Window Filter)
#             if hazard_detected_this_frame:
#                 self.consecutive_hazards += 1
#                 if self.consecutive_hazards == self.ALERT_TRIGGER_COUNT:
#                     self.trigger_industrial_alert(class_name, conf, frame)
#             else:
#                 self.consecutive_hazards = max(0, self.consecutive_hazards - 1)

#             # Display Output Stream
#             cv2.imshow("Industrial Safety Vision Suite", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detector = IndustrialSafetyDetector(model_path="yolov8n.pt", conf_threshold=0.40)
#     detector.process_stream(0)  # Pass 0 for local webcam, or RTSP stream URL for IP Camera