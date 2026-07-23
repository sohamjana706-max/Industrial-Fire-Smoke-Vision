import os
import requests
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks

app = FastAPI(title="Industrial Fire & Smoke Alert Gateway", version="1.0.0")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def dispatch_telegram_notification(hazard_type: str, confidence: str, image_bytes: bytes = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    caption_text = (
        f"🚨 *INDUSTRIAL SAFETY HAZARD DETECTED* 🚨\n\n"
        f"⚠️ *Hazard:* `{hazard_type}` ({confidence})\n"
        f"🕒 *Time:* `{timestamp}`\n"
        f"📍 *Location:* Zone 01 - Facility Floor\n\n"
        f"⚡ *Action Required:* Inspect CCTV stream!"
    )

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            if image_bytes:
                # SINGLE API CALL: Send photo with caption attached
                photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                files = {"photo": ("hazard_snapshot.jpg", image_bytes, "image/jpeg")}
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption_text,
                    "parse_mode": "Markdown"
                }
                requests.post(photo_url, data=data, files=files, timeout=5)
            else:
                msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                requests.post(msg_url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": caption_text,
                    "parse_mode": "Markdown"
                }, timeout=5)

            print(f"[ALERT SERVER]: Delivered 1 consolidated alert for {hazard_type}!")
        except Exception as e:
            print(f"[ALERT SERVER ERROR]: Telegram API call failed: {e}")
    else:
        print("\n[MOCK MODE]: Telegram env variables not configured.")

@app.post("/api/v1/alert")
async def receive_alert(
    background_tasks: BackgroundTasks,
    hazard_type: str = Form(...),
    confidence: str = Form(...),
    snapshot: UploadFile = File(None)
):
    image_bytes = await snapshot.read() if snapshot else None
    background_tasks.add_task(dispatch_telegram_notification, hazard_type, confidence, image_bytes)
    return {"status": "queued"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
