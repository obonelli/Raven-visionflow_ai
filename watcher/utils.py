import os
import cv2
import pytesseract
from plyer import notification


def ensure_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def list_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng+spa")
    text = text.strip()

    # 🔹 Log temporal para ver qué lee realmente
    if not text:
        print("[OCR] Sin texto detectado.")
    else:
        print(f"[OCR] Texto detectado: {text}")

    return text


def notify(msg):
    notification.notify(title="IA Visual", message=msg, timeout=4)
