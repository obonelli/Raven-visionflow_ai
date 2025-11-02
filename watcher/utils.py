import os
import cv2
import pytesseract
import numpy as np
from plyer import notification
from skimage.metrics import structural_similarity as ssim
from watcher import capture, config


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

    if not text:
        print("[OCR] Sin texto detectado.")
    else:
        print(f"[OCR] Texto detectado: {text}")

    return text


def notify(msg):
    notification.notify(title="IA Visual", message=msg, timeout=4)


# ==========================================================
# 🔹 1. Guardar muestras del HUD de combos
# ==========================================================
def save_combo_sample(frame_bgr, label: int):
    """
    Guarda una captura del HUD de combos en la carpeta del label.
    label: 0, 1, 2 o 3
    """
    label = int(label)
    if label == 0:
        out_dir = config.COMBO_0_DIR
    elif label == 1:
        out_dir = config.COMBO_1_DIR
    elif label == 2:
        out_dir = config.COMBO_2_DIR
    else:
        out_dir = config.COMBO_3_DIR

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"combo_{label}_{len(os.listdir(out_dir))}.png")
    cv2.imwrite(filename, frame_bgr)
    print(f"[COMBO] Sample guardado en: {filename}")


# ==========================================================
# 🔹 2. Cargar las plantillas guardadas
# ==========================================================
def load_combo_templates():
    """
    Devuelve un diccionario:
    {
        0: [img0, img0b, ...],
        1: [img1, ...],
        2: [...],
        3: [...]
    }
    """
    combos = {
        0: [],
        1: [],
        2: [],
        3: [],
    }

    folders = {
        0: config.COMBO_0_DIR,
        1: config.COMBO_1_DIR,
        2: config.COMBO_2_DIR,
        3: config.COMBO_3_DIR,
    }

    for label, folder in folders.items():
        for path in list_images(folder):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                combos[label].append(img)

    return combos


# ==========================================================
# 🔹 3. Detectar usando el dataset (SSIM contra plantillas)
# ==========================================================
def detect_combo_points(x, y, w, h):
    """
    Detecta cuántos puntos de combo hay usando LAS MUESTRAS que hayas guardado.
    Si no hay dataset, intenta color y devuelve lo que pueda.
    """
    frame = capture.grab_region(x, y, w, h)

    # cargar plantillas
    templates = load_combo_templates()

    # si no hay nada guardado aún, cae al detector por color
    if all(len(v) == 0 for v in templates.values()):
        return _detect_combo_points_color(frame)

    # normalizar tamaño de frame para compararlo igual que las plantillas
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_label = 0
    best_score = -1.0

    for label, imgs in templates.items():
        for tmpl in imgs:
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

            # redimensionar si no coinciden
            if tmpl_gray.shape != frame_gray.shape:
                tmpl_gray = cv2.resize(
                    tmpl_gray,
                    (frame_gray.shape[1], frame_gray.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

            score, _ = ssim(frame_gray, tmpl_gray, full=True)
            if score > best_score:
                best_score = score
                best_label = label

    print(f"[COMBO] Mejor match: {best_label} (ssim={best_score:.3f})")
    return best_label


def _detect_combo_points_color(frame):
    """
    Versión antigua por color. La dejamos como fallback.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_gold = np.array([15, 80, 120])
    upper_gold = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    active = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 20:
            active += 1

    print(f"[COMBO] (fallback color) detectados: {active}")
    return min(active, 3)
