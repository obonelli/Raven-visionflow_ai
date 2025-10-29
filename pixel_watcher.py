import time
import os
import cv2
import numpy as np
import pyautogui
import mss
import pytesseract
from plyer import notification
from skimage.metrics import structural_similarity as ssim
from transformers import pipeline, CLIPProcessor, CLIPModel
import torch
from PIL import Image

# ===== Parámetros =====
CHECK_INTERVAL_SEC = 0.5
CHANGE_SSIM_THRESHOLD = 0.985
REPEAT_NOTIFY_EVERY = 5.0
REQUIRED_CHANGE_FRAMES = 2
REQUIRED_STABLE_FRAMES = 6
SHOW_DIFF_WINDOW = True

DATASET_DIR = "dataset"
POS_DIR = os.path.join(DATASET_DIR, "positives")
NEG_DIR = os.path.join(DATASET_DIR, "negatives")

CLIP_DECISION_MARGIN = 0.05
CLIP_MIN_CONFIDENCE = 0.22
# =================================

# ===== Inicialización =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)

# ===== Modelo CLIP =====
try:
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = (
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    )
    print("[CLIP] Modelo cargado correctamente.")
except Exception as e:
    print(f"[CLIP] Error al cargar modelo: {e}")
    raise SystemExit

# ===== Texto =====
try:
    ai_classifier = pipeline(
        "text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception:
    ai_classifier = None
# =================================


# --- Helpers ---
def ensure_dirs():
    for d in [POS_DIR, NEG_DIR]:
        os.makedirs(d, exist_ok=True)


def grab_fullscreen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = np.array(sct.grab(monitor))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def grab_region(x, y, w, h):
    img = pyautogui.screenshot(region=(x, y, w, h))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng+spa")
    return text.strip()


@torch.no_grad()
def embed_image(img_bgr):
    """Devuelve embedding CLIP normalizado."""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=img_pil, return_tensors="pt").to(device)
    feats = clip_model.get_image_features(**inputs)
    return torch.nn.functional.normalize(feats, p=2, dim=-1)


def load_reference_embeddings():
    pos, neg = [], []
    for root, target in [(POS_DIR, pos), (NEG_DIR, neg)]:
        for f in os.listdir(root):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, f)
                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    emb = embed_image(img)
                    target.append(emb)
                    print(f"[CLIP] Referencia cargada: {path}")
                except Exception as e:
                    print(f"[CLIP] No se pudo cargar {path}: {e}")
    return pos, neg


pos_refs, neg_refs = load_reference_embeddings()


@torch.no_grad()
def visual_predict(frame_bgr):
    """Compara frame contra referencias usando CLIP."""
    if not pos_refs and not neg_refs:
        return 0
    frame_feat = embed_image(frame_bgr)

    # calcula similitudes promedio
    sim_pos = (
        np.mean([torch.cosine_similarity(frame_feat, ref).item() for ref in pos_refs])
        if pos_refs
        else 0
    )
    sim_neg = (
        np.mean([torch.cosine_similarity(frame_feat, ref).item() for ref in neg_refs])
        if neg_refs
        else 0
    )
    print(f"[CLIP] sim_pos={sim_pos:.3f} sim_neg={sim_neg:.3f}")

    if (sim_pos - sim_neg) >= CLIP_DECISION_MARGIN and sim_pos >= CLIP_MIN_CONFIDENCE:
        return 1
    if (sim_neg - sim_pos) >= CLIP_DECISION_MARGIN and sim_neg >= CLIP_MIN_CONFIDENCE:
        return 0
    return 0


def ai_decide(text, frame):
    visual_pred = visual_predict(frame)
    visual_decision = visual_pred == 1

    text_decision = False
    reason = ""

    if text and len(text) > 3:
        keywords = ["hola", "hi", "hey", "ok", "new", "msg", "mensaje", "sí", "no"]
        if any(k.lower() in text.lower() for k in keywords):
            text_decision = True
            reason = "Palabra clave detectada"
        elif ai_classifier:
            result = ai_classifier(text[:512])[0]
            text_decision = result["label"] == "POSITIVE"
            reason = f"Clasificador de texto → {result}"

    final_decision = visual_decision or text_decision
    print(
        f"[IA Debug] Visual={visual_decision} | Text={text_decision} | {reason} | Texto='{text[:50] if text else '---'}'"
    )
    return final_decision


def notify(msg="Cambio detectado 👀"):
    notification.notify(title="IA Visual", message=msg, timeout=4)


# --- Selección de ROI ---
print("Selecciona con el mouse el área a vigilar y presiona ENTER cuando termines...")
screen = grab_fullscreen()
cv2.namedWindow("Selecciona zona", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Selecciona zona", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("Selecciona zona", 1280, 720)
r = cv2.selectROI("Selecciona zona", screen)
cv2.destroyAllWindows()

if not r or r[2] == 0 or r[3] == 0:
    print("No se seleccionó ninguna región.")
    raise SystemExit

x, y, w, h = r
base = grab_region(x, y, w, h)
base_gray = to_gray(base)

print("Monitoreando cambios en la zona seleccionada... (ESC o Ctrl+C para salir)")

consec_change = 0
consec_stable = 0
in_change_phase = False
last_notify_ts = 0.0

ensure_dirs()

if SHOW_DIFF_WINDOW:
    cv2.namedWindow("Vigilando", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("Vigilando", cv2.WND_PROP_TOPMOST, 1)

try:
    while True:
        frame = grab_region(x, y, w, h)
        gray = to_gray(frame)

        score, diff_map = ssim(base_gray, gray, full=True)
        diff_vis = (1 - diff_map) * 255
        diff_vis = diff_vis.astype(np.uint8)

        is_changed = score < CHANGE_SSIM_THRESHOLD
        if is_changed:
            consec_change += 1
            consec_stable = 0
        else:
            consec_stable += 1
            consec_change = 0

        if not in_change_phase and consec_change >= REQUIRED_CHANGE_FRAMES:
            in_change_phase = True
            last_notify_ts = 0.0

        if in_change_phase and consec_stable >= REQUIRED_STABLE_FRAMES:
            in_change_phase = False
            base = frame.copy()
            base_gray = gray.copy()
            last_notify_ts = 0

        now = time.time()
        if in_change_phase:
            if (now - last_notify_ts) >= REPEAT_NOTIFY_EVERY:
                text = extract_text(frame)
                decision = ai_decide(text, frame)
                if decision:
                    notify("💬 Nuevo mensaje detectado")
                    print("[IA] Nuevo mensaje detectado con texto:", text)
                else:
                    print("[IA] Cambio visual ignorado.")
                last_notify_ts = now

        if SHOW_DIFF_WINDOW:
            diff_bgr = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
            vis = np.hstack([frame, diff_bgr])
            cv2.imshow("Vigilando", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        time.sleep(CHECK_INTERVAL_SEC)

except KeyboardInterrupt:
    pass
finally:
    if SHOW_DIFF_WINDOW:
        cv2.destroyAllWindows()
