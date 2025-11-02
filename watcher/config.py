import os

# ===== Parámetros generales =====
CHECK_INTERVAL_SEC = 0.5
CHANGE_SSIM_THRESHOLD = 0.95
REPEAT_NOTIFY_EVERY = 5.0
REQUIRED_CHANGE_FRAMES = 2
REQUIRED_STABLE_FRAMES = 6
SHOW_DIFF_WINDOW = True

# ===== Carpetas =====
DATASET_DIR = "dataset"
POS_DIR = os.path.join(DATASET_DIR, "positives")
NEG_DIR = os.path.join(DATASET_DIR, "negatives")

# ===== Carpetas para combo HUD =====
COMBO_DIR = os.path.join(DATASET_DIR, "combo")
COMBO_0_DIR = os.path.join(COMBO_DIR, "0")
COMBO_1_DIR = os.path.join(COMBO_DIR, "1")
COMBO_2_DIR = os.path.join(COMBO_DIR, "2")

for d in [
    DATASET_DIR,
    POS_DIR,
    NEG_DIR,
    COMBO_DIR,
    COMBO_0_DIR,
    COMBO_1_DIR,
    COMBO_2_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ===== CLIP =====
CLIP_DECISION_MARGIN = 0.05
CLIP_MIN_CONFIDENCE = 0.22

# ===== Parámetros de detección por diferencia =====
DIFF_MEAN_THRESHOLD = 5
MIN_CHANGE_AREA = 0.0005

# ===== Sensibilidad avanzada =====
PIXEL_DIFF_THRESHOLD = 5
LOCAL_SPIKE_THRESHOLD = 25
ADAPTIVE_AREA_FACTOR = 6
ADAPTIVE_BRIGHTNESS_FACTOR = 1.5
