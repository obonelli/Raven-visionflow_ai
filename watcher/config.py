import os

# ===== Parámetros generales =====
CHECK_INTERVAL_SEC = 0.5
# Si quieres que reaccione solo a cambios grandes, sube hacia 0.8-0.9.
CHANGE_SSIM_THRESHOLD = 0.95
REPEAT_NOTIFY_EVERY = 5.0
REQUIRED_CHANGE_FRAMES = 2
REQUIRED_STABLE_FRAMES = 6
SHOW_DIFF_WINDOW = True

# ===== Carpetas =====
DATASET_DIR = "dataset"
POS_DIR = os.path.join(DATASET_DIR, "positives")
NEG_DIR = os.path.join(DATASET_DIR, "negatives")

# ===== CLIP =====
CLIP_DECISION_MARGIN = 0.05
CLIP_MIN_CONFIDENCE = 0.22

# ===== Parámetros de detección por diferencia =====
# Detecta incluso variaciones muy pequeñas de brillo y área.
DIFF_MEAN_THRESHOLD = 5  # antes 8 → más sensible a diferencias leves
MIN_CHANGE_AREA = 0.0005  # antes 0.001 → detecta cambios del 0.05% del ROI

# ===== Sensibilidad avanzada =====
PIXEL_DIFF_THRESHOLD = 5  # antes 8 → cuenta más píxeles como “cambiados”
LOCAL_SPIKE_THRESHOLD = 25  # antes 40 → detecta mejor iconos o números pequeños
ADAPTIVE_AREA_FACTOR = 6  # mantiene detección adaptativa activa
ADAPTIVE_BRIGHTNESS_FACTOR = 1.5  # conserva buen equilibrio brillo/adaptación
