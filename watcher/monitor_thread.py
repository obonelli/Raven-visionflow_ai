import cv2
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from PyQt5.QtCore import QThread, pyqtSignal


class MonitorThread(QThread):
    frame_signal = pyqtSignal(np.ndarray, np.ndarray)  # (frame_bgr, diff_bgr)
    log_action_signal = pyqtSignal(str)
    log_monitor_signal = pyqtSignal(str)

    def __init__(self, training_tab=None, parent=None):
        super().__init__(parent)
        self.running = False
        self._stopped_clean = True
        self.base_locked = False
        self.base_frame = None
        self.training_tab = training_tab
        self.combo_model = None
        self.last_combo = None

        # Transformación coherente con el modelo de 96x96
        self.combo_transform = transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def run(self):
        from watcher import capture, brain, config, utils
        from watcher.combo_model import load_combo_model

        self._stopped_clean = False
        self.running = True
        self.log_action_signal.emit("Selecciona el área a vigilar...")

        # === Cargar modelo ===
        try:
            self.combo_model = load_combo_model()
            self.combo_model.eval()
            self.log_action_signal.emit("🤖 Modelo de combos cargado correctamente.")
        except Exception as e:
            self.log_action_signal.emit(f"⚠️ No se pudo cargar modelo de combos: {e}")
            self.combo_model = None

        try:
            # === Captura inicial ===
            screen = capture.grab_fullscreen()
            scale = 0.7
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            small_screen = cv2.resize(
                screen,
                (int(screen.shape[1] * scale), int(screen.shape[0] * scale)),
                interpolation=interp,
            )

            # === ROI principal ===
            cv2.namedWindow("Selecciona zona", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Selecciona zona", cv2.WND_PROP_TOPMOST, 1)
            r_small = cv2.selectROI("Selecciona zona", small_screen)
            cv2.destroyAllWindows()

            if not r_small or r_small[2] == 0 or r_small[3] == 0:
                self.log_action_signal.emit("No se seleccionó región.")
                return

            r = tuple(int(v / scale) for v in r_small)
            x, y, w, h = r

            # === HUD de combos ===
            self.log_action_signal.emit("Selecciona el área de los orbes de combo...")
            cv2.namedWindow("Selecciona orbes", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Selecciona orbes", cv2.WND_PROP_TOPMOST, 1)
            roi_combo = cv2.selectROI("Selecciona orbes", screen)
            cv2.destroyAllWindows()
            x_c, y_c, w_c, h_c = [int(v) for v in roi_combo]
            self.log_action_signal.emit(
                f"🎯 HUD combos definido en ({x_c},{y_c},{w_c},{h_c})"
            )

            # === Base inicial ===
            if not self.base_locked:
                self.base_frame = capture.grab_region(x, y, w, h)
                self.base_locked = True
                self.log_action_signal.emit("📸 Imagen base capturada y bloqueada.")
            else:
                self.log_action_signal.emit("⚠️ Intento de reemplazar base bloqueado.")

            last_notify = 0.0
            in_change_phase = False
            consec_change = 0
            consec_stable = 0

            self.log_action_signal.emit("Monitoreando zona...")

            # === Loop principal ===
            while self.running:
                frame = capture.grab_region(x, y, w, h)
                base = self.base_frame

                # === Comparación base-frame ===
                diff_map = cv2.absdiff(base, frame)
                gray_diff = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)

                _, mask = cv2.threshold(
                    gray_diff, config.PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
                )
                changed_pixels = int(np.count_nonzero(mask))
                total_pixels = mask.size
                changed_ratio = changed_pixels / float(total_pixels)
                diff_mean = float(np.mean(gray_diff))
                max_diff = float(np.max(gray_diff))

                mask_colored = cv2.merge([mask, mask, mask])
                highlighted = frame.copy()
                color = np.array([0, 255, 255], dtype=np.uint8)
                mask_bool = mask_colored.astype(bool)
                highlighted = np.where(
                    mask_bool,
                    (highlighted * 0.4 + color * 0.6).astype(np.uint8),
                    highlighted,
                )
                diff_bgr = highlighted

                meets_area = changed_ratio >= config.MIN_CHANGE_AREA
                meets_energy = diff_mean >= config.DIFF_MEAN_THRESHOLD
                local_spike = max_diff >= config.LOCAL_SPIKE_THRESHOLD

                adaptive_trigger = (
                    changed_ratio
                    >= config.MIN_CHANGE_AREA / config.ADAPTIVE_AREA_FACTOR
                    and diff_mean
                    >= config.DIFF_MEAN_THRESHOLD * config.ADAPTIVE_BRIGHTNESS_FACTOR
                ) or local_spike

                changed = (meets_area and meets_energy) or adaptive_trigger

                if changed:
                    consec_change += 1
                    consec_stable = 0
                else:
                    consec_stable += 1
                    consec_change = 0

                if (not in_change_phase) and (
                    consec_change >= config.REQUIRED_CHANGE_FRAMES
                ):
                    in_change_phase = True
                    last_notify = 0.0
                    trigger_type = "pico" if local_spike else "área"
                    self.log_monitor_signal.emit(
                        f"🟡 Cambio detectado ({trigger_type}) "
                        f"(área {changed_ratio*100:.2f}% | Δ {diff_mean:.1f} | max {max_diff:.1f})"
                    )
                    self.log_action_signal.emit("Cambio visual iniciado.")

                if in_change_phase and (consec_stable >= config.REQUIRED_STABLE_FRAMES):
                    in_change_phase = False
                    self.log_action_signal.emit("Zona estabilizada (base conservada).")

                # =====================================================
                # 🔹 Predicción combo (constante, no solo en cambio)
                # =====================================================
                try:
                    hud = capture.grab_region(x_c, y_c, w_c, h_c)
                    combo_label = self.predict_combo(hud)

                    # estabilidad de lectura (filtro de rebotes)
                    if (
                        self.last_combo is None
                        or combo_label == self.last_combo
                        or np.random.rand() < 0.15
                    ):
                        self.last_combo = combo_label

                    self.log_monitor_signal.emit(
                        f"🔥 Combo detectado: {self.last_combo}"
                    )

                    # Vista previa viva del HUD
                    if self.training_tab:
                        self.training_tab.update_frame(hud, self.last_combo)

                except Exception as err:
                    self.log_monitor_signal.emit(f"⚠️ Error detectando combo: {err}")

                self.frame_signal.emit(frame, diff_bgr)
                time.sleep(config.CHECK_INTERVAL_SEC)

        except Exception as e:
            self.log_action_signal.emit(f"❌ Error en hilo: {e}")
            self.running = False
            self._stopped_clean = True

        finally:
            self._stopped_clean = True
            self.running = False
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.log_action_signal.emit("🧹 Monitoreo detenido correctamente.")

    # =====================================================
    # 🔹 Predicción con el modelo CNN (96x96 + normalización)
    # =====================================================
    def predict_combo(self, frame_bgr):
        if self.combo_model is None:
            raise RuntimeError("Modelo de combos no cargado.")

        frame_bgr = cv2.medianBlur(frame_bgr, 3)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        tensor = self.combo_transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = self.combo_model(tensor)
            pred = torch.argmax(logits, dim=1).item()

        return pred

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.wait()
        self._stopped_clean = True
