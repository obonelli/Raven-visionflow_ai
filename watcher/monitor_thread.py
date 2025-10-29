import cv2
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class MonitorThread(QThread):
    frame_signal = pyqtSignal(np.ndarray, np.ndarray)  # (frame_bgr, diff_bgr)
    log_action_signal = pyqtSignal(str)
    log_monitor_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self._stopped_clean = True  # evita doble inicio
        self.base_locked = False  # bloquea reemplazo de imagen base
        self.base_frame = None

    def run(self):
        from watcher import capture, brain, config, utils

        self._stopped_clean = False
        self.running = True
        self.log_action_signal.emit("Selecciona el área a vigilar...")

        try:
            # === Captura inicial ===
            screen = capture.grab_fullscreen()

            # Escalado para selección (solo visual)
            scale = 0.7
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            small_screen = cv2.resize(
                screen,
                (int(screen.shape[1] * scale), int(screen.shape[0] * scale)),
                interpolation=interp,
            )

            # === Selección ROI ===
            cv2.namedWindow("Selecciona zona", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Selecciona zona", cv2.WND_PROP_TOPMOST, 1)
            r_small = cv2.selectROI("Selecciona zona", small_screen)
            cv2.destroyAllWindows()

            if not r_small or r_small[2] == 0 or r_small[3] == 0:
                self.log_action_signal.emit("No se seleccionó región.")
                return

            # Coordenadas reales (reescala a tamaño original)
            r = tuple(int(v / scale) for v in r_small)
            x, y, w, h = r

            # === Captura y bloqueo de base ===
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

            # === Bucle principal ===
            while self.running:
                frame = capture.grab_region(x, y, w, h)
                base = self.base_frame  # referencia fija

                # === Procesamiento interno ===
                diff_map = cv2.absdiff(base, frame)
                gray_diff = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)

                # Mapa binario de “cambio” y métricas
                _, mask = cv2.threshold(
                    gray_diff, config.PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
                )
                changed_pixels = int(np.count_nonzero(mask))
                total_pixels = mask.size
                changed_ratio = changed_pixels / float(total_pixels)
                diff_mean = float(np.mean(gray_diff))  # promedio de diferencia (0-255)
                max_diff = float(np.max(gray_diff))  # delta máximo local

                # === Visual híbrida (amarillo) ===
                mask_colored = cv2.merge([mask, mask, mask])
                highlighted = frame.copy()
                color = np.array([0, 255, 255], dtype=np.uint8)  # Amarillo
                mask_bool = mask_colored.astype(bool)
                highlighted = np.where(
                    mask_bool,
                    (highlighted * 0.4 + color * 0.6).astype(np.uint8),
                    highlighted,
                )
                diff_bgr = highlighted

                # === Lógica de detección (por área, energía y picos locales) ===
                meets_area = changed_ratio >= config.MIN_CHANGE_AREA
                meets_energy = diff_mean >= config.DIFF_MEAN_THRESHOLD
                local_spike = (
                    max_diff >= config.LOCAL_SPIKE_THRESHOLD
                )  # Detecta cambios brillantes pequeños

                # Detección adaptativa combinada
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

                # Fases de cambio
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

                # IA y OCR periódicos durante fase de cambio
                now = time.time()
                if (
                    in_change_phase
                    and (now - last_notify) >= config.REPEAT_NOTIFY_EVERY
                ):
                    text = utils.extract_text(frame)
                    decision = brain.ai_decide(frame, text)
                    if decision:
                        self.log_action_signal.emit("💬 Nuevo mensaje detectado.")
                    else:
                        self.log_monitor_signal.emit("Cambio visual ignorado.")
                    last_notify = now

                # Envía imágenes al UI
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

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.wait()
        self._stopped_clean = True
