import cv2
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from watcher import capture, utils


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color:#1e1e1e; color:white;")
        self.combo_roi = None  # (x, y, w, h)
        self.last_frame = None
        self.last_combo_label = None  # para mostrar combo actual

        # === Layout principal ===
        main_layout = QVBoxLayout()

        # --- Vista previa ---
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(300, 300)
        self.preview_label.setStyleSheet(
            "border:2px solid #444; background-color:#111; margin:10px;"
        )
        main_layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        # --- Botones de guardado ---
        btn_layout = QHBoxLayout()

        self.btn_combo0 = QPushButton("Guardar como 0️⃣ sin combo")
        self.btn_combo1 = QPushButton("Guardar como 1️⃣ punto")
        self.btn_combo2 = QPushButton("Guardar como 2️⃣ puntos")
        self.btn_select = QPushButton("📸 Seleccionar HUD de combo")

        for b in [self.btn_combo0, self.btn_combo1, self.btn_combo2, self.btn_select]:
            b.setStyleSheet(
                "background-color:#222; color:white; border:1px solid #444; padding:6px;"
            )

        btn_layout.addWidget(self.btn_combo0)
        btn_layout.addWidget(self.btn_combo1)
        btn_layout.addWidget(self.btn_combo2)
        btn_layout.addWidget(self.btn_select)
        main_layout.addLayout(btn_layout)

        # --- Log de acciones ---
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "background-color:#121212; color:#00FF88; border:1px solid #333;"
        )
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

        # === Conexiones ===
        self.btn_select.clicked.connect(self.select_combo_area)
        self.btn_combo0.clicked.connect(lambda: self.save_combo_state(0))
        self.btn_combo1.clicked.connect(lambda: self.save_combo_state(1))
        self.btn_combo2.clicked.connect(lambda: self.save_combo_state(2))

        # === Timer de actualización ===
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(500)  # medio segundo

        # desactivar botones hasta seleccionar el HUD
        self.set_buttons_enabled(False)

    # ==========================================================
    # 🔹 Activar/desactivar botones
    # ==========================================================
    def set_buttons_enabled(self, enabled: bool):
        self.btn_combo0.setEnabled(enabled)
        self.btn_combo1.setEnabled(enabled)
        self.btn_combo2.setEnabled(enabled)

    # ==========================================================
    # 🔹 Seleccionar HUD
    # ==========================================================
    def select_combo_area(self):
        self.log_box.append("Selecciona el área de los orbes...")
        screen = capture.grab_fullscreen()
        cv2.namedWindow("Selecciona orbes", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Selecciona orbes", cv2.WND_PROP_TOPMOST, 1)
        roi = cv2.selectROI("Selecciona orbes", screen)
        cv2.destroyAllWindows()
        if roi and roi[2] > 0 and roi[3] > 0:
            self.combo_roi = tuple(map(int, roi))
            self.log_box.append(f"✅ ROI HUD configurado: {self.combo_roi}")
            self.btn_select.setEnabled(False)
            self.set_buttons_enabled(True)
        else:
            self.log_box.append("⚠️ No se seleccionó región.")

    # ==========================================================
    # 🔹 Actualizar preview (solo desde pestaña entrenamiento)
    # ==========================================================
    def update_preview(self):
        if not self.combo_roi:
            return

        x, y, w, h = self.combo_roi
        frame = capture.grab_region(x, y, w, h)
        self.last_frame = frame
        self.show_frame(frame)

    # ==========================================================
    # 🔹 Mostrar frame (usado tanto por monitor como por preview)
    # ==========================================================
    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h_, w_, ch_ = frame_rgb.shape
        bytes_per_line = w_ * ch_
        qimg = QImage(frame_rgb.data, w_, h_, bytes_per_line, QImage.Format_RGB888)

        # dibujar texto del combo detectado si lo hay
        if self.last_combo_label is not None:
            pix = QPixmap.fromImage(qimg)
            painter = QPainter(pix)
            painter.setPen(QColor("#FFD700"))
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            painter.drawText(10, 25, f"Combo detectado: {self.last_combo_label}")
            painter.end()
        else:
            pix = QPixmap.fromImage(qimg)

        self.preview_label.setPixmap(pix.scaled(300, 300, Qt.KeepAspectRatio))

    # ==========================================================
    # 🔹 Actualizar frame desde monitor (llamado por update_frames)
    # ==========================================================
    def update_frame(self, frame, combo_label=None):
        """Recibe frame en vivo desde MonitorTab."""
        self.last_combo_label = combo_label
        self.show_frame(frame)

    # ==========================================================
    # 🔹 Guardar muestra según estado
    # ==========================================================
    def save_combo_state(self, label):
        if self.last_frame is not None:
            utils.save_combo_sample(self.last_frame, label)
            self.log_box.append(f"💾 Muestra guardada como combo {label}")
        else:
            self.log_box.append("⚠️ No hay imagen disponible.")
