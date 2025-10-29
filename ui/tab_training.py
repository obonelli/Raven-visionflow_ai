import os
import time
import cv2
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt
from ui.helpers import cv2_to_qpixmap
from watcher import config
from watcher.brain.model import train_siamese_model


class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()

        self.last_frame = None

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        self.preview_label = QLabel("Última captura detectada")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(540, 360)
        self.preview_label.setStyleSheet(
            "background-color:#111; border:2px solid #333; border-radius:6px;"
        )
        layout.addWidget(self.preview_label)

        # === Botones ===
        btns = QHBoxLayout()
        self.btn_pos = QPushButton("Guardar como positivo ✅")
        self.btn_neg = QPushButton("Guardar como negativo ❌")
        self.btn_train = QPushButton("Reentrenar modelo")
        for b in (self.btn_pos, self.btn_neg, self.btn_train):
            b.setStyleSheet(
                "background-color:#222; border:2px solid #555; border-radius:6px; padding:8px; font-weight:bold;"
            )
            b.setCursor(Qt.PointingHandCursor)
            btns.addWidget(b)
        layout.addLayout(btns)

        # === Log ===
        self.log_train = QTextEdit()
        self.log_train.setReadOnly(True)
        self.log_train.setStyleSheet(
            "background-color:#111; border:2px solid #333; color:#0f0; font-size:13px; padding:6px;"
        )
        layout.addWidget(self.log_train)
        self.setLayout(layout)

        # === Conexiones ===
        self.btn_pos.clicked.connect(lambda: self.save_image("positive"))
        self.btn_neg.clicked.connect(lambda: self.save_image("negative"))
        self.btn_train.clicked.connect(self.train_model)

    def update_frame(self, frame):
        """Recibe el frame actual desde MonitorThread."""
        self.last_frame = frame.copy()
        self.preview_label.setPixmap(cv2_to_qpixmap(self.preview_label, frame))

    def save_image(self, label):
        if self.last_frame is None:
            self.log_train.append("⚠️ No hay imagen para guardar.")
            return
        dir_path = config.POS_DIR if label == "positive" else config.NEG_DIR
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"{int(time.time())}.png")
        cv2.imwrite(file_path, self.last_frame)
        self.log_train.append(f"[{time.strftime('%H:%M:%S')}] Guardada en {file_path}")

        # Reentrena automáticamente
        self.train_model()

    def train_model(self):
        self.log_train.append("🚀 Reentrenando modelo...")
        try:
            train_siamese_model(config.POS_DIR, config.NEG_DIR)
            self.log_train.append(
                f"[{time.strftime('%H:%M:%S')}] ✅ Reentrenamiento completado."
            )
        except Exception as e:
            self.log_train.append(f"❌ Error durante reentrenamiento: {e}")
