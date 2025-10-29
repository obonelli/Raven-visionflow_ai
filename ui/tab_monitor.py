import time
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)
from PyQt5.QtCore import Qt
from ui.helpers import cv2_to_qpixmap
from watcher.monitor_thread import MonitorThread


class MonitorTab(QWidget):
    def __init__(self, training_tab=None):
        super().__init__()
        self.training_tab = training_tab

        self.freeze_base_preview = True
        self._base_preview_set = False

        layout = QVBoxLayout()

        # === Vistas ===
        self.label_frame = QLabel("Vista de vigilancia")
        self.label_diff = QLabel("Mapa de diferencias")

        for lbl in (self.label_frame, self.label_diff):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(540, 360)
            lbl.setStyleSheet(
                """
                background-color: #111;
                border: 2px solid #333;
                border-radius: 6px;
                font-size: 14px;
                """
            )

        # === Logs ===
        self.log_actions = QTextEdit()
        self.log_actions.setReadOnly(True)
        self.log_actions.setStyleSheet(
            "background-color:#111; border:2px solid #333; border-radius:6px; color:#0f0; font-size:13px; padding:6px;"
        )
        self.log_monitor = QTextEdit()
        self.log_monitor.setReadOnly(True)
        self.log_monitor.setStyleSheet(
            "background-color:#111; border:2px solid #333; border-radius:6px; color:#0af; font-size:13px; padding:6px;"
        )

        # === Botones ===
        self.button_start = QPushButton("Iniciar monitoreo")
        self.button_stop = QPushButton("Detener")
        self.button_stop.setEnabled(False)

        btn_style = """
            QPushButton {
                background-color: #222;
                border: 2px solid #666;
                border-radius: 6px;
                padding: 8px 14px;
                color: white;
                font-weight: bold;
                font-size: 13px;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
                border: 2px solid #999;
            }
            QPushButton:disabled {
                background-color: #181818;
                color: #555;
                border: 2px solid #333;
            }
        """
        for b in (self.button_start, self.button_stop):
            b.setStyleSheet(btn_style)
            b.setCursor(Qt.PointingHandCursor)

        # === Layout ===
        grid = QGridLayout()
        grid.setSpacing(15)
        grid.setContentsMargins(20, 20, 20, 20)
        grid.addWidget(self.label_frame, 0, 0)
        grid.addWidget(self.label_diff, 0, 1)
        grid.addWidget(self.log_actions, 1, 0)
        grid.addWidget(self.log_monitor, 1, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.button_start)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.button_stop)
        btn_layout.addStretch(1)

        layout.addLayout(grid)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # === Thread ===
        self.monitor_thread = MonitorThread()
        self.monitor_thread.frame_signal.connect(self.update_frames)
        self.monitor_thread.log_action_signal.connect(self.add_action_log)
        self.monitor_thread.log_monitor_signal.connect(self.add_monitor_log)
        self.monitor_thread.log_monitor_signal.connect(
            lambda m: print(f"[UI DEBUG] monitor_signal => {m}", flush=True)
        )

        self.button_start.clicked.connect(self.start_monitoring)
        self.button_stop.clicked.connect(self.stop_monitoring)

    def update_frames(self, frame, diff):
        if self.freeze_base_preview:
            if not self._base_preview_set:
                self.label_frame.setPixmap(cv2_to_qpixmap(self.label_frame, frame))
                self._base_preview_set = True
        else:
            self.label_frame.setPixmap(cv2_to_qpixmap(self.label_frame, frame))
        self.label_diff.setPixmap(cv2_to_qpixmap(self.label_diff, diff))

        # Manda el frame actual al tab de entrenamiento (para guardar)
        if self.training_tab:
            self.training_tab.update_frame(frame)

    def add_action_log(self, text):
        self.log_actions.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def add_monitor_log(self, text):
        self.log_monitor.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def start_monitoring(self):
        self.freeze_base_preview = True
        self._base_preview_set = False
        self.label_frame.setText("Vista de vigilancia")
        self.label_diff.setText("Mapa de diferencias")
        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.add_action_log("Monitoreo iniciado.")
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.button_start.setEnabled(True)
        self.button_stop.setEnabled(False)
        self.add_action_log("Deteniendo monitoreo...")
        self.monitor_thread.stop()
        self._base_preview_set = False
