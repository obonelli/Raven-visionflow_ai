import time
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QPushButton,
    QTextEdit,
    QLineEdit,
)
from watcher.brain.intent_processor import process_intent
from watcher import config


class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        # === Parámetros ===
        form = QFormLayout()
        self.spin_diff = QDoubleSpinBox()
        self.spin_diff.setRange(1, 50)
        self.spin_diff.setValue(config.DIFF_MEAN_THRESHOLD)
        self.spin_diff.setSingleStep(1)

        self.spin_area = QDoubleSpinBox()
        self.spin_area.setDecimals(5)
        self.spin_area.setRange(0.0001, 0.01)
        self.spin_area.setValue(config.MIN_CHANGE_AREA)
        self.spin_area.setSingleStep(0.0001)

        self.spin_pixel = QDoubleSpinBox()
        self.spin_pixel.setRange(1, 100)
        self.spin_pixel.setValue(config.PIXEL_DIFF_THRESHOLD)

        form.addRow("Umbral brillo medio:", self.spin_diff)
        form.addRow("Área mínima de cambio:", self.spin_area)
        form.addRow("Umbral por píxel:", self.spin_pixel)
        layout.addLayout(form)

        btn_apply = QPushButton("Aplicar cambios manuales")
        btn_apply.setStyleSheet("background-color:#333; padding:8px; font-weight:bold;")
        btn_apply.clicked.connect(self.apply_config_changes)
        layout.addWidget(btn_apply)

        # === Lenguaje natural ===
        self.input_intent = QLineEdit()
        self.input_intent.setPlaceholderText(
            "Ejemplo: 'vigila si llegan mensajes', 'hazte más sensible', etc."
        )
        self.input_intent.setStyleSheet(
            "background-color:#111; border:1px solid #333; color:white; padding:6px;"
        )
        btn_intent = QPushButton("Enviar instrucción a la IA")
        btn_intent.setStyleSheet(
            "background-color:#444; padding:8px; font-weight:bold;"
        )
        btn_intent.clicked.connect(self.send_intent_to_ai)
        layout.addWidget(self.input_intent)
        layout.addWidget(btn_intent)

        self.log_config = QTextEdit()
        self.log_config.setReadOnly(True)
        self.log_config.setStyleSheet(
            "background-color:#111; border:2px solid #333; color:#0f0; font-size:13px; padding:6px;"
        )
        layout.addWidget(self.log_config)

        self.setLayout(layout)

    def apply_config_changes(self):
        config.DIFF_MEAN_THRESHOLD = self.spin_diff.value()
        config.MIN_CHANGE_AREA = self.spin_area.value()
        config.PIXEL_DIFF_THRESHOLD = self.spin_pixel.value()
        self.log_config.append(
            f"[{time.strftime('%H:%M:%S')}] Parámetros actualizados."
        )

    def send_intent_to_ai(self):
        text = self.input_intent.text()
        result = process_intent(text)
        self.log_config.append(f"[{time.strftime('%H:%M:%S')}] {result}")
