import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtCore import Qt
from ui.tab_monitor import MonitorTab
from ui.tab_config import ConfigTab
from ui.tab_training import TrainingTab


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Watcher AI 👁️")
        self.resize(1200, 850)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # ===== Tabs =====
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab { background-color:#222; padding:6px 14px; } "
            "QTabBar::tab:selected { background-color:#333; font-weight:bold; }"
        )

        # === Crear tabs ===
        self.tab_training = TrainingTab()
        self.tab_monitor = MonitorTab(training_tab=self.tab_training)
        self.tab_config = ConfigTab()

        self.tabs.addTab(self.tab_monitor, "Vigilancia")
        self.tabs.addTab(self.tab_config, "Configuración")
        self.tabs.addTab(self.tab_training, "Entrenamiento activo")

        # === Layout principal ===
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)


def launch_ui():
    import qdarkstyle

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_ui()
