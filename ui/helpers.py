import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


def cv2_to_qpixmap(label, cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg)
    pixmap = pixmap.scaled(
        label.width(),
        label.height(),
        Qt.KeepAspectRatio,
        Qt.FastTransformation,
    )
    return pixmap
