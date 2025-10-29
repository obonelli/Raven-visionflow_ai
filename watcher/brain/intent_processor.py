"""
Módulo: intent_processor.py
Interpreta instrucciones del usuario en lenguaje natural y ajusta parámetros
de configuración o comportamiento del watcher.
"""

import re
from watcher import config


def process_intent(user_text: str):
    """
    Recibe una instrucción en lenguaje natural y ajusta la configuración.
    Ejemplo:
        "vigila si llegan mensajes"
        "detecta movimiento humano"
        "ignora cambios de luz"
        "sé más sensible"
        "reduce sensibilidad"
    """
    if not user_text or not user_text.strip():
        return "⚠️ No se recibió instrucción válida."

    text = user_text.lower().strip()
    actions = []

    # --- Sensibilidad general ---
    if "más sensible" in text or "alta sensibilidad" in text:
        config.DIFF_MEAN_THRESHOLD = max(2, config.DIFF_MEAN_THRESHOLD - 2)
        config.MIN_CHANGE_AREA = max(0.0001, config.MIN_CHANGE_AREA / 2)
        actions.append("Aumentando sensibilidad visual.")
    elif "menos sensible" in text or "reduce sensibilidad" in text:
        config.DIFF_MEAN_THRESHOLD += 2
        config.MIN_CHANGE_AREA *= 2
        actions.append("Reduciendo sensibilidad visual.")

    # --- Luz y movimiento ---
    if "luz" in text and ("ignora" in text or "no detectes" in text):
        config.DIFF_MEAN_THRESHOLD = 12
        config.PIXEL_DIFF_THRESHOLD = 10
        actions.append("Ignorando cambios leves de luz.")
    elif "movimiento" in text:
        config.MIN_CHANGE_AREA = 0.001
        config.DIFF_MEAN_THRESHOLD = 5
        actions.append("Configurado para detectar movimiento.")

    # --- Mensajes / notificaciones ---
    if "mensaje" in text or "notificación" in text or "msg" in text:
        actions.append(
            "Activando modo de detección de mensajes (mayor detalle visual)."
        )
        config.DIFF_MEAN_THRESHOLD = 5
        config.PIXEL_DIFF_THRESHOLD = 5
        config.LOCAL_SPIKE_THRESHOLD = 20
        config.MIN_CHANGE_AREA = 0.0005

    # --- Reinicio ---
    if "restablece" in text or "reinicia" in text:
        reload_defaults()
        actions.append("Parámetros restablecidos a valores predeterminados.")

    if not actions:
        actions.append("No se identificó una intención clara.")

    return " | ".join(actions)


def reload_defaults():
    """Restaura los valores base de sensibilidad."""
    config.DIFF_MEAN_THRESHOLD = 5
    config.MIN_CHANGE_AREA = 0.0005
    config.PIXEL_DIFF_THRESHOLD = 5
    config.LOCAL_SPIKE_THRESHOLD = 25
    config.ADAPTIVE_AREA_FACTOR = 6
    config.ADAPTIVE_BRIGHTNESS_FACTOR = 1.5
