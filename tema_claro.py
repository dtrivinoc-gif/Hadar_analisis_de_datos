"""
tema_claro.py — Colores que se adaptan a los temas claros (Blanco / Gris Claro).

Los colores de Predicción y Pronósticos se pensaron sobre fondo oscuro. Sobre
fondo claro, el ámbar pálido casi no se lee y las casillas de las tablas
quedaban negras. Aquí vive la corrección, en un solo lugar.
"""
from PySide6.QtGui import QColor

ACENTO = "#6366f1"


def es_claro(colors):
    """True si el tema activo tiene fondo claro (Blanco, Gris Claro)."""
    try:
        return QColor(colors["bg"]).lightness() > 140
    except Exception:
        return False


_OSCUROS = {"#fbbf24": "#b45309", "#34d399": "#047857", "#f87171": "#dc2626"}


def tinta(colors, color):
    """Versión legible de un color pálido (ámbar, verde, rojo) sobre fondo claro."""
    return _OSCUROS.get(color.lower(), color) if es_claro(colors) else color


def ambar(colors, base="#fbbf24"):
    """Ámbar para texto/líneas: el pálido en temas oscuros, uno oscuro en claros."""
    return "#b45309" if es_claro(colors) else base


def gris_texto(colors, base="#a1a1aa"):
    """Gris de encabezados y rótulos, con contraste suficiente en fondo claro."""
    return "#52525b" if es_claro(colors) else base


def qss_casillas(colors, selector="QTableWidget"):
    """Casillas de marcar dibujadas a mano: vacía = marco gris, marcada = índigo
    con una palomita. Sin esto, en los temas claros Qt las pinta negras."""
    claro = es_claro(colors)
    marco = "#71717a" if claro else "#a1a1aa"
    fondo = "#ffffff" if claro else "transparent"
    return (
        f"{selector}::indicator {{ width: 14px; height: 14px; border-radius: 3px; "
        f"border: 1.5px solid {marco}; background: {fondo}; }}"
        f"{selector}::indicator:unchecked:hover {{ border-color: {ACENTO}; }}"
        f"{selector}::indicator:checked {{ background: {ACENTO}; border: 1.5px solid {ACENTO}; "
        f"image: url({_palomita()}); }}"
    )


_RUTA_PALOMITA = None


def _palomita():
    """Crea una vez una palomita blanca en una carpeta temporal y devuelve su ruta."""
    global _RUTA_PALOMITA
    if _RUTA_PALOMITA is None:
        import os
        import tempfile
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPainter, QPen, QPixmap
        pm = QPixmap(28, 28)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#ffffff"), 3.4)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline([__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(7, 14.5),
                        __import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(12, 19.5),
                        __import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(21, 9)])
        p.end()
        ruta = os.path.join(tempfile.gettempdir(), "hadar_palomita.png")
        pm.save(ruta, "PNG")
        _RUTA_PALOMITA = ruta.replace("\\", "/")
    return _RUTA_PALOMITA
