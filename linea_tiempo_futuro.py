"""
linea_tiempo_futuro.py — Dibuja «lo que se viene» sobre el eje de fechas de la
Línea de Tiempo: una franja por cada ventana de fechas futuras que predice un
pronóstico que el usuario llevó al Panel (agotamiento, umbral, saturación, vida
útil). Todo nativo de PySide6/pyqtgraph; la lógica de qué fechas hay vive en el
motor (pronostico_base.ventana_fechas) y en tablero.py.

Cómo se lee lo dibujado:
  * franja sombreada = «entre estas dos fechas» (del caso más temprano al más tardío razonable)
  * línea punteada   = la fecha más probable
  * franja clara     = «en adelante»: no hay fecha tardía, solo hasta dónde se miró
  * línea sola       = «a partir de esta fecha» (o un día exacto si la franja sería muy angosta)
"""
from __future__ import annotations

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from .config import COLOR_ACCENT, COLOR_DANGER

AMBAR = "#D97706"
_DIA = 86400.0               # el eje está en segundos
_ALTURA_ETIQUETA = (0.96, 0.84, 0.72, 0.60)   # etiquetas escalonadas para que no se encimen


def _epoch(texto) -> float:
    """Misma conversión que usa la Línea de Tiempo (fecha sin zona horaria -> segundos)."""
    return pd.Timestamp(texto).value / 1e9


def _con_alfa(color, alfa):
    c = QColor(color)
    c.setAlpha(alfa)
    return c


def dibujar_ventanas(plot, ventanas):
    """Dibuja las ventanas en `plot`. Devuelve (items, (x_min, x_max) | None): los items para poder quitarlos
    después y el rango de fechas que ocupan, para que el gráfico se amplíe y se vean."""
    items, xs = [], []
    for i, v in enumerate(ventanas):
        try:
            x0 = _epoch(v["desde"])
            x1 = _epoch(v["hasta"]) if v.get("hasta") else None
            xp = _epoch(v["probable"]) if v.get("probable") else None
            xh = _epoch(v["horizonte_hasta"]) if v.get("horizonte_hasta") else None
        except (KeyError, ValueError, TypeError):
            continue                                   # una ventana mal formada no rompe las demás
        color = {"alerta": COLOR_DANGER, "atencion": AMBAR}.get(v.get("nivel"), COLOR_ACCENT)
        titulo, etiqueta = str(v.get("titulo", "")), str(v.get("etiqueta", ""))
        texto = f"{titulo}: {etiqueta}" if titulo and etiqueta and titulo.lower() != etiqueta.lower() else (titulo or etiqueta)

        if x1 is not None and x1 - x0 >= _DIA:                      # entre dos fechas
            izq, der, alfa = x0, x1, 60
        elif x1 is None and xh is not None and xh - x0 >= _DIA:     # en adelante, hasta donde se miró
            izq, der, alfa = x0, xh, 25
        else:
            izq = der = None
        if izq is not None:
            region = pg.LinearRegionItem(values=(izq, der), movable=False,
                                         brush=pg.mkBrush(_con_alfa(color, alfa)), pen=pg.mkPen(color, width=1))
            region.setZValue(5)
            plot.addItem(region, ignoreBounds=True)
            items.append(region)
        # la línea punteada: la fecha más probable; si no hay, el comienzo
        pos = xp if xp is not None else x0
        linea = pg.InfiniteLine(
            pos=pos, angle=90, movable=False, pen=pg.mkPen(color, width=2, style=Qt.DashLine),
            label=texto, labelOpts={"position": _ALTURA_ETIQUETA[i % len(_ALTURA_ETIQUETA)], "color": color,
                                    "movable": False})
        linea.setZValue(6)
        plot.addItem(linea, ignoreBounds=True)
        items.append(linea)
        xs += [x for x in (x0, x1, xp, xh) if x is not None]
    return items, ((min(xs), max(xs)) if xs else None)


def quitar_ventanas(plot, items):
    for it in items or []:
        try:
            plot.removeItem(it)
        except Exception:
            pass
