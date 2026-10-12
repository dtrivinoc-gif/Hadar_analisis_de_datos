"""
ajustes_ui.py — Subpestaña «Ajustes» de Predicción (PySide6 nativo, sin QtWebEngine).

Se dibuja a partir de `ajustes.OPCIONES` (el texto y las opciones viven en ajustes.py, no aquí). Misma interfaz
que PanelPrediccion y PanelPalabras para que main_window la trate igual:
    aplicar_tema(colors) · notificar_cambio() · config_a_dict() · cargar_config(datos)
Lo elegido está en `self.ajustes` (un AjustesPrediccion); PanelPalabras lo copia a cada análisis al calcular.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFrame, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from .ajustes import OPCIONES, AjustesPrediccion
from .panel import estilo_tarjeta

ACENTO = "#6366f1"
FUENTE_SERIF = '"Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif'
REGLA = "rgba(127,127,127,0.30)"


class PanelAjustes(QWidget):
    cambio = Signal()      # el usuario cambió algún ajuste

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.colors = host.colors
        self.ajustes = AjustesPrediccion()
        self._combos = {}            # clave -> QComboBox
        self._cargando = False
        self._construir()
        self.aplicar_tema(self.colors)
        self.ajustes.activar()

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    def _construir(self):
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        raiz.addWidget(scroll)
        cont = QWidget()
        scroll.setWidget(cont)
        lay = QVBoxLayout(cont)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(14)

        titulo = QLabel("Ajustes de Predicción")
        titulo.setStyleSheet(f"font-size: 24px; font-family: {FUENTE_SERIF};")
        lay.addWidget(titulo)
        sub = QLabel("Decisiones que valen para todo el proyecto. Los valores recomendados ya vienen elegidos: "
                     "no necesitas cambiar nada para obtener resultados correctos. Lo que elijas aquí se anota "
                     "como supuesto en cada respuesta.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        for i, (clave, titulo_op, porque, opciones) in enumerate(OPCIONES, start=1):
            marco = QFrame()
            marco.setObjectName("seccionTufte")
            marco.setStyleSheet(f"#seccionTufte{{background: transparent; border: none; border-top: 1px solid {REGLA};}}")
            caja = QVBoxLayout(marco)
            caja.setContentsMargins(0, 16, 0, 10)
            caja.setSpacing(8)
            cab = QLabel(f"{i}   {titulo_op}")
            cab.setStyleSheet("color: #a1a1aa; font-weight: 600; font-size: 13px; letter-spacing: 1px;")
            caja.addWidget(cab)
            combo = QComboBox()
            for valor, etiqueta in opciones:
                combo.addItem(etiqueta, valor)
            combo.setMaximumWidth(380)
            combo.currentIndexChanged.connect(lambda _i, k=clave: self._cambio(k))
            self._combos[clave] = combo
            caja.addWidget(combo)
            expl = QLabel(porque)
            expl.setObjectName("muted")
            expl.setWordWrap(True)
            caja.addWidget(expl)
            lay.addWidget(marco)

        self.btn_restaurar = QPushButton("Volver a los valores recomendados")
        self.btn_restaurar.setCursor(Qt.PointingHandCursor)
        self.btn_restaurar.clicked.connect(self._restaurar)
        lay.addWidget(self.btn_restaurar, 0, Qt.AlignLeft)
        lay.addStretch(1)
        self._poner_en_widgets()

    # ------------------------------------------------------------------
    # Lectura y escritura
    # ------------------------------------------------------------------
    def _poner_en_widgets(self):
        self._cargando = True
        try:
            for clave, combo in self._combos.items():
                idx = combo.findData(getattr(self.ajustes, clave))
                combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self._cargando = False

    def _cambio(self, clave):
        if self._cargando:
            return
        valor = self._combos[clave].currentData()
        self.ajustes = AjustesPrediccion.from_dict({**self.ajustes.to_dict(), clave: valor})
        self.ajustes.activar()
        self.cambio.emit()

    def _restaurar(self):
        self.ajustes = AjustesPrediccion()
        self._poner_en_widgets()
        self.ajustes.activar()
        self.cambio.emit()

    # ------------------------------------------------------------------
    # Ciclo de vida (misma interfaz que PanelPrediccion y PanelPalabras)
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        self.colors = colors
        self.setStyleSheet(estilo_tarjeta(colors) + f" QLabel {{ font-family: {FUENTE_SERIF}; }}")
        self.btn_restaurar.setStyleSheet(
            f"QPushButton{{background: transparent; color: {ACENTO}; font-weight: 600; padding: 7px 22px; "
            f"border: 1px solid {ACENTO}; border-radius: 3px;}} "
            f"QPushButton:hover{{background: rgba(99,102,241,0.16);}}")

    def notificar_cambio(self):
        """Los ajustes no dependen de las tablas: no hay nada que rehacer."""

    def config_a_dict(self) -> dict:
        return self.ajustes.to_dict()

    def cargar_config(self, datos):
        self.ajustes = AjustesPrediccion.from_dict(datos)
        self._poner_en_widgets()
        self.ajustes.activar()
