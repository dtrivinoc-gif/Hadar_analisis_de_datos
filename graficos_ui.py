"""
Pestaña Gráficos de Hadar (antes vivía dentro de main_window.py).

Es la lista de gráficos (cada uno es un ChartPanel de graficos.py) con su
botón "+ Agregar gráfico". Aquí solo se administra la lista: agregar,
quitar y mandar un gráfico al Panel. Cómo se dibuja cada gráfico, y las
opciones de cada uno, están en ChartPanel (graficos.py).

Cómo se conecta con la ventana principal (igual que Frecuencias o Panel):
    self.panel_graficos = PanelGraficos(host=self, paneles=self.chart_panels)
La lista `paneles` es la MISMA que main_window usa como self.chart_panels
(al cambiar de tema, renombrar columnas, armar el Reporte, etc.), por eso
se le entrega en vez de crear una nueva.
"""
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QMessageBox,
)

from .config import COLOR_ACCENT
from .graficos import ChartPanel

FUENTE_SERIFA = '"Georgia", "Cambria", "Times New Roman", serif'


class PanelGraficos(QWidget):
    def __init__(self, host, paneles=None, parent=None):
        super().__init__(parent)
        self.host = host
        self.paneles = paneles if paneles is not None else []

        layout = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.contenedor = QWidget()
        self.contenedor_layout = QVBoxLayout(self.contenedor)
        self.contenedor_layout.addStretch()
        self.scroll.setWidget(self.contenedor)
        layout.addWidget(self.scroll)

        fila = QHBoxLayout()
        self.btn_agregar = QPushButton("+ Agregar gráfico")
        self.btn_agregar.setObjectName("gfxAgregar")
        self.btn_agregar.setCursor(Qt.PointingHandCursor)
        self.btn_agregar.clicked.connect(self.agregar_grafico)
        fila.addWidget(self.btn_agregar)
        fila.addStretch()
        layout.addLayout(fila)

        self.agregar_grafico()
        self.aplicar_tema(host.colors)

    # ------------------------------------------------------------------
    # Tema
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        """Estilo del botón de agregar (los gráficos se re-tematizan solos,
        cada ChartPanel en su apply_theme)."""
        self.btn_agregar.setStyleSheet(f"""
            QPushButton#gfxAgregar {{
                background: transparent; border: none; border-radius: 0;
                border-bottom: 1px solid {COLOR_ACCENT}; padding: 5px 8px;
                font-family: {FUENTE_SERIFA}; font-size: 14px;
                font-weight: 700; color: {COLOR_ACCENT};
            }}
            QPushButton#gfxAgregar:hover {{
                background: transparent; border-bottom: 2px solid {COLOR_ACCENT};
            }}
        """)

    # ------------------------------------------------------------------
    # Agregar / quitar gráficos
    # ------------------------------------------------------------------
    def agregar_grafico(self):
        host = self.host
        panel = ChartPanel(host.colors)
        panel.set_render_callback(lambda p: p.render(host.filtered_df))
        panel.removeRequested.connect(self.quitar_grafico)
        panel.categoryClicked.connect(host._on_category_clicked)
        panel.panelRequested.connect(self._enviar_al_panel)
        panel.set_proveedor_df(lambda: host.filtered_df if host.filtered_df is not None else host.df)
        self.contenedor_layout.insertWidget(self.contenedor_layout.count() - 1, panel)
        self.paneles.append(panel)

        if host.df is not None:
            all_cols = host.df.columns.tolist()
            num_cols = host.df.select_dtypes(include=[np.number]).columns.tolist()
            panel.update_axis_choices(all_cols, num_cols)

        if host.filtered_df is not None:
            panel.render(host.filtered_df)

    def quitar_grafico(self, panel):
        if len(self.paneles) <= 1:
            QMessageBox.warning(self, "Atención", "Debes mantener al menos un gráfico.")
            return
        self.paneles.remove(panel)
        panel.setParent(None)
        panel.deleteLater()

    def _enviar_al_panel(self, chart_panel):
        host = self.host
        if not hasattr(host, "panel_control"):
            return
        resultado = host.panel_control.enviar_grafico(chart_panel.config_a_dict())
        mensajes = {
            "ok": "Listo: el gráfico ya aparece en la pestaña Panel.",
            "actualizado": "Listo: el gráfico del Panel se actualizó con los cambios que le hiciste.",
            "tipo": "Ese tipo de gráfico todavía no se puede mostrar en el Panel.\n"
                    "Se pueden enviar: Barras, Líneas, Histograma y Dona / Torta.",
            "lleno": "El Panel ya muestra 4 gráficos tuyos.\n"
                     "Quita uno en Panel → Personalizar → Gráficos y vuelve a intentarlo.",
        }
        QMessageBox.information(self, "Panel", mensajes.get(resultado, "Listo."))
