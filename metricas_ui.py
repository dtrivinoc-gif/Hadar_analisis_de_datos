"""
Pestaña Métricas de Hadar (antes vivía dentro de main_window.py).

Es una lista de bloques, uno por columna numérica. Cada bloque muestra las
cuatro estadísticas de siempre (Media, Mediana, Moda, Desviación Std) como una
fila de números, y debajo un gráfico de caja y bigote CON los puntos de los
datos encima (dispersados un poco hacia los lados, "jitter", para que no se
tapen entre sí): la caja resume, los puntos muestran los datos de verdad.

Cómo se conecta con la ventana principal (igual que Gráficos, Frecuencias o Reporte):
    self.panel_metricas = PanelMetricas(host=self)
`host` es la ventana principal: de ahí se leen los datos (df, filtered_df), las
reglas de alarma y los indicadores. Esta pestaña nunca los modifica.

Otros módulos (Reporte, Alarma) leen la lista de bloques como
`host.metricas_bloques`; cada bloque es un dict con, al menos:
    "columna"  nombre de la columna        "plot"    el PlotWidget (el Reporte le saca la foto)
    "filtro"   filtro de filas del bloque  "cards"   {estadístico: widget}  (para marcar alarmas)
    "lbls"     {estadístico: QLabel con el valor}

Estilo: el mismo de Gráficos y Reporte -- sin recuadros, serifa (Georgia),
botones que son solo texto con una línea inferior en el color de acento, una
línea fina entre bloques, y el rojo reservado para lo que de verdad importa
(alarmas y valores atípicos).
"""
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .config import COLOR_ACCENT, COLOR_DANGER
from .alarmas import calcular_valor_actual, evaluar_regla
from .filtro_filas import aplicar_filtro, describir_filtro, DialogoFiltroFilas

FUENTE_SERIFA = '"Georgia", "Cambria", "Times New Roman", serif'

ESTADISTICOS = ("Media", "Mediana", "Moda", "Desviación Std")

# Gráfico de caja con puntos ---------------------------------------------------
CAJA_HORIZONTAL = True   # False = caja vertical (valores en el eje de la izquierda)
MAX_PUNTOS = 1500        # tope de puntos dibujados: con archivos grandes se muestrea
MAX_ATIPICOS = 300       # los atípicos se dibujan todos hasta este tope (siempre se priorizan)
_SEMILLA = 0             # misma semilla = mismo dibujo con los mismos datos (determinista)


def dibujar_caja_con_puntos(plot, datos, colors, horizontal=CAJA_HORIZONTAL):
    """Dibuja en `plot` (un pg.PlotWidget) la caja y bigote de `datos` con sus
    puntos encima. Devuelve (n, dibujados): cuántos datos hay y cuántos puntos
    se dibujaron (si hay muchísimos, se muestrea; la caja siempre usa todos).

    Caja = cuartiles Q1-Q3, raya gruesa = mediana, bigotes = hasta el último
    dato que no sea atípico (regla de Tukey: más allá de 1,5 × el rango de la
    caja). Los puntos fuera de los bigotes (atípicos) van en rojo."""
    valores = np.asarray(datos, dtype=float)
    valores = valores[np.isfinite(valores)]
    plot.clear()
    n = len(valores)
    if n == 0:
        return 0, 0

    q1, mediana, q3 = np.percentile(valores, [25, 50, 75])
    iqr = q3 - q1
    atipico = (valores < q1 - 1.5 * iqr) | (valores > q3 + 1.5 * iqr)
    normales = valores[~atipico]
    bigote_inf = float(normales.min()) if normales.size else float(q1)
    bigote_sup = float(normales.max()) if normales.size else float(q3)

    # Qué puntos se dibujan (determinista; los atípicos tienen prioridad).
    rng = np.random.default_rng(_SEMILLA)

    def muestra(indices, k):
        return indices if len(indices) <= k else np.sort(rng.choice(indices, size=k, replace=False))

    idx_atip = muestra(np.flatnonzero(atipico), MAX_ATIPICOS)
    idx_norm = muestra(np.flatnonzero(~atipico), max(0, MAX_PUNTOS - len(idx_atip)))

    def xy(v, c):   # v = posición sobre el eje de los valores, c = posición cruzada
        return (v, c) if horizontal else (c, v)

    # Puntos (debajo de la caja, para que la caja se lea encima).
    color_punto = pg.mkColor(colors["muted"])
    color_punto.setAlpha(120)
    color_atip = pg.mkColor(COLOR_DANGER)
    color_atip.setAlpha(210)
    for idx, color in ((idx_norm, color_punto), (idx_atip, color_atip)):
        if len(idx) == 0:
            continue
        salto = rng.uniform(-0.3, 0.3, size=len(idx))
        x, y = xy(valores[idx], salto)
        puntos = pg.ScatterPlotItem(x=x, y=y, size=5, pen=pg.mkPen(None), brush=pg.mkBrush(color))
        puntos.setZValue(1)
        plot.addItem(puntos)

    # Caja, bigotes y mediana.
    color_caja = pg.mkColor(colors["text"])
    mitad = 0.2

    def linea(vs, cs, ancho, color, z):
        x, y = xy(np.array(vs, dtype=float), np.array(cs, dtype=float))
        item = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color, width=ancho))
        item.setZValue(z)
        plot.addItem(item)

    linea([q1, q3, q3, q1, q1], [-mitad, -mitad, mitad, mitad, -mitad], 1.3, color_caja, 2)
    linea([bigote_inf, q1], [0, 0], 1.3, color_caja, 2)
    linea([q3, bigote_sup], [0, 0], 1.3, color_caja, 2)
    linea([mediana, mediana], [-mitad, mitad], 3, pg.mkColor(COLOR_ACCENT), 3)

    # Rango visible fijo (el gráfico no se mueve con el mouse).
    vmin, vmax = float(valores.min()), float(valores.max())
    margen = (vmax - vmin) * 0.04 or 1.0
    rango_valores = (vmin - margen, vmax + margen)
    rango_cruzado = (-0.5, 0.5)
    if horizontal:
        plot.setXRange(*rango_valores, padding=0)
        plot.setYRange(*rango_cruzado, padding=0)
    else:
        plot.setXRange(*rango_cruzado, padding=0)
        plot.setYRange(*rango_valores, padding=0)
    return n, len(idx_norm) + len(idx_atip)


def _estilizar_plot(plot, colors, horizontal=CAJA_HORIZONTAL):
    """Gráfico sin adornos: solo el eje de los valores, en gris; nada de
    cuadrícula, botones ni menú; fondo igual al de la pestaña."""
    plot.setBackground(colors["bg"])
    item = plot.getPlotItem()
    item.hideAxis("left" if horizontal else "bottom")
    item.showAxis("bottom" if horizontal else "left")
    eje = item.getAxis("bottom" if horizontal else "left")
    eje.setPen(pg.mkPen(colors["border"]))
    eje.setTextPen(pg.mkPen(colors["muted"]))
    item.showGrid(x=False, y=False)
    item.hideButtons()
    item.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)


class _Estadistico(QFrame):
    """Un número con su nombre encima. En alarma, una barra roja a la izquierda
    y el valor en rojo."""

    def __init__(self, nombre, parent=None):
        super().__init__(parent)
        self.setObjectName("metEst")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 0, 0, 0)
        lay.setSpacing(0)
        self.lbl_nombre = QLabel(nombre)
        self.lbl_nombre.setObjectName("metEstNombre")
        self.lbl_valor = QLabel("-")
        self.lbl_valor.setObjectName("metEstValor")
        lay.addWidget(self.lbl_nombre)
        lay.addWidget(self.lbl_valor)

    def set_alarma(self, activa):
        if activa:
            self.setStyleSheet(f"QFrame#metEst {{ border-left: 3px solid {COLOR_DANGER}; }}")
            self.lbl_valor.setStyleSheet(f"color: {COLOR_DANGER};")
            self.setToolTip("Alarma activa")
        else:
            self.setStyleSheet("")
            self.lbl_valor.setStyleSheet("")
            self.setToolTip("")


class PanelMetricas(QWidget):
    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.bloques = []   # cada uno: dict (ver el docstring de arriba)

        layout = QVBoxLayout(self)

        barra = QHBoxLayout()
        self.btn_agregar = QPushButton("+ Agregar métrica de otra columna")
        self.btn_agregar.setObjectName("metAccion")
        self.btn_agregar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_agregar.clicked.connect(lambda: self.agregar_bloque())
        barra.addWidget(self.btn_agregar)
        barra.addStretch()
        layout.addLayout(barra)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("metScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        contenedor = QWidget()
        contenedor.setObjectName("metContenedor")
        self.contenedor_layout = QVBoxLayout(contenedor)
        self.lbl_vacio = QLabel("Todavía no hay métricas. Agrega una con «+ Agregar métrica de otra columna».")
        self.lbl_vacio.setObjectName("metVacio")
        self.lbl_vacio.setWordWrap(True)
        self.contenedor_layout.addWidget(self.lbl_vacio)
        self.contenedor_layout.addStretch()   # empuja los bloques hacia arriba
        self.scroll.setWidget(contenedor)
        layout.addWidget(self.scroll, stretch=1)

        self.aplicar_tema(host.colors)

    # ------------------------------------------------------------------
    # Tema
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        """Estilo de la pestaña (solo widgets con nombre propio de aquí) y
        colores de cada gráfico. Los puntos se redibujan con el siguiente
        recálculo, que la ventana principal dispara al cambiar de tema."""
        c = colors
        self.setStyleSheet(f"""
            QPushButton#metAccion {{
                background: transparent; border: none; border-radius: 0;
                border-bottom: 1px solid {COLOR_ACCENT}; padding: 5px 8px;
                font-family: {FUENTE_SERIFA}; font-size: 14px;
                font-weight: 700; color: {COLOR_ACCENT};
            }}
            QPushButton#metAccion:hover {{
                background: transparent; border-bottom: 2px solid {COLOR_ACCENT};
            }}
            QScrollArea#metScroll {{ background: transparent; border: none; }}
            QScrollArea#metScroll > QWidget > QWidget {{ background: transparent; }}
            QWidget#metContenedor {{ background: transparent; }}

            QFrame#metBloque {{
                background: transparent; border: none;
                border-top: 1px solid {c['border']};
            }}
            QLabel#metTitulo {{
                background: transparent; border: none; color: {c['text']};
                font-family: {FUENTE_SERIFA}; font-size: 18px;
            }}
            QLabel#metNota, QLabel#metVacio {{
                background: transparent; border: none; color: {c['muted']};
                font-family: {FUENTE_SERIFA}; font-size: 12px; font-style: italic;
            }}
            QLabel#metVacio {{ font-size: 15px; padding: 16px 0; }}
            QPushButton#metLink, QPushButton#metQuitar {{
                background: transparent; border: none; border-radius: 0;
                padding: 2px 6px; color: {c['muted']};
                font-family: {FUENTE_SERIFA}; font-size: 12px;
            }}
            QPushButton#metLink:hover {{ color: {c['text']}; text-decoration: underline; }}
            QPushButton#metQuitar:hover {{ color: {COLOR_DANGER}; text-decoration: underline; }}

            QFrame#metEst {{ background: transparent; border: none; border-left: 3px solid transparent; }}
            QLabel#metEstNombre {{
                background: transparent; border: none; color: {c['muted']};
                font-family: {FUENTE_SERIFA}; font-size: 11px;
            }}
            QLabel#metEstValor {{
                background: transparent; border: none; color: {c['text']};
                font-family: {FUENTE_SERIFA}; font-size: 22px;
            }}
        """)
        for bloque in self.bloques:
            _estilizar_plot(bloque["plot"], c)

    # ------------------------------------------------------------------
    # Bloques
    # ------------------------------------------------------------------
    def columnas_numericas_disponibles(self):
        df = self.host.df
        if df is None:
            return []
        ya_agregadas = {b["columna"] for b in self.bloques}
        return [
            str(c) for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and str(c) not in ya_agregadas
        ]

    def _refrescar_vacio(self):
        self.lbl_vacio.setVisible(not self.bloques)

    def agregar_bloque(self, columna=None):
        host = self.host
        if host.df is None:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        if columna is None:
            disponibles = self.columnas_numericas_disponibles()
            if not disponibles:
                QMessageBox.information(
                    self, "Nada que agregar",
                    "Ya agregaste todas las columnas numéricas disponibles, o no hay ninguna."
                )
                return
            columna, ok = QInputDialog.getItem(
                self, "Elegir columna", "¿Qué columna numérica quieres analizar?", disponibles, 0, False
            )
            if not ok or not columna:
                return

        frame = QFrame()
        frame.setObjectName("metBloque")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 12, 0, 14)
        frame_layout.setSpacing(8)

        # Encabezado: columna a la izquierda, acciones discretas a la derecha.
        encabezado = QHBoxLayout()
        lbl_titulo = QLabel(columna)
        lbl_titulo.setObjectName("metTitulo")
        encabezado.addWidget(lbl_titulo)
        lbl_filtro = QLabel("")
        lbl_filtro.setObjectName("metNota")
        lbl_filtro.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        encabezado.addWidget(lbl_filtro, stretch=1)
        btn_filtro = QPushButton("Filtrar filas")
        btn_filtro.setObjectName("metLink")
        btn_filtro.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_filtro.setToolTip("Calcula esta métrica solo con ciertas filas (por ejemplo, una máquina o un producto).")
        encabezado.addWidget(btn_filtro)
        btn_quitar = QPushButton("Quitar")
        btn_quitar.setObjectName("metQuitar")
        btn_quitar.setCursor(Qt.CursorShape.PointingHandCursor)
        encabezado.addWidget(btn_quitar)
        frame_layout.addLayout(encabezado)

        # Las cuatro estadísticas como una fila de números, y el n al final.
        fila_stats = QHBoxLayout()
        stats = {nombre: _Estadistico(nombre) for nombre in ESTADISTICOS}
        for est in stats.values():
            fila_stats.addWidget(est, stretch=1)
        lbl_n = QLabel("")
        lbl_n.setObjectName("metNota")
        lbl_n.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        fila_stats.addWidget(lbl_n, stretch=1)
        frame_layout.addLayout(fila_stats)

        plot = pg.PlotWidget()
        plot.setMinimumHeight(140)
        plot.setMaximumHeight(190)
        _estilizar_plot(plot, host.colors)
        frame_layout.addWidget(plot)

        bloque = {
            "columna": columna,
            "frame": frame,
            "lbl_titulo": lbl_titulo,
            "cards": stats,
            "lbls": {nombre: est.lbl_valor for nombre, est in stats.items()},
            "plot": plot,
            "filtro": None,          # filtro de filas de este bloque (ver filtro_filas.py)
            "lbl_filtro": lbl_filtro,
            "lbl_n": lbl_n,
        }
        btn_quitar.clicked.connect(lambda: self.quitar_bloque(bloque))
        btn_filtro.clicked.connect(lambda: self.filtrar_bloque(bloque))

        # Se inserta antes del stretch final, para que los bloques nuevos
        # se apilen de arriba hacia abajo y el espacio vacío quede al fondo.
        self.contenedor_layout.insertWidget(self.contenedor_layout.count() - 1, frame)
        self.bloques.append(bloque)
        self._refrescar_vacio()
        self.recalcular_bloque(bloque)
        host._actualizar_alarmas()

    def quitar_bloque(self, bloque):
        bloque["frame"].setParent(None)
        bloque["frame"].deleteLater()
        if bloque in self.bloques:
            self.bloques.remove(bloque)
        self._refrescar_vacio()
        # Las alarmas de tipo Métrica sobre esta columna NO se borran ni se
        # desactivan -- siguen evaluándose y pueden seguir mandando correo,
        # simplemente ya no tienen dónde pintarse en esta pestaña.

    def filtrar_bloque(self, bloque):
        host = self.host
        base = host.filtered_df if host.filtered_df is not None else host.df
        if base is None or base.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        dlg = DialogoFiltroFilas(base, bloque.get("filtro"),
                                 titulo=f"Filtrar filas · {bloque['columna']}", parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            bloque["filtro"] = dlg.get_filtro()
            texto = describir_filtro(bloque["filtro"])
            bloque["lbl_filtro"].setText(f"Solo donde {texto}" if texto else "")
            bloque["lbl_filtro"].setToolTip(
                texto + "\n\nLas alarmas de Métrica miran todas las filas, no este filtro, "
                        "así que en este bloque no se marcan en rojo." if texto else "")
            self.recalcular_bloque(bloque)
            host._actualizar_alarmas()

    def recalcular_bloque(self, bloque):
        host = self.host
        columna = bloque["columna"]
        if host.filtered_df is None or host.filtered_df.empty or columna not in host.filtered_df.columns:
            return
        filtro = bloque.get("filtro")
        base = aplicar_filtro(host.filtered_df, filtro) if filtro else host.filtered_df
        data = pd.to_numeric(base[columna], errors="coerce").dropna()
        if data.empty:
            if filtro:   # que no quede a la vista el resultado anterior
                for lbl in bloque["lbls"].values():
                    lbl.setText("-")
                bloque["lbl_n"].setText("")
                bloque["plot"].clear()
            return

        bloque["lbls"]["Media"].setText(f"{data.mean():,.2f}")
        bloque["lbls"]["Mediana"].setText(f"{data.median():,.2f}")
        moda = data.mode()
        bloque["lbls"]["Moda"].setText(f"{moda.iloc[0]:,.2f}" if not moda.empty else "N/A")
        bloque["lbls"]["Desviación Std"].setText(f"{data.std():,.2f}")

        _estilizar_plot(bloque["plot"], host.colors)
        n, dibujados = dibujar_caja_con_puntos(bloque["plot"], data, host.colors)
        bloque["lbl_n"].setText(
            f"n = {n:,}" if dibujados == n else f"n = {n:,} · se dibujan {dibujados:,} puntos"
        )

    def actualizar(self):
        for bloque in self.bloques:
            self.recalcular_bloque(bloque)
        self.host._actualizar_alarmas()

    # ------------------------------------------------------------------
    # Lo que la ventana principal le pide a esta pestaña
    # ------------------------------------------------------------------
    def renombrar_columna(self, nombre_actual, nuevo_nombre):
        for bloque in self.bloques:
            if bloque["columna"] == nombre_actual:
                bloque["columna"] = nuevo_nombre
                bloque["lbl_titulo"].setText(nuevo_nombre)

    def podar_columnas(self, columnas_numericas):
        """Quita los bloques cuya columna ya no existe en el dataset actual
        (ej. se cargó un archivo distinto): si no, quedarían mostrando el
        último valor calculado como si todavía fuera válido."""
        for bloque in list(self.bloques):
            if bloque["columna"] not in columnas_numericas:
                self.quitar_bloque(bloque)

    def marcar_alarmas(self, df):
        """Pinta en rojo las estadísticas que tengan una alarma de tipo Métrica activa."""
        host = self.host
        for bloque in self.bloques:
            columna = bloque["columna"]
            estado = {nombre: False for nombre in bloque["cards"]}
            for regla in host.reglas_alarma:
                if regla["tipo"] != "Métrica" or regla["objetivo"] != columna:
                    continue
                if bloque.get("filtro"):
                    continue   # la alarma mira todas las filas: no se pinta sobre un bloque filtrado
                valor = calcular_valor_actual(regla, df, host.indicadores)
                if evaluar_regla(regla, valor) and regla["estadistico"] in estado:
                    estado[regla["estadistico"]] = True
            for nombre, card in bloque["cards"].items():
                card.set_alarma(estado[nombre])
