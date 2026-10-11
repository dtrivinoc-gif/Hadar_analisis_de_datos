"""
Vista de la pestaña Indicadores, pensada como una página impresa y no como un
tablero: una tabla tipográfica donde cada fila es un indicador.

Ideas de Edward Tufte que se aplican acá:
  - Sin tarjetas, bordes, rellenos ni sombras: solo texto y datos. El color se
    reserva para lo que importa (el último punto de la línea, las alarmas).
  - Sparklines: la evolución de cada indicador en una línea del tamaño de una
    palabra, al lado de su número. El punto marca el último valor, los puntos
    chicos el mínimo y el máximo, y la banda gris el rango habitual (entre el
    25 % y el 75 % de los valores), para ver de un vistazo si el último valor
    es normal o se salió.
  - Notas al margen: lo accesorio (filtro, columnas de las que depende,
    alarmas, acciones) va en una columna angosta a la derecha, en cursiva y
    más chico, sin competir con el número.
  - Una tipografía con serifa y números alineados a la derecha.

Es deliberadamente distinto del Panel (tarjetas redondeadas, número grande
y línea con relleno): el Panel es para mirar de pasada, esto es para leer y
comparar con calma.

Este módulo solo dibuja; los cálculos los hace main_window y los pasa listos.
"""
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QScrollArea, QFrame,
)

from .config import COLOR_ACCENT
from .indicadores import INDICADOR_OPERACIONES_INV
from .panel import elegir_columna_fecha, colores_estado, _es_tema_claro

# Palatino en Windows y Mac, Cambria si no está. Se evita Georgia a propósito:
# sus números "de texto" (3, 4, 5, 7, 9 que bajan de la línea) no se alinean
# bien en una columna de cifras.
SERIF = "'Palatino Linotype', 'Book Antiqua', Palatino, Cambria, 'Times New Roman', serif"


def columna_fecha_para(host, df):
    """(columna, fechas) para dibujar la evolución, con el mismo criterio del
    Panel: primero la columna que usa la Línea de Tiempo."""
    rango = getattr(host, "rango_tiempo", None)
    preferida = rango[2] if rango else None
    dayfirst = getattr(getattr(host, "panel_linea_tiempo", None), "_dayfirst", True)
    fechas_de = None
    if hasattr(host, "_fechas_de_columna"):
        fechas_de = lambda d, c: host._fechas_de_columna(d, c, dayfirst)
    return elegir_columna_fecha(df, preferida, fechas_de)


def texto_cambio(variacion):
    """'+12,3 %' / '−4,0 %' / '—' (sin fechas no hay con qué comparar)."""
    if variacion is None or not np.isfinite(variacion):
        return "—"
    if abs(variacion) < 0.05:
        return "0,0 %"
    signo = "+" if variacion > 0 else "−"
    return f"{signo}{abs(variacion):.1f}".replace(".", ",") + " %"


def describir_operacion(ind):
    """'promedio de temperatura', 'A ÷ B', 'fórmula personalizada'."""
    if ind.operacion == "formula":
        return "fórmula personalizada"
    etiqueta = INDICADOR_OPERACIONES_INV.get(ind.operacion, ind.operacion)
    if ind.operacion == "razon":
        return f"{ind.columna} ÷ {ind.columna2}"
    if ind.operacion == "conteo_unico":
        return f"valores únicos de {ind.columna}"
    etiqueta = etiqueta.split(" (")[0].lower()   # 'Conteo (no nulos)' -> 'conteo'
    return f"{etiqueta} de {ind.columna}" if ind.columna else etiqueta


class Sparkline(QWidget):
    """Línea del tamaño de una palabra: sin ejes ni números."""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setMinimumWidth(140)
        self._colors = colors
        self._v = []

    def poner(self, valores, tooltip=""):
        self._v = [float(x) if np.isfinite(x) else np.nan for x in valores]
        self.setToolTip(tooltip)
        self.update()

    def paintEvent(self, _event):
        c = self._colors
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, mx, my = self.width(), self.height(), 6, 7
        n = len(self._v)
        validos = [x for x in self._v if np.isfinite(x)]

        if len(validos) < 3:   # sin serie de tiempo: solo una guía punteada
            guia = QColor(c["muted"])
            guia.setAlpha(110)
            p.setPen(QPen(guia, 1, Qt.DotLine))
            p.drawLine(QPointF(mx, h / 2), QPointF(w - mx, h / 2))
            p.end()
            return

        lo, hi = min(validos), max(validos)
        rango = (hi - lo) or 1.0

        def y_de(valor):
            return h - my - (valor - lo) / rango * (h - 2 * my) if hi != lo else h / 2

        def x_de(i):
            return mx + i * (w - 2 * mx) / (n - 1)

        # Rango habitual (25 %-75 %): una banda gris muy suave detrás de la línea.
        q1, q3 = np.percentile(validos, [25, 75])
        if q3 > q1:
            banda = QColor(c["muted"])
            banda.setAlpha(45)
            p.fillRect(QRectF(mx, y_de(q3), w - 2 * mx, y_de(q1) - y_de(q3)), banda)

        linea = QPainterPath()
        vacio = True
        for i, valor in enumerate(self._v):
            if not np.isfinite(valor):
                vacio = True
                continue
            punto = QPointF(x_de(i), y_de(valor))
            if vacio:
                linea.moveTo(punto)
                vacio = False
            else:
                linea.lineTo(punto)
        tinta = QColor(c["text"])
        tinta.setAlpha(210)
        p.setPen(QPen(tinta, 1.1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(linea)

        # Mínimo y máximo: puntos chicos y apagados. Último valor: el único con color.
        idx = [i for i, x in enumerate(self._v) if np.isfinite(x)]
        i_min = min(idx, key=lambda i: self._v[i])
        i_max = max(idx, key=lambda i: self._v[i])
        i_fin = idx[-1]
        p.setPen(Qt.NoPen)
        apagado = QColor(c["muted"])
        p.setBrush(apagado)
        for i in {i_min, i_max} - {i_fin}:
            p.drawEllipse(QPointF(x_de(i), y_de(self._v[i])), 1.8, 1.8)
        p.setBrush(QColor(COLOR_ACCENT))
        p.drawEllipse(QPointF(x_de(i_fin), y_de(self._v[i_fin])), 2.8, 2.8)
        p.end()


class VistaIndicadores(QWidget):
    """Encabezado, tabla de indicadores y leyenda. `acciones` es un dict de
    funciones: nuevo, sugerir, guardar, cargar (sin argumentos) y editar,
    quitar, panel (reciben el indicador)."""

    def __init__(self, acciones, colors, parent=None):
        super().__init__(parent)
        self._acc = acciones
        self._colors = colors
        self._filas_actuales = []

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(32, 22, 32, 14)
        raiz.setSpacing(0)

        self.lbl_titulo = QLabel("Indicadores")
        raiz.addWidget(self.lbl_titulo)
        self.lbl_bajada = QLabel(
            "Cada fila es una cifra que tú definiste. Se recalcula sola cuando cambian los datos o los filtros.")
        self.lbl_bajada.setWordWrap(True)
        raiz.addWidget(self.lbl_bajada)
        raiz.addSpacing(10)

        fila_acciones = QHBoxLayout()
        fila_acciones.setSpacing(24)
        self.lbl_izq = QLabel()
        self.lbl_der = QLabel()
        for lbl in (self.lbl_izq, self.lbl_der):
            lbl.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
            lbl.linkActivated.connect(self._al_pulsar_enlace)
        fila_acciones.addWidget(self.lbl_izq)
        fila_acciones.addStretch(1)
        fila_acciones.addWidget(self.lbl_der)
        raiz.addLayout(fila_acciones)
        raiz.addSpacing(14)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        raiz.addWidget(self.scroll, 1)

        self.lbl_leyenda = QLabel(
            "La línea muestra cómo evoluciona cada indicador a lo largo del período. "
            "El punto de color es el último valor; los puntos grises, el mínimo y el máximo; "
            "la banda gris, el rango habitual (del 25 % al 75 % de los valores). "
            "El cambio compara la segunda mitad del período con la primera.")
        self.lbl_leyenda.setWordWrap(True)
        raiz.addSpacing(8)
        raiz.addWidget(self.lbl_leyenda)

        self.aplicar_colores(colors)

    # -- estilo --------------------------------------------------------------
    def _css(self, tamano, color, extra=""):
        return (f"background: transparent; border: none; font-family: {SERIF}; "
                f"font-size: {tamano}px; color: {color}; {extra}")

    def aplicar_colores(self, colors):
        self._colors = colors
        c = colors
        self.setObjectName("vistaIndicadores")
        self.setStyleSheet(f"QWidget#vistaIndicadores {{ background-color: {c['bg']}; }}")
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.lbl_titulo.setStyleSheet(self._css(26, c["text"]))
        self.lbl_bajada.setStyleSheet(self._css(14, c["muted"], "font-style: italic;"))
        self.lbl_leyenda.setStyleSheet(self._css(12, c["muted"], "font-style: italic;"))
        enlace = lambda href, txt, color: (
            f'<a href="{href}" style="color:{color}; text-decoration:underline">{txt}</a>')
        self.lbl_izq.setText(
            enlace("nuevo", "Nuevo indicador", c["text"]) + "&nbsp;&nbsp;&nbsp;&nbsp;"
            + enlace("sugerir", "Sugerir automáticamente", c["muted"]))
        self.lbl_der.setText(
            enlace("guardar", "Guardar", c["muted"]) + "&nbsp;&nbsp;&nbsp;&nbsp;"
            + enlace("cargar", "Cargar", c["muted"]))
        for lbl in (self.lbl_izq, self.lbl_der):
            lbl.setStyleSheet(self._css(14, c["muted"]))

    def _al_pulsar_enlace(self, destino):
        funcion = self._acc.get(destino)
        if funcion:
            funcion()

    # -- contenido ------------------------------------------------------------
    def mostrar(self, filas, colors, hay_datos=True):
        """filas: lista de dicts con indicador, valor, variacion, tendencia,
        en_alarma, mensaje_alarma, columnas_afectadas y en_panel."""
        if colors is not self._colors:
            self.aplicar_colores(colors)
        self._filas_actuales = filas
        c = self._colors

        contenedor = QWidget()
        contenedor.setObjectName("tablaIndicadores")
        contenedor.setStyleSheet(f"QWidget#tablaIndicadores {{ background-color: {c['bg']}; }}")
        self.lbl_leyenda.setVisible(bool(filas))

        if not filas:
            lay = QVBoxLayout(contenedor)
            lay.setContentsMargins(0, 24, 0, 0)
            if hay_datos:
                titulo = "Todavía no hay indicadores."
                detalle = ("Crea el primero con «Nuevo indicador», o deja que Hadar "
                           "proponga algunos con «Sugerir automáticamente».")
            else:
                titulo = "Aún no hay datos."
                detalle = "Carga una tabla y aquí podrás definir tus indicadores."
            lbl = QLabel(titulo)
            lbl.setStyleSheet(self._css(18, c["text"]))
            lay.addWidget(lbl)
            lbl2 = QLabel(detalle)
            lbl2.setWordWrap(True)
            lbl2.setStyleSheet(self._css(14, c["muted"], "font-style: italic;"))
            lay.addWidget(lbl2)
            lay.addStretch(1)
            self.scroll.setWidget(contenedor)
            return

        grid = QGridLayout(contenedor)
        grid.setContentsMargins(0, 0, 8, 0)
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(0)
        for col, estiramiento in enumerate((5, 3, 3, 2, 6)):
            grid.setColumnStretch(col, estiramiento)
        grid.setColumnMinimumWidth(0, 190)
        grid.setColumnMinimumWidth(1, 150)
        grid.setColumnMinimumWidth(2, 120)
        grid.setColumnMinimumWidth(3, 80)
        grid.setColumnMinimumWidth(4, 220)

        # Encabezados de columna: texto chico y apagado, y una línea firme debajo.
        derecha = Qt.AlignRight | Qt.AlignBottom
        izquierda = Qt.AlignLeft | Qt.AlignBottom
        for col, (texto, alineacion) in enumerate((
                ("Indicador", izquierda), ("Evolución", izquierda), ("Valor", derecha),
                ("Cambio", derecha), ("Notas", izquierda))):
            lbl = QLabel(texto)
            lbl.setStyleSheet(self._css(12, c["muted"], "padding-bottom: 5px;"))
            if texto == "Cambio":
                lbl.setToolTip("Segunda mitad del período contra la primera")
            grid.addWidget(lbl, 0, col, alignment=alineacion)
        grid.addWidget(self._regla(c["text"], 150), 1, 0, 1, 5)

        fila_grid = 2
        for k, fila in enumerate(filas):
            self._agregar_fila(grid, fila_grid, fila)
            fila_grid += 1
            es_ultima = k == len(filas) - 1
            grid.addWidget(self._regla(c["border"], 255 if es_ultima else 90), fila_grid, 0, 1, 5)
            fila_grid += 1
        grid.setRowStretch(fila_grid, 1)
        self.scroll.setWidget(contenedor)

    def _regla(self, color, alfa):
        linea = QFrame()
        linea.setFixedHeight(1)
        q = QColor(color)
        q.setAlpha(alfa)
        linea.setStyleSheet(f"background-color: rgba({q.red()},{q.green()},{q.blue()},{q.alpha()}); border: none;")
        return linea

    def _agregar_fila(self, grid, r, fila):
        c = self._colors
        ind = fila["indicador"]
        est = colores_estado(c)
        en_alarma = fila.get("en_alarma")

        # Indicador: nombre y, debajo, qué calcula.
        bloque = QWidget()
        bl = QVBoxLayout(bloque)
        bl.setContentsMargins(0, 14, 0, 14)
        bl.setSpacing(1)
        nombre = QLabel(ind.nombre)
        nombre.setWordWrap(True)
        nombre.setStyleSheet(self._css(17, c["text"]))
        bl.addWidget(nombre)
        op = QLabel(describir_operacion(ind))
        op.setWordWrap(True)
        op.setStyleSheet(self._css(12, c["muted"], "font-style: italic;"))
        bl.addWidget(op)
        grid.addWidget(bloque, r, 0, alignment=Qt.AlignVCenter)

        # Evolución
        spark = Sparkline(c)
        tendencia = fila.get("tendencia") or []
        tip = ""
        validos = [v for v in tendencia if np.isfinite(v)]
        if validos:
            tip = (f"Mínimo {ind.formatear(min(validos))} · máximo {ind.formatear(max(validos))} "
                   f"(en {len(tendencia)} tramos del período)")
        else:
            tip = "Sin una columna de fechas no se puede dibujar la evolución."
        spark.poner(tendencia, tip)
        grid.addWidget(spark, r, 1, alignment=Qt.AlignVCenter)

        # Valor: cifras alineadas a la derecha.
        valor = QLabel(ind.formatear(fila.get("valor")))
        color_valor = est["baja"] if en_alarma else c["text"]
        valor.setStyleSheet(self._css(28, color_valor))
        grid.addWidget(valor, r, 2, alignment=Qt.AlignRight | Qt.AlignVCenter)

        # Cambio: solo el número con su signo, sin flechas ni colores.
        cambio = QLabel(texto_cambio(fila.get("variacion")))
        cambio.setStyleSheet(self._css(15, c["muted"]))
        cambio.setToolTip("Segunda mitad del período contra la primera")
        grid.addWidget(cambio, r, 3, alignment=Qt.AlignRight | Qt.AlignVCenter)

        # Notas al margen.
        notas = QWidget()
        nl = QVBoxLayout(notas)
        nl.setContentsMargins(0, 12, 0, 12)
        nl.setSpacing(3)

        def nota(texto, color=None, extra=""):
            lbl = QLabel(texto)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(self._css(12, color or c["muted"], "font-style: italic; " + extra))
            nl.addWidget(lbl)

        acotado = ind.describir_acotado()
        if acotado:
            nota(acotado)
        if ind.operacion == "formula":
            usa = sorted(ind.columnas_de_las_que_depende())
            if usa:
                nota("Usa " + ", ".join(usa))
        afectadas = fila.get("columnas_afectadas") or []
        if afectadas:
            ambar = "#b45309" if _es_tema_claro(c) else "#FBBF24"
            lista = ", ".join(f"«{x}»" for x in afectadas)
            nota(f"{lista} tiene una anomalía activa (ver Narrativa)" if len(afectadas) == 1
                 else f"{lista} tienen una anomalía activa (ver Narrativa)", ambar, "font-style: normal;")
        if en_alarma and fila.get("mensaje_alarma"):
            nota(f"Alarma: {fila['mensaje_alarma']}", est["baja"], "font-style: normal; font-weight: 600;")
        elif en_alarma:
            nota("Alarma activa", est["baja"], "font-style: normal; font-weight: 600;")
        if fila.get("en_panel"):
            nota("Se muestra en el Panel")

        enlaces = QLabel()
        enlaces.setStyleSheet(self._css(12, c["muted"]))
        estilo = f'style="color:{c["muted"]}; text-decoration:underline"'
        partes = [f'<a href="editar" {estilo}>editar</a>']
        if not fila.get("en_panel"):
            partes.append(f'<a href="panel" {estilo}>al Panel</a>')
        partes.append(f'<a href="quitar" {estilo}>quitar</a>')
        enlaces.setText("&nbsp;&nbsp;&nbsp;".join(partes))
        enlaces.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        enlaces.linkActivated.connect(lambda destino, i=ind: self._acc[destino](i))
        nl.addWidget(enlaces)
        grid.addWidget(notas, r, 4, alignment=Qt.AlignVCenter)
