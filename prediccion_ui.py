"""
prediccion_ui.py — La pestaña Predicción (la lógica de cálculo vive en
prediccion.py). Misma idea que PanelControl en panel.py: `host` es la
ventana principal y de ahí se leen tablas, nombre_tabla_activa y colors.

Flujo para el usuario:
  1. Elegir la tabla de entrenamiento, (opcional) la tabla de prueba y la
     columna a predecir.
  2. Revisar la lista de columnas: Hadar deja marcadas las que sirven y
     desmarca ids, constantes, etc. -- cualquiera se puede cambiar a mano.
  3. "Entrenar modelo": muestra qué tan bien acierta frente a una respuesta
     tonta y en qué columnas se fijó.
  4. "Generar predicciones…": guarda un CSV listo para entregar.

Cada proyecto guarda su propia configuración (ver proyecto.py); el modelo
entrenado NO se guarda, se reentrena con un clic (ver nota en prediccion.py).
"""
import html
import re

from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .panel import estilo_tarjeta

try:
    import pyqtgraph as pg
except Exception:  # sin pyqtgraph se omite solo el gráfico
    pg = None
from .relacional import Enlace, SIN_FECHA, enlaces_disponibles, preparar
from .prediccion import _fmt as _formatear
from .prediccion import (
    ConfigPrediccion, TIPOS_LEGIBLES, columnas_a_usar, elegir_columna_id, entrenar,
    predecir, sugerir_columnas,
)

# Estilo "Tufte": poca tinta, todo lo que se dibuja es dato. Un solo color de énfasis
# (el índigo de Hadar), gris para lo demás, y verde/ámbar/rojo solo para el veredicto.
ACENTO = "#6366f1"
VERDE = "#34d399"
AMBAR = "#fbbf24"
ROJO = "#f87171"
GRIS = "#71717a"
COLOR_NIVEL = {"muy_bueno": VERDE, "bueno": VERDE, "regular": AMBAR, "no_aporta": ROJO, "sospechoso": AMBAR}
FUENTE_SERIF = '"Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif'
REGLA = "rgba(127,127,127,0.30)"

_QSS_TABLA = (
    "QTableWidget { background: transparent; border: none; gridline-color: transparent; "
    f"font-family: {FUENTE_SERIF}; }}"
    "QTableWidget::item { padding: 3px 6px; border: none; }"
    "QHeaderView::section { background: transparent; border: none; "
    "border-bottom: 1px solid rgba(127,127,127,0.55); padding: 4px 6px; color: #a1a1aa; font-weight: 600; }"
)


def _color_texto(widget, alfa=255):
    c = QColor(widget.palette().color(widget.foregroundRole()))
    c.setAlpha(int(alfa))
    return c


def _vaciar(layout):
    """Quita todo lo que hay dentro de un layout (para redibujar el resultado)."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        else:
            sub = item.layout()
            if sub is not None:
                _vaciar(sub)


def _geometria_mancuerna(ancho, a, b, maximo, wa, wb):
    """Posiciones (en píxeles) del gráfico de dos puntos. Separado del dibujo para poder probarlo."""
    m = 10
    x0, x1 = m, max(ancho - m, m + 1)

    def X(v):
        return x0 + (x1 - x0) * min(max(v / maximo, 0.0), 1.0) if maximo > 0 else x0

    xa, xb = X(a), X(b)
    ta = min(max(xa - wa / 2, x0), max(x1 - wa, x0))
    tb = min(max(xb - wb / 2, x0), max(x1 - wb, x0))
    choque = not (ta + wa + 8 <= tb or tb + wb + 8 <= ta)
    return {"x0": x0, "x1": x1, "xa": xa, "xb": xb, "ta": ta, "tb": tb, "b_arriba": choque}


class _Mancuerna(QWidget):
    """Dos puntos sobre un mismo eje unidos por una línea fina, con las etiquetas escritas
    directamente al lado (sin leyenda). Muestra el «antes y después» de un vistazo."""

    def __init__(self, valor_a, valor_b, maximo, texto_a, texto_b, brecha="", parent=None):
        super().__init__(parent)
        self._a, self._b, self._max = float(valor_a), float(valor_b), float(maximo)
        self._ta, self._tb, self._brecha = texto_a, texto_b, brecha
        self.setFixedHeight(66)
        self.setMinimumWidth(260)

    def paintEvent(self, evento):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        fm = p.fontMetrics()
        wa, wb = fm.horizontalAdvance(self._ta), fm.horizontalAdvance(self._tb)
        g = _geometria_mancuerna(self.width(), self._a, self._b, self._max, wa, wb)
        y = 32
        p.setPen(QPen(_color_texto(self, 70), 1))
        p.drawLine(int(g["x0"]), y, int(g["x1"]), y)                      # eje
        p.setPen(QPen(QColor(ACENTO), 2))
        p.drawLine(int(g["xa"]), y, int(g["xb"]), y)                      # distancia entre los dos
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACENTO))
        p.drawEllipse(QPointF(g["xa"], y), 5, 5)
        p.setBrush(_color_texto(self, 150))
        p.drawEllipse(QPointF(g["xb"], y), 5, 5)
        p.setPen(_color_texto(self, 255))
        p.drawText(int(g["ta"]), y + 24, self._ta)
        p.drawText(int(g["tb"]), (y - 12) if g["b_arriba"] else (y + 24), self._tb)
        if self._brecha and abs(g["xb"] - g["xa"]) > fm.horizontalAdvance(self._brecha) + 16:
            p.setPen(QColor(ACENTO))
            p.drawText(int((g["xa"] + g["xb"]) / 2 - fm.horizontalAdvance(self._brecha) / 2), y - 10, self._brecha)
        p.end()


class _PuntoLinea(QWidget):
    """Una fila de un gráfico de puntos: línea punteada muy fina y un punto en el valor (0 a 100)."""

    def __init__(self, porcentaje, color=ACENTO, parent=None):
        super().__init__(parent)
        self._pct = max(0.0, min(float(porcentaje), 100.0))
        self._color = QColor(color)
        self.setFixedHeight(18)
        self.setMinimumWidth(80)

    def paintEvent(self, evento):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        y = self.height() / 2
        p.setPen(QPen(_color_texto(self, 70), 1, Qt.DotLine))
        p.drawLine(0, int(y), self.width(), int(y))
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(QPointF(4 + (self.width() - 8) * self._pct / 100.0, y), 4, 4)
        p.end()


class _PuntoDivergente(QWidget):
    """Punto a la izquierda o derecha de una marca central (valor entre -1 y 1)."""

    def __init__(self, valor, color, parent=None):
        super().__init__(parent)
        self._v = max(-1.0, min(float(valor), 1.0))
        self._color = QColor(color)
        self.setFixedHeight(18)
        self.setMinimumWidth(90)

    def paintEvent(self, evento):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, y = self.width(), self.height() / 2
        c = w / 2
        p.setPen(QPen(_color_texto(self, 60), 1))
        p.drawLine(0, int(y), w, int(y))
        p.drawLine(int(c), int(y - 5), int(c), int(y + 5))
        x = c + (w / 2 - 6) * self._v
        p.setPen(QPen(self._color, 1))
        p.drawLine(int(c), int(y), int(x), int(y))
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(QPointF(x, y), 4, 4)
        p.end()


_NINGUNA = "(ninguna)"
_ELIGE = "(elige una columna)"
_AUTO = "(automática)"


if pg is not None:
    class _GraficoEstatico(pg.PlotWidget):
        """Gráfico que no hace zoom ni se arrastra: la rueda del mouse sigue
        desplazando la pantalla (antes, al pasar por encima, el gráfico se alejaba)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setMouseEnabled(False, False)

        def wheelEvent(self, ev):
            ev.ignore()


class _HiloEntrenamiento(QThread):
    """Entrena en segundo plano para que la ventana no se congele."""
    progreso = Signal(str)
    terminado = Signal(object)
    fallo = Signal(str)

    def __init__(self, df, config, columnas, forzar, parent=None):
        super().__init__(parent)
        self._df = df
        self._config = config
        self._columnas = columnas
        self._forzar = forzar

    def run(self):
        try:
            resultado = entrenar(self._df, self._config, self._columnas,
                                 progreso=self.progreso.emit, forzar_categoria=self._forzar)
            self.terminado.emit(resultado)
        except Exception as e:  # el mensaje llega al usuario en español desde prediccion.py
            self.fallo.emit(str(e))


class PanelPrediccion(QWidget):
    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.colors = host.colors
        self._config = ConfigPrediccion()
        self._sugerencias = []
        self._cols_fecha = []
        self._resultado = None
        self._hilo = None
        self._entrenando = False
        self._cargando = False
        self._firma_tablas = None
        self._enlaces_lista = []
        self._prep = None
        self._df_train_ef = None
        self._df_test_ef = None
        self._claves_forzadas = set()
        self._rel_cache = []
        self._rel_cache_firma = None
        self._construir()
        self.aplicar_tema(self.colors)

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    def _tarjeta(self, titulo, subtitulo=""):
        """Una sección: sin caja ni fondo, separada de la anterior por una línea fina."""
        marco = QFrame()
        marco.setObjectName("seccionTufte")
        marco.setStyleSheet(f"#seccionTufte{{background: transparent; border: none; border-top: 1px solid {REGLA};}}")
        lay = QVBoxLayout(marco)
        lay.setContentsMargins(0, 16, 0, 10)
        lay.setSpacing(8)
        m = re.match(r"^(\d)\.\s+(.*)$", titulo)
        etiqueta = f"{m.group(1)}   {m.group(2)}" if m else titulo
        lay.addWidget(self._titulo_chico(etiqueta))
        if subtitulo:
            sub = QLabel(subtitulo)
            sub.setObjectName("muted")
            sub.setWordWrap(True)
            lay.addWidget(sub)
        return marco, lay

    def _titulo_chico(self, texto):
        """Título de sección en versalitas, gris y discreto."""
        lbl = QLabel(texto)
        lbl.setStyleSheet("color: #a1a1aa; font-weight: 600;")
        try:
            f = lbl.font()
            f.setCapitalization(QFont.SmallCaps)
            f.setLetterSpacing(QFont.AbsoluteSpacing, 0.8)
            f.setPixelSize(14)
            lbl.setFont(f)
        except Exception:
            lbl.setStyleSheet("color: #a1a1aa; font-weight: 600; font-size: 13px;")
        return lbl

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

        titulo = QLabel("Predicción")
        titulo.setStyleSheet(f"font-size: 24px; font-family: {FUENTE_SERIF};")
        lay.addWidget(titulo)
        sub = QLabel("Entrena un modelo con una tabla de este proyecto y predice una columna. "
                     "El modelo es solo de este proyecto: si abres otro, parte desde cero.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        self.lbl_vacio = QLabel("Carga datos para usar esta pestaña.")
        self.lbl_vacio.setObjectName("muted")
        self.lbl_vacio.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_vacio)

        # --- tarjeta 1: qué predecir --------------------------------
        self.card_config, c1 = self._tarjeta("1. Qué quieres predecir")
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(8)
        c1.addLayout(g)

        self.combo_train = QComboBox()
        self.combo_test = QComboBox()
        self.combo_obj = QComboBox()
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(["Automático", "Categoría (clasificar)", "Número (estimar)"])
        self.chk_log = QCheckBox("Predecir en escala logarítmica (para números que crecen muy rápido; solo valores 0 o más)")
        self.combo_valid = QComboBox()
        self.combo_valid.addItems(["Aleatoria", "Por fecha (probar con las últimas filas)",
                                   "Por grupos (una misma persona o cosa no queda a ambos lados)"])
        self.combo_fecha = QComboBox()
        self.combo_grupo = QComboBox()
        self.spin_pliegues = QSpinBox()
        self.spin_pliegues.setRange(1, 10)
        self.combo_salida = QComboBox()
        self.combo_salida.addItems(["Clase / valor", "Probabilidades"])
        self.combo_id = QComboBox()

        filas = [
            ("Tabla de entrenamiento", self.combo_train, "Tabla de prueba (a la que se predice)", self.combo_test),
            ("Columna a predecir", self.combo_obj, "Tipo de problema", self.combo_tipo),
            ("Cómo probar el modelo", self.combo_valid, "Columna de fecha (si es por fecha)", self.combo_fecha),
            ("Cuántas pruebas hacer", self.spin_pliegues, "Columna de grupos (si es por grupos)", self.combo_grupo),
            ("Qué entregar", self.combo_salida, "Columna id del archivo de entrega", self.combo_id),
        ]
        for i, (t1, w1, t2, w2) in enumerate(filas):
            g.addWidget(QLabel(t1), i, 0)
            g.addWidget(w1, i, 1)
            g.addWidget(QLabel(t2), i, 2)
            g.addWidget(w2, i, 3)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        c1.addWidget(self.chk_log)
        lay.addWidget(self.card_config)
        self.combo_tipo.setToolTip("Automático lo decide solo: si lo que predices es una categoría (sí/no, tipo A/B) "
                                   "clasifica; si es una cantidad, estima el número.")
        self.combo_valid.setToolTip("Aleatoria: separa al azar partes de las filas para probar el modelo.\n"
                                    "Por fecha: prueba siempre con filas más nuevas que las de entrenamiento, que es "
                                    "lo que pasa en la realidad cuando se predice el futuro.\n"
                                    "Por grupos: si una misma persona o cosa aparece en varias filas, todas sus filas "
                                    "van juntas al entrenamiento o a la prueba (el resultado es más realista).")
        self.spin_pliegues.setToolTip("El modelo se prueba varias veces, cada vez con una parte distinta de los datos, "
                                      "y se muestra el promedio y cuánto varía. 1 = una sola prueba (más rápido). "
                                      "Con tablas muy grandes se hace una sola prueba.")
        self.combo_grupo.setToolTip("La columna que identifica a la persona o cosa que se repite (por ejemplo un código "
                                    "de cliente o de producto).")
        self.combo_salida.setToolTip("Clase / valor: entrega la respuesta directa.\n"
                                     "Probabilidades: entrega qué tan probable es cada opción.")
        self.combo_id.setToolTip("La columna que identifica cada fila en el archivo que entregas. "
                                 "«Automática» la busca sola.")
        self.chk_log.setToolTip("Útil si los valores crecen muy rápido (precios, montos muy dispares): el modelo aprende "
                                "en una escala más pareja y entrega el resultado en la escala original.")
        self.combo_obj.setToolTip("El dato que quieres adelantar.")

        # --- tarjeta: tablas relacionadas ---------------------------
        self.card_rel, cr = self._tarjeta(
            "Tablas relacionadas (opcional)",
            "Si tu base tiene varias tablas (ventas, productos, clientes…), Hadar puede traer al modelo datos de "
            "las otras tablas usando las relaciones de la ontología. Con una sola tabla todo funciona igual que antes.")
        self.chk_rel = QCheckBox("Incluir datos de tablas relacionadas")
        cr.addWidget(self.chk_rel)
        self.panel_rel = QWidget()
        pr = QVBoxLayout(self.panel_rel)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.setSpacing(8)
        gr = QGridLayout()
        self.spin_saltos = QSpinBox()
        self.spin_saltos.setRange(1, 3)
        self.spin_saltos.setValue(2)
        self.combo_fecha_rel = QComboBox()
        gr.addWidget(QLabel("Saltos máximos entre tablas"), 0, 0)
        gr.addWidget(self.spin_saltos, 0, 1)
        gr.addWidget(QLabel("Fecha para resumir tablas con muchas filas"), 0, 2)
        gr.addWidget(self.combo_fecha_rel, 0, 3)
        gr.setColumnStretch(1, 1)
        gr.setColumnStretch(3, 1)
        pr.addLayout(gr)
        self.tabla_enl = QTableWidget(0, 2)
        self.tabla_enl.setHorizontalHeaderLabels(["Usar", "Relación entre tablas"])
        self.tabla_enl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla_enl.setSelectionMode(QAbstractItemView.NoSelection)
        self.tabla_enl.verticalHeader().setVisible(False)
        self.tabla_enl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tabla_enl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tabla_enl.setMinimumHeight(90)
        self.tabla_enl.setMaximumHeight(170)
        pr.addWidget(self.tabla_enl)
        fila_m = QHBoxLayout()
        fila_m.addWidget(QLabel("Relación manual:"))
        self.combo_ma, self.combo_ca = QComboBox(), QComboBox()
        self.combo_mb, self.combo_cb = QComboBox(), QComboBox()
        for w in (self.combo_ma, self.combo_ca):
            fila_m.addWidget(w, 1)
        fila_m.addWidget(QLabel("↔"))
        for w in (self.combo_mb, self.combo_cb):
            fila_m.addWidget(w, 1)
        self.btn_add_enl = QPushButton("Agregar")
        fila_m.addWidget(self.btn_add_enl)
        pr.addLayout(fila_m)
        self.lbl_rel_info = QLabel("")
        self.lbl_rel_info.setObjectName("muted")
        self.lbl_rel_info.setWordWrap(True)
        pr.addWidget(self.lbl_rel_info)
        cr.addWidget(self.panel_rel)
        self.panel_rel.setVisible(False)
        self.card_rel.setVisible(False)
        lay.addWidget(self.card_rel)
        self.chk_rel.setToolTip("Trae al modelo datos de las otras tablas (por ejemplo, la categoría de cada producto) "
                                "usando las relaciones entre tablas.")
        self.spin_saltos.setToolTip("Hasta cuántas tablas encadenadas se recorren (venta → producto → proveedor = 2).")
        self.combo_fecha_rel.setToolTip("Cuando una fila tiene muchas en otra tabla (un cliente con muchas compras), se "
                                        "resume usando solo lo ocurrido ANTES de la fecha de cada fila, para que el modelo "
                                        "no se entere del futuro.")

        self.chk_rel.toggled.connect(lambda _v: self._cambio_relaciones())
        self.spin_saltos.valueChanged.connect(lambda _v: self._cambio_relaciones())
        self.combo_fecha_rel.activated.connect(lambda _i: self._cambio_relaciones())
        self.tabla_enl.itemChanged.connect(self._cambio_enlace_casilla)
        self.combo_ma.activated.connect(lambda _i: self._llenar_columnas_de(self.combo_ma, self.combo_ca))
        self.combo_mb.activated.connect(lambda _i: self._llenar_columnas_de(self.combo_mb, self.combo_cb))
        self.btn_add_enl.clicked.connect(self._agregar_enlace_manual)

        self.combo_train.activated.connect(self._cambio_tabla)
        self.combo_test.activated.connect(self._cambio_tabla)
        self.combo_obj.activated.connect(self._cambio_objetivo)
        for w in (self.combo_tipo, self.combo_valid, self.combo_fecha, self.combo_grupo, self.combo_salida, self.combo_id):
            w.activated.connect(self._cambio_opcion)
        self.chk_log.toggled.connect(lambda _v: self._cambio_opcion())
        self.spin_pliegues.valueChanged.connect(lambda _v: self._cambio_opcion())

        # --- tarjeta 2: columnas ------------------------------------
        self.card_cols, c2 = self._tarjeta(
            "2. En qué debe fijarse el modelo",
            "Hadar deja marcadas las columnas que sirven y desmarca las que no (identificadores, "
            "valores constantes, columnas que no existen en la tabla de prueba…). Cambia lo que quieras.")
        fila_btn = QHBoxLayout()
        self.btn_marcar = QPushButton("Marcar todas")
        self.btn_desmarcar = QPushButton("Desmarcar todas")
        self.btn_restablecer = QPushButton("Volver a la sugerencia")
        for b in (self.btn_marcar, self.btn_desmarcar, self.btn_restablecer):
            fila_btn.addWidget(b)
        fila_btn.addStretch()
        c2.addLayout(fila_btn)
        self.btn_marcar.clicked.connect(lambda: self._marcar_todas(True))
        self.btn_desmarcar.clicked.connect(lambda: self._marcar_todas(False))
        self.btn_restablecer.clicked.connect(self._restablecer_columnas)

        self.tabla = QTableWidget(0, 4)
        self.tabla.setHorizontalHeaderLabels(["Usar", "Columna", "Tipo", "Motivo"])
        self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla.setSelectionMode(QAbstractItemView.NoSelection)
        self.tabla.verticalHeader().setVisible(False)
        cab = self.tabla.horizontalHeader()
        cab.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        cab.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        cab.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        cab.setSectionResizeMode(3, QHeaderView.Stretch)
        self.tabla.setMinimumHeight(240)
        self.tabla.itemChanged.connect(self._cambio_casilla)
        c2.addWidget(self.tabla)
        lay.addWidget(self.card_cols)

        # --- opciones avanzadas -------------------------------------
        self.btn_avanzado = QPushButton("Opciones avanzadas ▸")
        self.btn_avanzado.setFlat(True)
        self.btn_avanzado.setCursor(Qt.PointingHandCursor)
        self.btn_avanzado.clicked.connect(self._alternar_avanzado)
        lay.addWidget(self.btn_avanzado, alignment=Qt.AlignLeft)

        self.card_avanzado = QFrame()
        self.card_avanzado.setObjectName("seccionTufte")
        self.card_avanzado.setStyleSheet(f"#seccionTufte{{background: transparent; border: none; border-top: 1px solid {REGLA};}}")
        ga = QGridLayout(self.card_avanzado)
        ga.setContentsMargins(16, 12, 16, 12)
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 5000)
        self.spin_tasa = QDoubleSpinBox()
        self.spin_tasa.setRange(0.005, 1.0)
        self.spin_tasa.setDecimals(3)
        self.spin_tasa.setSingleStep(0.01)
        self.spin_prof = QSpinBox()
        self.spin_prof.setRange(0, 60)
        self.spin_prof.setSpecialValueText("automática")
        self.spin_filas = QSpinBox()
        self.spin_filas.setRange(1000, 100_000_000)
        self.spin_filas.setSingleStep(50_000)
        self.spin_semilla = QSpinBox()
        self.spin_semilla.setRange(0, 99_999)
        self.combo_balancear = QComboBox()
        self.combo_balancear.addItems(["Automático", "Siempre dar más peso", "No dar más peso"])
        self.combo_balancear.activated.connect(self._cambio_opcion)
        opciones = [
            ("Iteraciones máximas (más = más preciso y lento)", self.spin_iter),
            ("Tasa de aprendizaje", self.spin_tasa),
            ("Profundidad máxima de los árboles", self.spin_prof),
            ("Máximo de filas para entrenar (si hay más, usa una muestra)", self.spin_filas),
            ("Semilla (para que el resultado se repita)", self.spin_semilla),
            ("Clase poco frecuente (sí/no raros): más peso para que no la ignore", self.combo_balancear),
        ]
        for i, (texto, w) in enumerate(opciones):
            ga.addWidget(QLabel(texto), i, 0)
            ga.addWidget(w, i, 1)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(lambda _v: self._cambio_opcion())
        ga.setColumnStretch(0, 1)
        self.card_avanzado.setVisible(False)
        lay.addWidget(self.card_avanzado)

        # --- entrenar ------------------------------------------------
        fila = QHBoxLayout()
        self.btn_entrenar = QPushButton("Entrenar modelo")
        self.btn_entrenar.clicked.connect(self._entrenar)
        fila.addWidget(self.btn_entrenar)
        self.barra = QProgressBar()
        self.barra.setRange(0, 0)
        self.barra.setMaximumWidth(180)
        self.barra.setTextVisible(False)
        self.barra.setVisible(False)
        fila.addWidget(self.barra)
        self.lbl_estado = QLabel("")
        self.lbl_estado.setObjectName("muted")
        fila.addWidget(self.lbl_estado, 1)
        lay.addLayout(fila)

        # --- resultado -----------------------------------------------
        self.card_resultado, c4 = self._tarjeta("3. Resultado")
        self.resultado_cont = QWidget()
        self.resultado_lay = QVBoxLayout(self.resultado_cont)
        self.resultado_lay.setContentsMargins(0, 0, 0, 0)
        self.resultado_lay.setSpacing(12)
        c4.addWidget(self.resultado_cont)
        self._df_revision = None
        fila2 = QHBoxLayout()
        self.btn_predecir = QPushButton("Generar predicciones…")
        self.btn_predecir.clicked.connect(self._generar_predicciones)
        fila2.addWidget(self.btn_predecir)
        self.lbl_pred = QLabel("")
        self.lbl_pred.setObjectName("muted")
        self.lbl_pred.setWordWrap(True)
        fila2.addWidget(self.lbl_pred, 1)
        c4.addLayout(fila2)
        self.card_resultado.setVisible(False)
        lay.addWidget(self.card_resultado)
        lay.addStretch(1)

        self.spin_iter.setToolTip("Cuánto aprende el modelo. Más es más preciso pero más lento; se detiene solo si deja de mejorar.")
        self.spin_tasa.setToolTip("Velocidad de aprendizaje. Más baja es más cuidadosa y lenta.")
        self.spin_prof.setToolTip("Complejidad de cada árbol de decisión. «Automática» funciona bien.")
        self.spin_filas.setToolTip("Si tu tabla tiene más filas, se entrena con una muestra para ir más rápido.")
        self.spin_semilla.setToolTip("Hace que el resultado se repita igual cada vez.")
        self.combo_balancear.setToolTip("Cuando una categoría es muy rara (por ejemplo, fraudes entre miles de ventas) "
                                        "el modelo tiende a ignorarla. «Automático» le da más peso si es menos del 10%.")
        self._cargar_avanzado_desde_config()
        self._alternar_visibilidad_vacio()

    # ------------------------------------------------------------------
    # Ciclo de vida (misma interfaz que PanelControl)
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        self.colors = colors
        self.setStyleSheet(estilo_tarjeta(colors) + f" QLabel {{ font-family: {FUENTE_SERIF}; }}")
        self.btn_entrenar.setStyleSheet(
            f"QPushButton{{background: transparent; color: {ACENTO}; font-weight: 600; padding: 7px 22px; "
            f"border: 1px solid {ACENTO}; border-radius: 3px;}} "
            f"QPushButton:hover{{background: rgba(99,102,241,0.16);}} "
            f"QPushButton:disabled{{color: rgba(127,127,127,0.6); border-color: rgba(127,127,127,0.35);}}")
        for tabla in (self.tabla, self.tabla_enl):
            tabla.setStyleSheet(_QSS_TABLA)
            tabla.setShowGrid(False)

    def notificar_cambio(self):
        """La ventana principal avisa que algo cambió. Solo se rehace la
        pantalla si cambiaron las TABLAS (no en cada filtro o indicador)."""
        if self._firma() != self._firma_tablas and self.isVisible():
            self._refrescar_todo()

    def showEvent(self, event):
        super().showEvent(event)
        if self._firma() != self._firma_tablas:
            self._refrescar_todo()

    def _tablas(self):
        return getattr(self.host, "tablas", None) or {}

    def _firma(self):
        rel = getattr(self.host, "relaciones_ontologia", None)
        return tuple((n, df.shape) for n, df in self._tablas().items()) + (("__rel", len(rel or [])),)

    # ------------------------------------------------------------------
    # Guardar / cargar configuración del proyecto
    # ------------------------------------------------------------------
    def config_a_dict(self) -> dict:
        # Si la pestaña nunca se mostró, los combos están vacíos: leerlos
        # borraría la configuración que el proyecto trae guardada.
        if self._firma_tablas is not None:
            self._leer_formulario()
        return self._config.to_dict()

    def cargar_config(self, datos):
        self._config = ConfigPrediccion.from_dict(datos)
        self._resultado = None
        self._firma_tablas = None
        self._cargar_avanzado_desde_config()
        if self.isVisible():
            self._refrescar_todo()

    def _cargar_avanzado_desde_config(self):
        c = self._config
        self._cargando = True
        self.spin_iter.setValue(int(c.max_iter))
        self.spin_tasa.setValue(float(c.tasa_aprendizaje))
        self.spin_prof.setValue(int(c.profundidad_max))
        self.spin_filas.setValue(int(c.max_filas))
        self.spin_semilla.setValue(int(c.semilla))
        self.combo_balancear.setCurrentIndex({"auto": 0, "si": 1, "no": 2}.get(c.balancear, 0))
        self.spin_pliegues.setValue(int(c.pliegues))
        self._cargando = False

    # ------------------------------------------------------------------
    # Rellenar la pantalla
    # ------------------------------------------------------------------
    def _alternar_visibilidad_vacio(self):
        hay = bool(self._tablas())
        self.lbl_vacio.setVisible(not hay)
        for w in (self.card_config, self.card_cols, self.btn_avanzado, self.btn_entrenar):
            w.setVisible(hay)
        if not hay:
            self.card_avanzado.setVisible(False)
            self.card_resultado.setVisible(False)
            self.card_rel.setVisible(False)

    def _refrescar_todo(self):
        self._firma_tablas = self._firma()
        self._resultado = None
        self.card_resultado.setVisible(False)
        self._alternar_visibilidad_vacio()
        tablas = self._tablas()
        if not tablas:
            return
        c = self._config
        self._cargando = True
        nombres = list(tablas.keys())

        self.combo_train.clear()
        self.combo_train.addItems(nombres)
        entrena = c.tabla_entrenamiento if c.tabla_entrenamiento in tablas else (
            getattr(self.host, "nombre_tabla_activa", None) or nombres[0])
        if entrena not in tablas:
            entrena = nombres[0]
        self.combo_train.setCurrentText(entrena)

        self.combo_test.clear()
        self.combo_test.addItem(_NINGUNA)
        self.combo_test.addItems([n for n in nombres if n != entrena])
        self.combo_test.setCurrentText(c.tabla_prueba if c.tabla_prueba in tablas and c.tabla_prueba != entrena else _NINGUNA)

        self._llenar_objetivo(entrena, c.objetivo)
        self.combo_tipo.setCurrentIndex({"auto": 0, "clasificacion": 1, "regresion": 2}.get(c.tipo, 0))
        self.chk_log.setChecked(bool(c.log_objetivo))
        self.combo_valid.setCurrentIndex({"temporal": 1, "grupos": 2}.get(c.validacion, 0))
        self.combo_salida.setCurrentIndex(1 if c.salida == "probabilidad" else 0)
        self.chk_rel.setChecked(bool(c.usar_relaciones))
        self.spin_saltos.setValue(int(c.profundidad_relaciones))
        self._cargando = False
        self._reconstruir_columnas()

    def _llenar_objetivo(self, tabla, objetivo):
        df = self._tablas().get(tabla)
        self.combo_obj.clear()
        self.combo_obj.addItem(_ELIGE)
        if df is not None:
            self.combo_obj.addItems([str(col) for col in df.columns])
            if objetivo in [str(col) for col in df.columns]:
                self.combo_obj.setCurrentText(objetivo)

    def _df_entrenamiento(self):
        return self._tablas().get(self.combo_train.currentText())

    def _df_prueba(self):
        nombre = self.combo_test.currentText()
        return None if nombre in ("", _NINGUNA) else self._tablas().get(nombre)

    def _objetivo_actual(self):
        t = self.combo_obj.currentText()
        return None if t in ("", _ELIGE) else t

    def _reconstruir_columnas(self, refrescar_relaciones=True):
        """Recalcula la sugerencia de columnas y llena la tabla. Si hay tablas
        relacionadas activadas, las columnas nuevas también aparecen aquí."""
        df_plana = self._df_entrenamiento()
        obj = self._objetivo_actual()
        if refrescar_relaciones:
            self._llenar_relaciones()
        self._cargando = True
        self.tabla.setRowCount(0)
        self._sugerencias = []
        self._cols_fecha = []
        self._df_train_ef = None
        self._df_test_ef = None
        self._prep = None
        self._cargando = False
        if df_plana is None or obj is None or obj not in [str(c) for c in df_plana.columns]:
            self._llenar_combos_dependientes()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            df, df_prueba = self._calcular_efectivos()
            self._df_train_ef, self._df_test_ef = df, df_prueba
            self._sugerencias = sugerir_columnas(df, obj, df_prueba)
        finally:
            QApplication.restoreOverrideCursor()
        info = self._prep.info if self._prep is not None else {}
        claves = self._claves_forzadas
        cambios = self._config.cambios_columnas
        self._cargando = True
        self.tabla.setRowCount(len(self._sugerencias))
        for fila, sug in enumerate(self._sugerencias):
            if sug.tipo in ("fecha", "fecha_texto"):
                self._cols_fecha.append(sug.nombre)
            usar = sug.usar if sug.nombre not in cambios else (cambios[sug.nombre] == "usar")
            motivo = sug.motivo
            if not motivo and sug.nombre in info:
                motivo = info[sug.nombre]
            elif sug.nombre in claves and not sug.usar:
                motivo = ("Es la clave que une con otra tabla y sus datos ya se traen. Márcala si quieres "
                          "que el modelo vea el código como una categoría.")
            item = QTableWidgetItem("")
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if usar else Qt.Unchecked)
            self.tabla.setItem(fila, 0, item)
            self.tabla.setItem(fila, 1, QTableWidgetItem(sug.nombre))
            self.tabla.setItem(fila, 2, QTableWidgetItem(TIPOS_LEGIBLES.get(sug.tipo, sug.tipo)))
            self.tabla.setItem(fila, 3, QTableWidgetItem(motivo))
        self._cargando = False
        self._llenar_combos_dependientes()

    def _llenar_combos_dependientes(self):
        """Combos de fecha y de id, que dependen de las columnas disponibles."""
        c = self._config
        self._cargando = True
        self.combo_fecha.clear()
        self.combo_fecha.addItems(self._cols_fecha)
        if c.columna_fecha in self._cols_fecha:
            self.combo_fecha.setCurrentText(c.columna_fecha)
        self.combo_id.clear()
        self.combo_id.addItem(_AUTO)
        df_prueba = self._df_prueba()
        base = df_prueba if df_prueba is not None else self._df_entrenamiento()
        if base is not None:
            self.combo_id.addItems([str(col) for col in base.columns])
            if c.columna_id_salida in [str(col) for col in base.columns]:
                self.combo_id.setCurrentText(c.columna_id_salida)
        self.combo_grupo.clear()
        base_g = self._df_train_ef if self._df_train_ef is not None else self._df_entrenamiento()
        if base_g is not None:
            self.combo_grupo.addItems([str(col) for col in base_g.columns if str(col) != self._objetivo_actual()])
            if c.columna_grupo in [str(col) for col in base_g.columns]:
                self.combo_grupo.setCurrentText(c.columna_grupo)
        self._habilitar_validacion()
        self._cargando = False
        self._actualizar_botones()

    # ------------------------------------------------------------------
    # Cambios del usuario
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Tablas relacionadas
    # ------------------------------------------------------------------
    def _config_efectiva(self):
        """Copia de la configuración con lo que muestran los controles AHORA
        (sin modificar la configuración guardada)."""
        c = ConfigPrediccion.from_dict(self._config.to_dict())
        if self.combo_train.count():
            c.tabla_entrenamiento = self.combo_train.currentText()
        t = self.combo_test.currentText()
        c.tabla_prueba = None if t in ("", _NINGUNA) else t
        c.usar_relaciones = self.chk_rel.isChecked()
        c.profundidad_relaciones = self.spin_saltos.value()
        i = self.combo_fecha_rel.currentIndex()
        c.fecha_relaciones = None if i <= 0 else (SIN_FECHA if i == 1 else self.combo_fecha_rel.currentText())
        return c

    def _relaciones_ontologia(self):
        rel = getattr(self.host, "relaciones_ontologia", None)
        if rel is not None:
            return rel
        firma = self._firma()
        if self._rel_cache_firma != firma:
            try:
                from .ontologia import inferir_relaciones
                self._rel_cache = inferir_relaciones(self._tablas())
            except Exception:
                self._rel_cache = []
            self._rel_cache_firma = firma
        return self._rel_cache

    def _fechas_de_tabla_base(self):
        df = self._df_entrenamiento()
        if df is None:
            return []
        from .prediccion import tipo_columna
        return [str(c) for c in df.columns
                if tipo_columna(df[c].head(2000)) in ("fecha", "fecha_texto")]

    def _llenar_relaciones(self):
        """Actualiza la lista de relaciones, la fecha y los combos manuales."""
        tablas = self._tablas()
        activa = self.chk_rel.isChecked()
        self._cargando = True
        self.card_rel.setVisible(len(tablas) >= 2)
        self.panel_rel.setVisible(activa)
        self.tabla_enl.setRowCount(0)
        self._enlaces_lista = []
        if len(tablas) >= 2 and activa:
            cfg = self._config_efectiva()
            self._enlaces_lista = enlaces_disponibles(tablas, self._relaciones_ontologia(), cfg)
            desactivados = set(self._config.enlaces_desactivados or [])
            self.tabla_enl.setRowCount(len(self._enlaces_lista))
            for fila, e in enumerate(self._enlaces_lista):
                item = QTableWidgetItem("")
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked if e.clave() in desactivados else Qt.Checked)
                self.tabla_enl.setItem(fila, 0, item)
                texto = e.texto() + ("  (manual)" if e.manual else "")
                self.tabla_enl.setItem(fila, 1, QTableWidgetItem(texto))
            # fecha
            guardada = self._config.fecha_relaciones
            self.combo_fecha_rel.clear()
            self.combo_fecha_rel.addItems(["(automática)", "(ninguna: usar todo el historial)"])
            fechas = self._fechas_de_tabla_base()
            self.combo_fecha_rel.addItems(fechas)
            if guardada == SIN_FECHA:
                self.combo_fecha_rel.setCurrentIndex(1)
            elif guardada in fechas:
                self.combo_fecha_rel.setCurrentText(guardada)
            else:
                self.combo_fecha_rel.setCurrentIndex(0)
            self._llenar_manual()
        self._cargando = False

    def _llenar_manual(self):
        prueba = self._config_efectiva().tabla_prueba
        nombres = [n for n in self._tablas() if n != prueba]
        for combo_t, combo_c in ((self.combo_ma, self.combo_ca), (self.combo_mb, self.combo_cb)):
            previa_t, previa_c = combo_t.currentText(), combo_c.currentText()
            combo_t.clear()
            combo_t.addItems(nombres)
            if previa_t in nombres:
                combo_t.setCurrentText(previa_t)
            self._llenar_columnas_de(combo_t, combo_c, previa_c)

    def _llenar_columnas_de(self, combo_t, combo_c, previa=""):
        df = self._tablas().get(combo_t.currentText())
        combo_c.clear()
        if df is not None:
            cols = [str(c) for c in df.columns]
            combo_c.addItems(cols)
            if previa in cols:
                combo_c.setCurrentText(previa)

    def _calcular_efectivos(self):
        """Tablas de entrenamiento y prueba listas para el modelo: planas, o
        ampliadas con las tablas relacionadas si el usuario lo activó."""
        df = self._df_entrenamiento()
        df_prueba = self._df_prueba()
        self._prep = None
        self._claves_forzadas = set()
        self.lbl_rel_info.setText("")
        cfg = self._config_efectiva()
        if df is None or not cfg.usar_relaciones or len(self._tablas()) < 2:
            return df, df_prueba
        desactivados = set(cfg.enlaces_desactivados or [])
        activos = [e for e in self._enlaces_lista if e.clave() not in desactivados]
        if not activos:
            self.lbl_rel_info.setText("No hay ninguna relación activa: marca alguna de la lista o agrega una a mano.")
            return df, df_prueba
        try:
            prep = preparar(self._tablas(), cfg, self._relaciones_ontologia())
        except Exception as e:
            self.lbl_rel_info.setText("No se pudieron cruzar las tablas: " + str(e))
            return df, df_prueba
        self._prep = prep
        self._claves_forzadas = set(prep.claves_base)
        texto = f"Se sumaron {len(prep.columnas_nuevas)} columnas de tablas relacionadas."
        if prep.fecha_usada:
            texto += f" Los resúmenes usan solo lo anterior a la fecha «{prep.fecha_usada}» de cada fila."
        if prep.advertencias:
            texto += "\n⚠ " + "\n⚠ ".join(prep.advertencias)
        self.lbl_rel_info.setText(texto)
        return prep.df_entrenamiento, (prep.df_prueba if prep.df_prueba is not None else df_prueba)

    def _cambio_relaciones(self):
        if self._cargando:
            return
        self._leer_formulario()
        self._invalidar_resultado()
        self._reconstruir_columnas()

    def _cambio_enlace_casilla(self, item):
        if self._cargando or item.column() != 0 or item.row() >= len(self._enlaces_lista):
            return
        clave = self._enlaces_lista[item.row()].clave()
        desactivados = set(self._config.enlaces_desactivados or [])
        if item.checkState() == Qt.Checked:
            desactivados.discard(clave)
        else:
            desactivados.add(clave)
        self._config.enlaces_desactivados = sorted(desactivados)
        self._invalidar_resultado()
        self._reconstruir_columnas(refrescar_relaciones=False)

    def _agregar_enlace_manual(self):
        ta, ca = self.combo_ma.currentText(), self.combo_ca.currentText()
        tb, cb = self.combo_mb.currentText(), self.combo_cb.currentText()
        if not (ta and ca and tb and cb):
            return
        if ta == tb:
            QMessageBox.information(self, "Relación inválida", "Elige dos tablas distintas.")
            return
        nuevo = Enlace(ta, ca, tb, cb, True)
        if nuevo.clave() in {e.clave() for e in self._enlaces_lista}:
            QMessageBox.information(self, "Ya existe", "Esa relación ya está en la lista.")
            return
        self._leer_formulario()
        self._config.enlaces_manuales = list(self._config.enlaces_manuales or []) + [nuevo.a_dict()]
        self._config.enlaces_desactivados = [c for c in (self._config.enlaces_desactivados or []) if c != nuevo.clave()]
        self._invalidar_resultado()
        self._reconstruir_columnas()

    def _invalidar_resultado(self):
        self._resultado = None
        self.card_resultado.setVisible(False)
        self.lbl_estado.setText("")
        self._actualizar_botones()

    def _cambio_tabla(self, _i=None):
        if self._cargando:
            return
        self._leer_formulario()
        self._config.cambios_columnas = {}
        entrena = self.combo_train.currentText()
        previa = self._config.tabla_prueba          # lo que el usuario acaba de elegir
        otras = [n for n in self._tablas() if n != entrena]
        self._cargando = True
        self.combo_test.clear()
        self.combo_test.addItem(_NINGUNA)
        self.combo_test.addItems(otras)
        # Antes faltaba esta línea: al reconstruir la lista se perdía la elección
        # y la tabla de prueba volvía siempre a "(ninguna)".
        self.combo_test.setCurrentText(previa if previa in otras else _NINGUNA)
        self._cargando = False
        if self._config.objetivo not in [str(c) for c in self._tablas()[entrena].columns]:
            self._config.objetivo = None
        self._llenar_objetivo(entrena, self._config.objetivo)
        self._invalidar_resultado()
        self._reconstruir_columnas()

    def _cambio_objetivo(self, _i=None):
        if self._cargando:
            return
        self._leer_formulario()
        self._config.cambios_columnas = {}
        self._invalidar_resultado()
        self._reconstruir_columnas(refrescar_relaciones=False)

    def _cambio_opcion(self, *_args):
        if self._cargando:
            return
        self._leer_formulario()
        self._habilitar_validacion()
        self._invalidar_resultado()

    def _habilitar_validacion(self):
        modo = self.combo_valid.currentIndex()
        self.combo_fecha.setEnabled(modo == 1 and bool(self._cols_fecha))
        self.combo_grupo.setEnabled(modo == 2 and self.combo_grupo.count() > 0)

    def _cambio_casilla(self, item):
        if self._cargando or item.column() != 0:
            return
        sug = self._sugerencias[item.row()]
        marcada = item.checkState() == Qt.Checked
        if marcada == sug.usar:
            self._config.cambios_columnas.pop(sug.nombre, None)
        else:
            self._config.cambios_columnas[sug.nombre] = "usar" if marcada else "ignorar"
        self._invalidar_resultado()

    def _marcar_todas(self, valor):
        self._cargando = True
        for fila in range(self.tabla.rowCount()):
            self.tabla.item(fila, 0).setCheckState(Qt.Checked if valor else Qt.Unchecked)
        self._cargando = False
        self._config.cambios_columnas = {
            sug.nombre: ("usar" if valor else "ignorar")
            for sug in self._sugerencias if sug.usar != valor
        }
        self._invalidar_resultado()

    def _restablecer_columnas(self):
        self._config.cambios_columnas = {}
        self._invalidar_resultado()
        self._reconstruir_columnas()

    def _alternar_avanzado(self):
        visible = not self.card_avanzado.isVisible()
        self.card_avanzado.setVisible(visible)
        self.btn_avanzado.setText("Opciones avanzadas ▾" if visible else "Opciones avanzadas ▸")

    def _leer_formulario(self):
        """Pasa lo que muestra la pantalla a self._config."""
        c = self._config
        if self.combo_train.count():
            c.tabla_entrenamiento = self.combo_train.currentText()
        t = self.combo_test.currentText()
        c.tabla_prueba = None if t in ("", _NINGUNA) else t
        c.objetivo = self._objetivo_actual()
        c.tipo = ["auto", "clasificacion", "regresion"][max(self.combo_tipo.currentIndex(), 0)]
        c.log_objetivo = self.chk_log.isChecked()
        c.validacion = ["aleatoria", "temporal", "grupos"][max(self.combo_valid.currentIndex(), 0)]
        c.columna_fecha = self.combo_fecha.currentText() or None
        c.columna_grupo = self.combo_grupo.currentText() or None
        c.pliegues = self.spin_pliegues.value()
        c.balancear = ["auto", "si", "no"][max(self.combo_balancear.currentIndex(), 0)]
        c.salida = "probabilidad" if self.combo_salida.currentIndex() == 1 else "clase"
        idc = self.combo_id.currentText()
        c.columna_id_salida = None if idc in ("", _AUTO) else idc
        c.max_iter = self.spin_iter.value()
        c.tasa_aprendizaje = self.spin_tasa.value()
        c.profundidad_max = self.spin_prof.value()
        c.max_filas = self.spin_filas.value()
        c.semilla = self.spin_semilla.value()
        c.usar_relaciones = self.chk_rel.isChecked()
        c.profundidad_relaciones = self.spin_saltos.value()
        i = self.combo_fecha_rel.currentIndex()
        c.fecha_relaciones = None if i <= 0 else (SIN_FECHA if i == 1 else self.combo_fecha_rel.currentText())

    def _columnas_marcadas(self):
        return columnas_a_usar(self._sugerencias, self._config.cambios_columnas)

    def _actualizar_botones(self):
        entrenando = self._entrenando
        self.btn_entrenar.setEnabled(not entrenando and self._objetivo_actual() is not None)
        self.btn_predecir.setEnabled(
            not entrenando and self._resultado is not None and self._df_prueba() is not None)
        if self._resultado is not None and self._df_prueba() is None:
            self.btn_predecir.setToolTip("Elige una tabla de prueba (arriba) para poder predecir sobre ella.")
        else:
            self.btn_predecir.setToolTip("")

    # ------------------------------------------------------------------
    # Entrenar
    # ------------------------------------------------------------------
    def _entrenar(self):
        self._leer_formulario()
        df = self._df_train_ef if self._df_train_ef is not None else self._df_entrenamiento()
        if df is None or self._config.objetivo is None:
            QMessageBox.information(self, "Falta algo", "Elige la tabla y la columna que quieres predecir.")
            return
        columnas = self._columnas_marcadas()
        if not columnas:
            QMessageBox.information(self, "Sin columnas", "Marca al menos una columna para que el modelo se fije en ella.")
            return
        if self._config.validacion == "temporal" and not self._config.columna_fecha:
            QMessageBox.information(
                self, "Falta la fecha",
                "Para probar «por fecha» la tabla necesita una columna de fecha. "
                "Cambia a validación aleatoria o elige otra tabla.")
            return
        if self._config.validacion == "grupos" and not self._config.columna_grupo:
            QMessageBox.information(
                self, "Falta la columna de grupos",
                "Para probar «por grupos» elige la columna que identifica a la persona o cosa que se repite "
                "(por ejemplo, un código de cliente).")
            return
        self._resultado = None
        self.card_resultado.setVisible(False)
        self.lbl_estado.setText("Preparando…")
        self.barra.setVisible(True)
        self._entrenando = True
        forzar = {c for c in self._claves_forzadas if c in columnas}
        self._hilo = _HiloEntrenamiento(df, ConfigPrediccion.from_dict(self._config.to_dict()), columnas, forzar, self)
        self._hilo.progreso.connect(self.lbl_estado.setText)
        self._hilo.terminado.connect(self._fin_entrenamiento)
        self._hilo.fallo.connect(self._fallo_entrenamiento)
        self._hilo.finished.connect(self._hilo_terminado)
        self._actualizar_botones()
        self._hilo.start()

    def _hilo_terminado(self):
        self._entrenando = False
        self.barra.setVisible(False)
        self._actualizar_botones()

    def _fallo_entrenamiento(self, mensaje):
        self.lbl_estado.setText("")
        QMessageBox.warning(self, "No se pudo entrenar", mensaje)

    def _fin_entrenamiento(self, resultado):
        self._resultado = resultado
        self.lbl_estado.setText("Listo.")
        try:
            self._mostrar_resultado(resultado)
        except Exception as e:  # nunca dejar la pantalla sin resultado por un detalle de dibujo
            _vaciar(self.resultado_lay)
            texto = QLabel("\n".join(resultado.lineas_resumen()) + f"\n(No se pudo dibujar el detalle: {e})")
            texto.setWordWrap(True)
            self.resultado_lay.addWidget(texto)
        self.card_resultado.setVisible(True)
        self.lbl_pred.setText("" if self._df_prueba() is not None else
                              "Para predecir sobre otra tabla, elígela arriba como «Tabla de prueba».")
        self._actualizar_botones()

    # --- piezas del resultado ------------------------------------------
    def _seccion(self, titulo, explicacion=""):
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {REGLA}; border: none;")
        self.resultado_lay.addSpacing(8)
        self.resultado_lay.addWidget(sep)
        self.resultado_lay.addWidget(self._titulo_chico(titulo))
        if explicacion:
            e = QLabel(explicacion)
            e.setObjectName("muted")
            e.setWordWrap(True)
            self.resultado_lay.addWidget(e)

    def _grilla_puntos(self, filas):
        """filas: [(etiqueta, porcentaje, color, texto)] -> etiqueta · puntos · valor."""
        cont = QWidget()
        g = QGridLayout(cont)
        g.setContentsMargins(0, 2, 0, 2)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(4)
        g.setColumnMinimumWidth(0, 190)
        g.setColumnStretch(1, 1)
        for i, (etiqueta, pct, color, valor) in enumerate(filas):
            corta = etiqueta if len(etiqueta) <= 34 else etiqueta[:33] + "…"
            lbl = QLabel(corta)
            lbl.setToolTip(etiqueta)
            g.addWidget(lbl, i, 0)
            g.addWidget(_PuntoLinea(pct, color), i, 1, Qt.AlignVCenter)
            v = QLabel(valor)
            v.setMinimumWidth(60)
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            g.addWidget(v, i, 2)
        self.resultado_lay.addWidget(cont)

    def _celda_punto(self, widget):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 0, 6, 0)
        h.addWidget(widget, 1, Qt.AlignVCenter)
        return w

    def _mostrar_resultado(self, r):
        lay = self.resultado_lay
        _vaciar(lay)
        nivel, titulo, frase = r.veredicto()
        color = COLOR_NIVEL.get(nivel, GRIS)

        # 1) el veredicto es una frase: solo la primera palabra lleva color
        veredicto = QLabel(f'<span style="color:{color}; font-weight:600;">{html.escape(titulo)}.</span> '
                           f'{html.escape(frase)}')
        veredicto.setTextFormat(Qt.RichText)
        veredicto.setWordWrap(True)
        veredicto.setStyleSheet(f"font-size: 17px; font-family: {FUENTE_SERIF};")
        lay.addWidget(veredicto)
        for aviso in r.advertencias:
            a = QLabel("Nota: " + aviso)
            a.setWordWrap(True)
            a.setStyleSheet(f"color: {AMBAR};")
            lay.addWidget(a)

        # 2) tres cifras: tipografía y nada más
        cifras = QWidget()
        g = QGridLayout(cifras)
        g.setContentsMargins(0, 8, 0, 4)
        g.setHorizontalSpacing(28)
        g.setVerticalSpacing(1)
        for col, (valor, tit, expl) in enumerate(r.tarjetas()):
            n = QLabel(valor)
            n.setStyleSheet(f"font-size: 32px; font-family: {FUENTE_SERIF};")
            b = QLabel(tit)
            b.setStyleSheet("font-weight: 600;")
            e = QLabel(expl)
            e.setObjectName("muted")
            e.setWordWrap(True)
            g.addWidget(n, 0, col)
            g.addWidget(b, 1, col)
            g.addWidget(e, 2, col, Qt.AlignTop)
            g.setColumnStretch(col, 1)
        lay.addWidget(cifras)

        # 3) el modelo frente a la respuesta fácil: dos puntos, un eje
        m, base = r.metricas, r.baseline
        if r.tipo == "regresion":
            self._seccion("El modelo frente a adivinar el promedio", "Más a la izquierda = menos error.")
            rb = max(base.get("rmse", 0.0), 1e-12)
            mej = r.mejora()
            brecha = f"{(1 - m['rmse'] / rb) * 100:.0f}% menos error" if m["rmse"] < rb else "sin mejora"
            lay.addWidget(_Mancuerna(m["rmse"], rb, max(m["rmse"], rb),
                                     f"con el modelo  {_formatear(m['rmse'])}",
                                     f"respondiendo el promedio  {_formatear(rb)}", brecha))
        elif r.desbalanceado():
            self._seccion("El modelo frente a la respuesta fácil",
                          "Con clases muy desparejas, acertar en general no mide nada: se compara cuánto acierta "
                          "en cada clase por igual. Más a la derecha = mejor.")
            bal, fb = m["bal_acc"] * 100, base["bal_acc"] * 100
            lay.addWidget(_Mancuerna(bal, fb, 100,
                                     f"modelo  {bal:.0f}%", f"siempre la clase más frecuente  {fb:.0f}%",
                                     f"+{bal - fb:.0f} puntos" if bal > fb else "sin mejora"))
        else:
            self._seccion("El modelo frente a la respuesta fácil", "Más a la derecha = más aciertos.")
            acc, fac = m["accuracy"] * 100, base["accuracy"] * 100
            lay.addWidget(_Mancuerna(acc, fac, 100,
                                     f"modelo  {acc:.0f}%", f"siempre la clase más frecuente  {fac:.0f}%",
                                     f"+{acc - fac:.0f} puntos" if acc > fac else "sin mejora"))
        ref = r.texto_lineal()
        if ref:
            t = QLabel(ref)
            t.setObjectName("muted")
            t.setWordWrap(True)
            lay.addWidget(t)

        # 4) en qué columnas se fijó: gráfico de puntos ordenado
        if r.importancias:
            self._seccion("En qué columnas se fijó más", "Cuánto ayudó cada columna a adivinar. Todas suman 100%.")
            self._grilla_puntos([(str(c), pct, ACENTO, "<1%" if pct < 1 else f"{pct:.0f}%")
                                 for c, pct in r.importancias[:8]])

        # 5) lo predicho frente a lo real
        if r.tipo == "regresion" and r.validacion is not None and pg is not None:
            self._grafico_predicho_real(r)

        # 6) casos para revisar primero
        self._lista_revision(r)

        info = QLabel(r.texto_validacion())
        info.setObjectName("muted")
        info.setWordWrap(True)
        lay.addSpacing(6)
        lay.addWidget(info)

    def _grafico_predicho_real(self, r):
        try:
            import numpy as np
            reales, preds = r.validacion
            self._seccion("Lo predicho frente a lo real",
                          "Cada punto es una fila de la parte de prueba. Mientras más cerca de la línea, mejor acertó.")
            plot = _GraficoEstatico(background=self._c("bg", "#000000"))
            plot.setMinimumHeight(260)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.showGrid(x=False, y=False)
            plot.setLabel("bottom", "valor real")
            plot.setLabel("left", "valor predicho")
            for eje in ("bottom", "left"):
                ax = plot.getAxis(eje)
                ax.setPen(pg.mkPen(GRIS, width=1))
                ax.setTextPen(pg.mkPen("#a1a1aa"))
            lo = float(min(np.min(reales), np.min(preds)))
            hi = float(max(np.max(reales), np.max(preds)))
            plot.setXRange(lo, hi, padding=0.03)       # los ejes cubren solo el rango de los datos
            plot.setYRange(lo, hi, padding=0.03)
            plot.addItem(pg.PlotCurveItem([lo, hi], [lo, hi], pen=pg.mkPen("#a1a1aa", width=1)))
            plot.addItem(pg.ScatterPlotItem(x=reales, y=preds, size=4, pen=pg.mkPen(None),
                                            brush=pg.mkBrush(99, 102, 241, 130)))
            nota = pg.TextItem("acierto perfecto", color="#a1a1aa", anchor=(1, 1))
            nota.setPos(hi, hi)                         # la línea se rotula ella misma, sin leyenda
            plot.addItem(nota)
            self.resultado_lay.addWidget(plot)
        except Exception:
            pass   # el gráfico es un extra: si falla, el resto del resultado se muestra igual

    def _c(self, clave, defecto):
        try:
            v = self.colors[clave]
            return v if v else defecto
        except Exception:
            return defecto

    def _lista_revision(self, r):
        rev = r.revision
        if not rev or len(rev["pos"]) == 0:
            return
        df = self._df_train_ef if self._df_train_ef is not None else self._df_entrenamiento()
        idc = elegir_columna_id(df) if df is not None else None
        pos = [int(p) for p in rev["pos"]]
        if idc is not None and idc in df.columns:
            etiquetas = [str(df.iloc[p][idc]) for p in pos]
            nombre_fila = str(idc)
        else:
            etiquetas = [str(p + 1) for p in pos]
            nombre_fila = "Fila n.º"

        def texto(v):
            try:
                return _formatear(float(v)) if not isinstance(v, str) else v
            except Exception:
                return str(v)

        regresion = rev["tipo"] == "regresion"
        extra = [float(x) for x in rev["extra"]]
        if regresion:
            self._seccion("Casos para revisar primero",
                          "Las filas donde el modelo más se equivocó (en la parte de prueba). A la derecha de la marca: el "
                          "valor real fue mayor de lo esperado; a la izquierda: menor. Pueden ser errores de registro, "
                          "casos especiales u oportunidades: tú decides.")
            titulos = [nombre_fila, "Real", "Predicho", "Diferencia", ""]
            escala = max(max(abs(x) for x in extra), 1e-12)
            self._df_revision = {
                nombre_fila: etiquetas, "real": [float(x) for x in rev["real"]],
                "predicho": [float(x) for x in rev["pred"]], "diferencia": extra}
        else:
            self._seccion("Casos para revisar primero",
                          "Filas donde el modelo se equivocó estando muy seguro. Conviene mirar si el dato real está "
                          "bien registrado o si es un caso especial.")
            titulos = [nombre_fila, "Real", "Predijo", "Seguridad del modelo", ""]
            escala = 1.0
            self._df_revision = {
                nombre_fila: etiquetas, "real": [str(x) for x in rev["real"]],
                "predijo": [str(x) for x in rev["pred"]], "seguridad_del_modelo": extra}

        n = min(len(pos), 15)
        tabla = QTableWidget(n, 5)
        tabla.setStyleSheet(_QSS_TABLA)
        tabla.setShowGrid(False)
        tabla.setHorizontalHeaderLabels(titulos)
        tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tabla.setSelectionMode(QAbstractItemView.NoSelection)
        tabla.verticalHeader().setVisible(False)
        tabla.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cab = tabla.horizontalHeader()
        for c in range(4):
            cab.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        cab.setSectionResizeMode(4, QHeaderView.Stretch)
        for i in range(n):
            tabla.setRowHeight(i, 26)
            tabla.setItem(i, 0, QTableWidgetItem(etiquetas[i]))
            tabla.setItem(i, 1, QTableWidgetItem(texto(rev["real"][i])))
            tabla.setItem(i, 2, QTableWidgetItem(texto(rev["pred"][i])))
            if regresion:
                d = extra[i]
                color = VERDE if d > 0 else ROJO
                it = QTableWidgetItem(("+" if d > 0 else "−") + _formatear(abs(d)))
                it.setForeground(QColor(color))
                tabla.setItem(i, 3, it)
                tabla.setCellWidget(i, 4, self._celda_punto(_PuntoDivergente(d / escala, color)))
            else:
                tabla.setItem(i, 3, QTableWidgetItem(f"{extra[i] * 100:.0f}%"))
                tabla.setCellWidget(i, 4, self._celda_punto(_PuntoLinea(extra[i] * 100, AMBAR)))
        tabla.setFixedHeight(26 * n + 32)
        self.resultado_lay.addWidget(tabla)
        btn = QPushButton("Guardar lista de revisión (CSV)")
        btn.clicked.connect(self._guardar_revision)
        self.resultado_lay.addWidget(btn, alignment=Qt.AlignLeft)

    def _guardar_revision(self):
        if not self._df_revision:
            return
        import os
        import pandas as pd
        carpeta = ""
        try:
            from .proyecto import ruta_carpeta_proyectos
            carpeta = ruta_carpeta_proyectos()
        except Exception:
            pass
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar lista de revisión", os.path.join(carpeta, "lista_de_revision.csv"), "CSV (*.csv)")
        if not ruta:
            return
        if not ruta.lower().endswith(".csv"):
            ruta += ".csv"
        try:
            pd.DataFrame(self._df_revision).to_csv(ruta, index=False, encoding="utf-8-sig")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
            return
        QMessageBox.information(self, "Lista guardada", f"Guardada en:\n{ruta}")

    # ------------------------------------------------------------------
    # Predecir y guardar
    # ------------------------------------------------------------------
    def _generar_predicciones(self):
        df = self._df_test_ef if self._df_test_ef is not None else self._df_prueba()
        if self._resultado is None or df is None:
            return
        self._leer_formulario()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            salida = predecir(self._resultado, df, self._config.columna_id_salida, self._config.salida)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "No se pudo predecir", str(e))
            return
        QApplication.restoreOverrideCursor()
        import os
        carpeta = ""
        try:
            from .proyecto import ruta_carpeta_proyectos
            carpeta = ruta_carpeta_proyectos()
        except Exception:
            pass
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar predicciones", os.path.join(carpeta, "predicciones.csv"), "CSV (*.csv)")
        if not ruta:
            return
        if not ruta.lower().endswith(".csv"):
            ruta += ".csv"
        try:
            salida.to_csv(ruta, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
            return
        self.lbl_pred.setText(f"Guardado: {ruta} ({len(salida):,} filas)")
        QMessageBox.information(self, "Predicciones guardadas", f"{len(salida):,} filas guardadas en:\n{ruta}")