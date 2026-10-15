"""
linea_tiempo_ui.py — Widget de la pestaña "Línea de Tiempo" de Hadar.

100% PySide6 + PyQtGraph (sin QtWebEngine, sin librerías nuevas), mismo
lenguaje visual que el resto de la app (self.colors, COLOR_ACCENT, etc.).

Ya no existe un "Modo Intervalo (Gantt)" separado: en su lugar, cualquier
par de fechas (automático, sugerido por sugerir_pares_temporales, o
manual, punto a punto) se dibuja como una LÍNEA entre los dos puntos ya
mostrados en Modo Hito. Un clic en la línea muestra cuánto tiempo pasó
entre ambos, en lenguaje natural.

Modelo de recálculo (decidido por el usuario): al arrastrar el slider de
rango, se recalculan en vivo Métricas/Indicadores/Frecuencias/Alarmas
(son baratos, ya es lo único que hace apply_table_filter()) y también la
vista de puntos de abajo, que muestra solo lo que cae en el período
recortado (el histograma de arriba siempre muestra el total). Las
anomalías NO se recalculan solas -- para eso hay un botón explícito
"Actualizar anomalías en este rango", porque correr
SemanticAnomalyDetector en cada pixel de arrastre puede sentirse pegado
en equipos modestos con datasets grandes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer, QPoint, Signal
from PySide6.QtGui import QColor, QPen, QCursor, QPixmap, QPainter, QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QGraphicsLineItem, QDialog, QDialogButtonBox, QMessageBox, QToolButton,
    QMenu, QFrame, QScrollArea, QCheckBox,
)

from .config import COLOR_ACCENT, COLOR_DANGER, DONUT_PALETTE
from .linea_tiempo_futuro import dibujar_ventanas, quitar_ventanas
from .anomalias import detectar_columnas_fecha, SemanticAnomalyDetector
from .linea_tiempo import (
    sugerir_pares_temporales, construir_serie_hitos, construir_intervalos,
    construir_histograma, anomalias_con_fecha, tiene_componente_horario,
    minutos_desde_medianoche,
    formatear_duracion, resumen_duraciones,
    detectar_formato_fecha, parsear_fechas, FormatoFecha,
)

# Debounce del slider: recalcula Métricas/Indicadores como máximo cada
# tantos ms mientras se arrastra, en vez de en cada micro-movimiento del
# mouse. Sigue sintiéndose "en vivo" pero no satura la CPU.
_DEBOUNCE_SLIDER_MS = 120

# Color neutro para conexiones manuales (no confundirlas con los colores
# por columna ni por par automático).
_COLOR_CONEXION_MANUAL = "#C9C9D6"

# Radio (en píxeles de pantalla, no en unidades de dato) dentro del cual un
# clic "acierta" un punto o una línea. En píxeles porque el zoom cambia
# cuántas unidades de dato caben en un píxel, pero el dedo/mouse no.
_TOLERANCIA_PUNTO_PX = 12
_TOLERANCIA_LINEA_PX = 6

# Estilo de "subpestaña" (texto sin caja, con línea inferior índigo cuando está
# activo): el mismo de Preguntas / Entrenar un modelo en Predicción. También
# existe como QSS_SUBPESTANA en main_window.py; si se cambia uno, cambiar el otro.
QSS_SUBPESTANA = (
    "QPushButton{background: transparent; border: none; border-bottom: 2px solid transparent; "
    "border-radius: 0; padding: 6px 14px; font-weight: 600; color: #a1a1aa;} "
    "QPushButton:checked{color: #6366f1; border-bottom: 2px solid #6366f1;} "
    "QPushButton:hover{background: transparent; color: #6366f1;} "
    "QPushButton:pressed{background: transparent;} "
    "QPushButton:disabled{color: rgba(127,127,127,0.6);}"
)


def _epoch_a_timestamp(valor_epoch_seg):
    return pd.Timestamp(valor_epoch_seg, unit="s")


def _timestamp_a_epoch(ts):
    return ts.value / 1e9  # ns -> s, formato que espera pg (eje en segundos)


class _MenuMultiCheck(QMenu):
    """QMenu normal, pero un clic en una opción marcable NO cierra el menú
    -- así se pueden tildar 'fecha_envio', 'fecha_entrega' y una tercera
    sin tener que reabrir el botón cada vez. Es el truco estándar de Qt
    para esto (nada de PySide6 nuevo, solo interceptar el clic)."""

    def mouseReleaseEvent(self, event):
        accion = self.activeAction()
        if accion is not None and accion.isCheckable():
            accion.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _PopupFlotante(QFrame):
    """Recuadro oscuro que aparece al hacer clic en un punto o una línea.

    Es un widget HIJO del panel de la Línea de Tiempo, no una ventana
    independiente. Antes era una ventana de tipo ToolTip: como cualquier
    ventana de nivel superior, no sabe nada de las pestañas, así que al
    cambiar a Indicadores (o a otra aplicación) quedaba flotando encima.
    Siendo hijo del panel, se oculta solo cuando el panel deja de verse."""

    _ESTILO = (
        "#popupHadar { background-color: #2B2B36; border: 1px solid #46465A; border-radius: 6px; }"
        "#popupHadar QLabel { color: #F0F0F0; background: transparent; border: none; }"
        "#popupHadar QScrollArea, #popupHadar QScrollArea > QWidget > QWidget "
        "{ background: transparent; border: none; }"
    )

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("popupHadar")
        self.setStyleSheet(self._ESTILO)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(9, 7, 9, 7)

        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.RichText)
        # Con muchas columnas la fila no cabe en el alto del panel: el
        # texto va dentro de un área con scroll que solo aparece si hace falta.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(self._label)
        self._layout.addWidget(self._scroll)
        self.hide()

    def _colocar(self, posicion_global):
        """Junto al cursor, pero siempre dentro del panel: si no cabe a la
        derecha o abajo, se abre hacia el otro lado (así nunca tapa el
        punto donde se hizo clic)."""
        padre = self.parentWidget()
        self.setMaximumSize(max(padre.width() - 8, 120), max(padre.height() - 8, 120))
        self.adjustSize()
        cursor = padre.mapFromGlobal(posicion_global)
        x = cursor.x() + 16
        if x + self.width() > padre.width():
            x = cursor.x() - 16 - self.width()
        y = cursor.y() + 16
        if y + self.height() > padre.height():
            y = cursor.y() - 16 - self.height()
        self.move(max(x, 0), max(y, 0))
        self.show()
        self.raise_()

    def ocultar(self):
        self.hide()


class _PopupInfoFila(_PopupFlotante):
    """Info de la fila del punto clicado. Se abre con el clic y se queda
    hasta el próximo clic (sobre otro punto, o sobre el vacío del gráfico)
    -- un QToolTip nativo se cerraba solo con el primer micro-movimiento."""

    def mostrar(self, texto, posicion_global):
        self._label.setText(texto)
        self._colocar(posicion_global)


class _PopupConexion(_PopupFlotante):
    """Mismo look, pero para el clic sobre una LÍNEA de conexión: muestra la
    duración entre los dos puntos, y si la conexión es manual, un botón
    para eliminarla (las automáticas no se eliminan una por una -- se
    destildan desde 'Unir fechas')."""

    def __init__(self, parent):
        super().__init__(parent)
        self._btn_eliminar = QPushButton("Eliminar conexión  (Supr)")
        self._btn_eliminar.setStyleSheet(
            "background-color: #46465A; color: #F0F0F0; border: none; "
            "border-radius: 4px; padding: 4px 8px;"
        )
        self._layout.addWidget(self._btn_eliminar)
        self._btn_eliminar.hide()

    def mostrar(self, texto, posicion_global, on_eliminar=None):
        self._label.setText(texto)
        try:
            self._btn_eliminar.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        if on_eliminar is not None:
            self._btn_eliminar.setVisible(True)
            self._btn_eliminar.clicked.connect(on_eliminar)
        else:
            self._btn_eliminar.setVisible(False)
        self._colocar(posicion_global)


class _TarjetaAnomalia(QDialog):
    """Diálogo simple con el detalle de una anomalía al hacer clic en su
    marcador, y el botón para mandarla al lienzo de Reporte."""

    def __init__(self, anomalia, colors, on_enviar_reporte, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detalle de la anomalía")
        self.resize(420, 260)
        layout = QVBoxLayout(self)

        descripcion = QLabel(anomalia.get("descripcion", ""))
        descripcion.setWordWrap(True)
        descripcion.setObjectName("title")
        layout.addWidget(descripcion)

        gravedad = QLabel(f"Gravedad: {anomalia.get('gravedad', 'media')}")
        gravedad.setObjectName("muted")
        layout.addWidget(gravedad)

        frase_memoria = anomalia.get("frase_memoria")
        if frase_memoria:
            lbl = QLabel(frase_memoria)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        frase_impacto = anomalia.get("frase_impacto_acumulado")
        if frase_impacto:
            lbl2 = QLabel(frase_impacto)
            lbl2.setWordWrap(True)
            layout.addWidget(lbl2)

        layout.addStretch()

        botones = QHBoxLayout()
        btn_reporte = QPushButton("Enviar a Reporte")
        btn_reporte.setObjectName("accentButton")
        btn_reporte.clicked.connect(lambda: (on_enviar_reporte(anomalia), self.accept()))
        botones.addWidget(btn_reporte)
        botones.addStretch()
        cerrar = QDialogButtonBox(QDialogButtonBox.Close)
        cerrar.rejected.connect(self.reject)
        cerrar.clicked.connect(self.reject)
        botones.addWidget(cerrar)
        layout.addLayout(botones)


class _DialogoMetricasTiempo(QDialog):
    """Promedio/mínimo/máximo de cada conexión activa (automática o el
    conjunto de las manuales). Se manda a Reporte como bloque de texto
    ('lista'), no como imagen -- son solo números, igual que ya hace
    Frecuencias."""

    def __init__(self, resumenes, on_enviar_reporte, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Métricas de tiempo")
        self.resize(440, 340)
        layout = QVBoxLayout(self)

        html_partes = []
        for etiqueta, resumen in resumenes:
            aviso_incoherentes = (
                f"<br>Atención: {resumen['n_incoherentes']} con orden invertido"
                if resumen["n_incoherentes"] else ""
            )
            bloque = QLabel(
                f"<b>{etiqueta}</b> ({resumen['n']} par(es))<br>"
                f"Promedio: {resumen['promedio']}<br>"
                f"Mínimo: {resumen['minimo']}<br>"
                f"Máximo: {resumen['maximo']}{aviso_incoherentes}"
            )
            bloque.setWordWrap(True)
            bloque.setObjectName("card")
            layout.addWidget(bloque)
            html_partes.append(
                f"<b>{etiqueta}</b> ({resumen['n']} par(es))<br>"
                f"Promedio: {resumen['promedio']} · Mínimo: {resumen['minimo']} · "
                f"Máximo: {resumen['maximo']}{aviso_incoherentes}"
            )

        layout.addStretch()

        botones = QHBoxLayout()
        btn_reporte = QPushButton("Enviar a Reporte")
        btn_reporte.setObjectName("accentButton")
        html_final = "<br><br>".join(html_partes)
        btn_reporte.clicked.connect(
            lambda: (on_enviar_reporte("Métricas de tiempo — Línea de Tiempo", html_final), self.accept())
        )
        botones.addWidget(btn_reporte)
        botones.addStretch()
        cerrar = QDialogButtonBox(QDialogButtonBox.Close)
        cerrar.rejected.connect(self.reject)
        cerrar.clicked.connect(self.reject)
        botones.addWidget(cerrar)
        layout.addLayout(botones)


class LineaTiempoPanel(QWidget):
    """
    host: la ventana principal (HadarApp). Se usa para leer host.df /
    host.filtered_df / host.colors / host.memoria / host.fingerprint_actual,
    y para avisarle el rango de tiempo activo (host.rango_tiempo) llamando
    a host.apply_table_filter() -- así Métricas/Indicadores/Frecuencias/
    Alarmas se refrescan solos, sin que este widget sepa nada de ellos.
    """

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.colors = host.colors

        self._columna_fecha = None
        # Cómo leer las fechas escritas como texto. "auto" (por defecto)
        # mira los datos de CADA columna y decide entre AAAA/MM/DD,
        # DD/MM/AAAA y MM/DD/AAAA (ver detectar_formato_fecha); "dmy",
        # "mdy" o "ymd" fuerzan uno solo para todas las columnas, por si la
        # detección se equivoca o el archivo es genuinamente ambiguo.
        self._modo_formato = "auto"
        self._formatos_detectados = {}   # {(tabla_o_None, columna): FormatoFecha}, cache por dataset
        self._columnas_mostrar = set()   # columnas de fecha tildadas en "Fechas a mostrar"
        self._color_por_columna = {}     # {columna: color}, estable mientras no cambien de dataset
        self._marcadores_anomalias = []  # [(scatter_item, anomalia_dict)]
        self._actualizando_desde_slider = False  # ver _aplicar_rango_activo

        # -- Conexiones entre puntos (reemplaza al extinto Modo Gantt) --
        self._pares_disponibles = {}  # {(col_inicio,col_fin): ParTemporal}
        self._pares_conectar = set()  # claves tildadas en "Unir fechas"
        self._color_por_par = {}      # {(col_inicio,col_fin): color}
        self._conexiones_manuales = []  # [{"desde":(indice,columna), "hasta":(indice,columna)}]
        self._modo_conectar_manual = False
        self._primer_punto_manual = None
        self._posicion_punto = {}     # {(indice,columna): (x,y)} -- para ubicar los extremos de una línea
        self._lineas_conexion = []    # [{"item":QGraphicsLineItem,"desde":...,"hasta":...,"duracion":...,"tipo":...,"etiqueta":...,"p_ini":...,"p_fin":...,"color":...,"estilo":...}]
        self._segmentos = None        # (x0, y0, x1, y1) en arrays numpy, paralelos a _lineas_conexion -- para acertar clics en líneas
        self._conexion_seleccionada = None   # la línea resaltada (Supr la borra si es manual)
        self._cruces_anomalias = []   # todas las anomalías calculadas con su fecha; se dibujan las que caen en el rango activo
        self._total_fechas = 0        # fechas válidas de la columna de referencia SIN recortar por el slider (para el texto de estado)

        self._timer_debounce = QTimer(self)
        self._timer_debounce.setSingleShot(True)
        self._timer_debounce.setInterval(_DEBOUNCE_SLIDER_MS)
        self._timer_debounce.timeout.connect(self._aplicar_rango_activo)

        self._construir_ui()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _construir_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Filtrar por:"))
        self.combo_fecha = QComboBox()
        self.combo_fecha.setToolTip(
            "Esta es la columna que usa el slider de rango para recortar "
            "Métricas/Indicadores/Frecuencias/Alarmas."
        )
        # Sin este límite, el combo se ensancha solo según el nombre de
        # columna más largo que le carguen -- con columnas reales eso
        # empujaba "Métricas de tiempo" y "Actualizar" (al final de la
        # fila, después del addStretch) fuera del borde de la ventana.
        self.combo_fecha.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_fecha.setMinimumContentsLength(14)
        self.combo_fecha.setMaximumWidth(180)
        self.combo_fecha.currentTextChanged.connect(self._on_columna_fecha_cambiada)
        toolbar.addWidget(self.combo_fecha)

        toolbar.addWidget(QLabel("Formato:"))
        self.combo_formato_fecha = QComboBox()
        self.combo_formato_fecha.addItem("Automático", "auto")
        self.combo_formato_fecha.addItem("DD/MM/AAAA (Chile)", "dmy")
        self.combo_formato_fecha.addItem("MM/DD/AAAA (EE.UU.)", "mdy")
        self.combo_formato_fecha.addItem("AAAA/MM/DD (año primero)", "ymd")
        self.combo_formato_fecha.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_formato_fecha.setToolTip(
            "Cómo leer las fechas escritas como texto. 'Automático' mira los "
            "datos de cada columna y decide solo; entre paréntesis te dice "
            "cuál eligió. Solo hace falta cambiarlo a mano si la detección "
            "se equivoca o si todas las fechas caben en más de un orden "
            "(ej. 03/04/2024 puede ser 3 de abril o 4 de marzo)."
        )
        self.combo_formato_fecha.currentIndexChanged.connect(self._on_formato_fecha_cambiado)
        toolbar.addWidget(self.combo_formato_fecha)

        self.btn_columnas_mostrar = QToolButton()
        self.btn_columnas_mostrar.setText("Fechas a mostrar")
        self.btn_columnas_mostrar.setToolTip(
            "Si tu dataset tiene más de una fecha (ej. envío/entrega, "
            "última mantención/próxima mantención), marca las que quieras "
            "ver juntas en el gráfico -- cada una queda en un color distinto. "
            "Si hay otras tablas relacionadas con fechas propias, también "
            "aparecen acá, con un símbolo cuadrado en vez de circular."
        )
        self.btn_columnas_mostrar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_columnas_mostrar = _MenuMultiCheck(self.btn_columnas_mostrar)
        self.btn_columnas_mostrar.setMenu(self.menu_columnas_mostrar)
        toolbar.addWidget(self.btn_columnas_mostrar)

        # «Lo que se viene»: dibuja sobre el eje de fechas las ventanas de tiempo que predicen los
        # pronósticos que se llevaron al Panel (agotamiento, umbral, saturación, vida útil).
        self._items_futuro = []
        self.chk_futuro = QCheckBox("Mostrar lo que se viene")
        self.chk_futuro.setToolTip(
            "Marca en la línea de tiempo las fechas futuras que predicen los pronósticos que llevaste al Panel "
            "(desde Predicción > Preguntas > «Llevar al Panel»). Franja = entre dos fechas; línea punteada = la "
            "fecha más probable. Son estimaciones, no datos.")
        self.chk_futuro.setEnabled(False)
        self.chk_futuro.toggled.connect(lambda _=False: self._dibujar_futuro())
        toolbar.addWidget(self.chk_futuro)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Segunda fila: la primera ya reúne 3 controles más el combo que se
        # ensancha solo -- separarlas deja margen de sobra en pantallas
        # angostas, en vez de que todo compita en una sola fila kilométrica.
        toolbar2 = QHBoxLayout()
        self.btn_conectar = QToolButton()
        self.btn_conectar.setText("Unir fechas")
        self.btn_conectar.setToolTip(
            "Marca un par de fechas (ej. envío→entrega) para unir con una "
            "línea cada fila donde ambas existan. Los pares vienen de lo "
            "mismo que 'Configurar relaciones temporales' guarda (al final "
            "de este mismo menú)."
        )
        self.btn_conectar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_conectar = _MenuMultiCheck(self.btn_conectar)
        self.btn_conectar.setMenu(self.menu_conectar)
        toolbar2.addWidget(self.btn_conectar)

        self.btn_conectar_manual = QPushButton("Conectar puntos")
        self.btn_conectar_manual.setCheckable(True)
        self.btn_conectar_manual.setToolTip(
            "Actívalo y haz clic en dos puntos cualquiera (de la misma fila "
            "o no) para unirlos con una línea propia. Vuelve a apretarlo "
            "para salir del modo."
        )
        self.btn_conectar_manual.toggled.connect(self._on_toggle_conectar_manual)
        self.btn_conectar_manual.setCursor(Qt.PointingHandCursor)
        self.btn_conectar_manual.setStyleSheet(QSS_SUBPESTANA)
        toolbar2.addWidget(self.btn_conectar_manual)

        toolbar2.addStretch()

        self.btn_metricas_tiempo = QPushButton("Métricas de tiempo")
        self.btn_metricas_tiempo.clicked.connect(self._abrir_metricas_tiempo)
        self.btn_metricas_tiempo.setCursor(Qt.PointingHandCursor)
        self.btn_metricas_tiempo.setStyleSheet(QSS_SUBPESTANA)
        toolbar2.addWidget(self.btn_metricas_tiempo)

        btn_actualizar_vista = QPushButton("Actualizar")
        btn_actualizar_vista.setToolTip(
            "Actualizar la Línea de Tiempo con los filtros actuales de Datos "
            "(normalmente se actualiza sola; este botón es por si acaso)."
        )
        btn_actualizar_vista.clicked.connect(self.refrescar_vista)
        btn_actualizar_vista.setCursor(Qt.PointingHandCursor)
        btn_actualizar_vista.setStyleSheet(QSS_SUBPESTANA)
        toolbar2.addWidget(btn_actualizar_vista)

        layout.addLayout(toolbar2)

        self.aviso = QLabel(
            "Arrastra los bordes del área sombreada para recortar un período: "
            "el gráfico de abajo, Métricas, Indicadores, Frecuencias y Alarmas "
            "se actualizan solos. Un clic en un punto muestra su fila; un clic "
            "en una línea muestra cuánto tiempo pasó entre sus dos puntos (si "
            "es manual, Supr la borra). Un clic en el vacío cierra el recuadro."
        )
        self.aviso.setObjectName("muted")
        self.aviso.setWordWrap(True)
        layout.addWidget(self.aviso)

        # Leyenda de colores por columna (solo se llena cuando hay 2+
        # columnas marcadas; ver _actualizar_leyenda).
        self.fila_leyenda = QHBoxLayout()
        layout.addLayout(self.fila_leyenda)

        # --- Histograma de densidad + slider de rango ---
        self.plot_histograma = pg.PlotWidget()
        self.plot_histograma.setMaximumHeight(140)
        self.plot_histograma.setBackground(self.colors["card"])
        self.plot_histograma.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_histograma.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_histograma.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_histograma.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_histograma.getAxis("bottom").setStyle(showValues=True)
        eje_fecha = pg.DateAxisItem(orientation="bottom")
        self.plot_histograma.setAxisItems({"bottom": eje_fecha})
        self.plot_histograma.showGrid(x=False, y=False)
        self._barra_histograma = None
        layout.addWidget(self.plot_histograma)

        self.region_rango = pg.LinearRegionItem(brush=pg.mkBrush(QColor(0, 0, 0, 0)))
        self.region_rango.setZValue(10)
        self.plot_histograma.addItem(self.region_rango)
        self.region_rango.sigRegionChanged.connect(self._on_region_cambiada)

        # --- Vista principal: puntos + líneas de conexión + anomalías ---
        self.plot_principal = pg.PlotWidget()
        self.plot_principal.setBackground(self.colors["card"])
        eje_fecha_2 = pg.DateAxisItem(orientation="bottom")
        self.plot_principal.setAxisItems({"bottom": eje_fecha_2})
        self.plot_principal.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_principal.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_principal.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_principal.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_principal.showGrid(x=False, y=False)
        # El eje Y se decide solo, al dibujar: "hora del día" si la columna
        # de fecha trae hora real, o "orden de aparición" si no (ver
        # _configurar_eje_y_hora/_configurar_eje_y_orden).
        layout.addWidget(self.plot_principal, stretch=1)

        self._scatters_hitos = {}  # {columna: pg.ScatterPlotItem}, se arma en cada redibujo
        self._hitos_xs_ordenados = np.array([])
        self._hitos_ys_ordenados = np.array([])
        self._hitos_indices_ordenados = []
        self._hitos_columna_ordenada = []

        self._scatter_anomalias = pg.ScatterPlotItem(
            size=14, symbol="t1", brush=pg.mkBrush(QColor(COLOR_DANGER))
        )
        self._scatter_anomalias.sigClicked.connect(self._on_click_marcador_anomalia)
        self.plot_principal.addItem(self._scatter_anomalias)

        self._popup_info_fila = _PopupInfoFila(self)
        self._popup_conexion = _PopupConexion(self)
        self.plot_principal.scene().sigMouseClicked.connect(self._on_click_principal)

        # Supr / Retroceso borra la conexión manual seleccionada. El atajo
        # solo vale mientras el foco esté en ESTA pestaña, para no pisar
        # el Supr de la tabla de Datos.
        for tecla in (QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace)):
            atajo = QShortcut(tecla, self)
            atajo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            atajo.activated.connect(self._borrar_conexion_seleccionada)

        pie = QHBoxLayout()
        self.btn_actualizar_anomalias = QPushButton("Actualizar anomalías en este rango")
        self.btn_actualizar_anomalias.clicked.connect(self._actualizar_anomalias_en_rango)
        self.btn_actualizar_anomalias.setCursor(Qt.PointingHandCursor)
        self.btn_actualizar_anomalias.setStyleSheet(QSS_SUBPESTANA)
        pie.addWidget(self.btn_actualizar_anomalias)
        self.lbl_estado = QLabel("")
        self.lbl_estado.setObjectName("muted")
        pie.addWidget(self.lbl_estado)
        pie.addStretch()
        layout.addLayout(pie)

    # ------------------------------------------------------------------
    # Carga / refresco de datos (llamado desde main_window al cambiar de
    # dataset o al abrir la pestaña -- ver integración en main_window.py)
    # ------------------------------------------------------------------
    def refrescar_datos(self):
        df = self.host.df
        self.combo_fecha.blockSignals(True)
        self.combo_fecha.clear()
        if df is None or df.empty:
            self.combo_fecha.blockSignals(False)
            self._limpiar_plots()
            return

        # Dataset (o esquema) nuevo: hay que volver a mirar cómo vienen escritas
        # las fechas, y lo que se calculó antes ya no corresponde.
        self._formatos_detectados.clear()
        self._cruces_anomalias = []

        columnas_fecha = detectar_columnas_fecha(df)
        self.combo_fecha.addItems(columnas_fecha)
        self.combo_fecha.blockSignals(False)

        if not columnas_fecha:
            self.lbl_estado.setText("No se detectó ninguna columna de fecha en este dataset.")
            self._columna_fecha = None
            self._fuentes_fecha = {}
            self.host.rango_tiempo = None
            self._limpiar_plots()
            return

        self._columna_fecha = columnas_fecha[0]
        self.combo_fecha.setCurrentText(self._columna_fecha)

        # Registro de columnas de fecha disponibles para "Fechas a
        # mostrar": las de la tabla activa (clave (None, columna)) y, si
        # hay más tablas cargadas (base de datos relacional), también las
        # de esas otras tablas (clave (nombre_tabla, columna)) -- para
        # verlas juntas aunque no compartan fila ni esquema.
        self._fuentes_fecha = {(None, col): "local" for col in columnas_fecha}
        for nombre_tabla, tabla_df in self.host.tablas.items():
            if nombre_tabla == getattr(self.host, "nombre_tabla_activa", None):
                continue
            for col in detectar_columnas_fecha(tabla_df):
                self._fuentes_fecha[(nombre_tabla, col)] = "externa"

        # Color estable por columna (según su posición en el registro, no
        # según cuáles estén tildadas) -- así una misma columna no cambia
        # de color solo porque destildaste otra.
        self._color_por_columna = {
            clave: DONUT_PALETTE[i % len(DONUT_PALETTE)] for i, clave in enumerate(self._fuentes_fecha)
        }
        self._columnas_mostrar = {(None, self._columna_fecha)}
        self._poblar_menu_columnas_mostrar()

        # Nuevo dataset: las conexiones manuales de otro esquema no tienen
        # sentido acá (los índices de fila ni siquiera representan lo mismo).
        self._conexiones_manuales = []
        self._modo_conectar_manual = False
        self.btn_conectar_manual.setChecked(False)

        self._refrescar_menu_conexiones(columnas_fecha)
        self._redibujar_todo()

    def _poblar_menu_columnas_mostrar(self):
        self.menu_columnas_mostrar.clear()
        locales = [c for c in self._fuentes_fecha if c[0] is None]
        externas = [c for c in self._fuentes_fecha if c[0] is not None]
        for clave in locales:
            self._agregar_accion_columna_mostrar(clave, clave[1])
        if externas:
            self.menu_columnas_mostrar.addSeparator()
            for clave in externas:
                tabla, col = clave
                self._agregar_accion_columna_mostrar(clave, f"{tabla}.{col}  (otra tabla)")

    def _agregar_accion_columna_mostrar(self, clave, etiqueta):
        accion = QAction(etiqueta, self.menu_columnas_mostrar)
        accion.setCheckable(True)
        accion.setChecked(clave in self._columnas_mostrar)
        accion.toggled.connect(lambda marcado, clave=clave: self._on_columna_mostrar_toggled(clave, marcado))
        self.menu_columnas_mostrar.addAction(accion)

    def _on_columna_mostrar_toggled(self, columna, marcado):
        if marcado:
            self._columnas_mostrar.add(columna)
        else:
            self._columnas_mostrar.discard(columna)
        self._redibujar_todo()

    def _refrescar_menu_conexiones(self, columnas_fecha):
        pares_manuales = []
        if self.host.fingerprint_actual:
            pares_manuales = self.host.memoria.obtener_pares_temporales(self.host.fingerprint_actual)
        pares = sugerir_pares_temporales(columnas_fecha, pares_manuales)

        self._pares_disponibles = {(p.col_inicio, p.col_fin): p for p in pares}
        # +3 de desfase para no repetir tal cual los primeros colores que
        # ya usan las columnas sueltas en "Fechas a mostrar".
        self._color_por_par = {
            clave: DONUT_PALETTE[(i + 3) % len(DONUT_PALETTE)]
            for i, clave in enumerate(self._pares_disponibles)
        }
        self._pares_conectar = {c for c in self._pares_conectar if c in self._pares_disponibles}

        self.menu_conectar.clear()
        for clave, par in self._pares_disponibles.items():
            marca = "manual" if par.origen == "manual" else "auto"
            accion = QAction(f"{par.etiqueta}  ({marca})", self.menu_conectar)
            accion.setCheckable(True)
            accion.setChecked(clave in self._pares_conectar)
            accion.toggled.connect(lambda marcado, clave=clave: self._on_par_conectar_toggled(clave, marcado))
            self.menu_conectar.addAction(accion)
        self.menu_conectar.addSeparator()
        self.menu_conectar.addAction("Configurar relaciones temporales...", self._abrir_configuracion_pares)

    def _on_par_conectar_toggled(self, clave, marcado):
        if marcado:
            self._pares_conectar.add(clave)
        else:
            self._pares_conectar.discard(clave)
        self._redibujar_todo()

    def _abrir_configuracion_pares(self):
        # Reutiliza el diálogo que ya existe (ver Narrativa) -- esta
        # pestaña no define su propia forma de configurar pares, así los
        # pares quedan iguales se configuren desde donde se configuren.
        self.host._configurar_relaciones_temporales()
        if self.host.df is not None:
            self._refrescar_menu_conexiones(detectar_columnas_fecha(self.host.df))
            self._redibujar_todo()

    def aplicar_tema(self, colors):
        """Llamado desde apply_theme() de HadarApp cuando cambia claro/oscuro
        -- self.colors ahí se reemplaza por un diccionario nuevo cada vez,
        así que este panel necesita que se lo pasen de nuevo explícitamente."""
        self.colors = colors
        for plot in (self.plot_histograma, self.plot_principal):
            plot.setBackground(colors["card"])
            plot.getAxis("bottom").setPen(pg.mkPen(colors["muted"]))
            plot.getAxis("left").setPen(pg.mkPen(colors["muted"]))
            plot.getAxis("bottom").setTextPen(pg.mkPen(colors["text"]))
            plot.getAxis("left").setTextPen(pg.mkPen(colors["text"]))

    # ------------------------------------------------------------------
    # Redibujo
    # ------------------------------------------------------------------
    def _limpiar_plots(self):
        self._ocultar_popups()
        quitar_ventanas(self.plot_principal, getattr(self, "_items_futuro", []))
        self._items_futuro = []
        for scatter in self._scatters_hitos.values():
            self.plot_principal.removeItem(scatter)
        self._scatters_hitos.clear()
        self._actualizar_leyenda([])
        self._hitos_xs_ordenados = np.array([])
        self._hitos_ys_ordenados = np.array([])
        self._hitos_indices_ordenados = []
        self._hitos_columna_ordenada = []
        self._posicion_punto = {}
        for conexion in self._lineas_conexion:
            self.plot_principal.removeItem(conexion["item"])
        self._lineas_conexion.clear()
        self._segmentos = None
        self._conexion_seleccionada = None
        self._scatter_anomalias.setData([])
        self._marcadores_anomalias.clear()
        self._cruces_anomalias = []
        if self._barra_histograma is not None:
            self.plot_histograma.removeItem(self._barra_histograma)
            self._barra_histograma = None

    def _ocultar_popups(self):
        self._popup_info_fila.ocultar()
        self._popup_conexion.ocultar()
        self._deseleccionar_conexion()

    def hideEvent(self, evento):
        # Al cambiar de pestaña (o minimizar) el recuadro no debe quedar
        # esperando a que vuelvas: se cierra, como si hubieras hecho clic en el vacío.
        self._ocultar_popups()
        super().hideEvent(evento)

    # ------------------------------------------------------------------
    # Formato de fecha (automático o forzado por el usuario)
    # ------------------------------------------------------------------
    def _formato_de(self, clave) -> FormatoFecha:
        """Cómo leer la columna `clave` = (tabla_o_None, columna). En
        automático se decide mirando la columna COMPLETA de su tabla de
        origen (no el subconjunto filtrado), así el resultado no cambia
        según qué filtros haya puestos en Datos, y se guarda en caché."""
        if self._modo_formato != "auto":
            return FormatoFecha(
                orden=self._modo_formato, dayfirst=(self._modo_formato == "dmy"),
                seguro=True, descripcion="",
            )
        formato = self._formatos_detectados.get(clave)
        if formato is None:
            tabla, columna = clave
            df_fuente = self._df_de_tabla(tabla)
            if df_fuente is None or columna not in df_fuente.columns:
                formato = FormatoFecha(orden=None, dayfirst=True, seguro=False, descripcion="")
            else:
                formato = detectar_formato_fecha(df_fuente[columna])
            self._formatos_detectados[clave] = formato
        return formato

    def _dayfirst_de(self, clave) -> bool:
        return self._formato_de(clave).dayfirst

    def _df_de_tabla(self, tabla):
        """DataFrame completo (sin filtros de Datos) de la tabla activa
        (tabla=None) o de otra tabla cargada."""
        return self.host.df if tabla is None else self.host.tablas.get(tabla)

    def _actualizar_texto_formato_auto(self):
        """Deja escrito en el combo qué eligió el modo automático, para que
        no sea una caja negra ("Automático: AAAA/MM/DD")."""
        texto = "Automático"
        if self._modo_formato == "auto" and self._columna_fecha:
            descripcion = self._formato_de((None, self._columna_fecha)).descripcion
            if descripcion:
                texto = f"Automático: {descripcion}"
        # Sin bloquear señales, cambiar el texto del ítem ACTUAL del combo dispara
        # currentIndexChanged y se tomaría por un cambio de formato del usuario
        # (que borra el rango del slider).
        self.combo_formato_fecha.blockSignals(True)
        self.combo_formato_fecha.setItemText(0, texto)
        self.combo_formato_fecha.blockSignals(False)

    # ------------------------------------------------------------------
    # Vista de abajo: qué se dibuja según el rango del slider
    # ------------------------------------------------------------------
    def _ventana_de_tiempo(self):
        """(t0, t1) del rango activo del slider, o None si no hay recorte.
        Mismas reglas que apply_table_filter(): el buscador de filas
        puntuales de Datos ignora a propósito el rango."""
        rango = self.host.rango_tiempo
        if rango is None or self.host.filas_buscadas:
            return None
        return rango[0], rango[1]

    def _df_en_rango(self):
        """Lo que se dibuja abajo: la base filtrada por TODO lo de Datos
        (buscador de filas, filtro por columna, clic en un gráfico) más el
        recorte del slider. El histograma de arriba, en cambio, usa la base
        SIN el recorte -- si no, se achicaría sobre sí mismo al arrastrar."""
        df = self.host._construir_df_filtrado_base()
        if df is None:
            return None
        return self.host._recortar_por_rango_tiempo(df)

    def _redibujar_todo(self):
        self._ocultar_popups()
        df_base = self.host._construir_df_filtrado_base()
        if df_base is None or not self._columna_fecha or self._columna_fecha not in df_base.columns:
            self._limpiar_plots()
            return

        clave_ref = (None, self._columna_fecha)
        serie_hitos = construir_serie_hitos(df_base, self._columna_fecha, dayfirst=self._dayfirst_de(clave_ref))
        self._total_fechas = len(serie_hitos)
        self._dibujar_histograma(serie_hitos)
        self._actualizar_texto_formato_auto()
        self._redibujar_principal()

    def _redibujar_principal(self):
        """Puntos + líneas de conexión + marcadores de anomalías de la vista
        de abajo, para lo que cae dentro del rango del slider. NO toca el
        histograma ni el propio slider: es lo único que hay que rehacer
        cuando solo se movió el rango."""
        self._ocultar_popups()
        df = self._df_en_rango()
        if df is None or not self._columna_fecha or self._columna_fecha not in df.columns:
            return

        self.plot_principal.enableAutoRange(axis="xy")
        self._dibujar_hitos_multi(df)
        self._redibujar_conexiones(df)
        self._pintar_marcadores_anomalias()
        self._dibujar_futuro()

        # Bug real: self._columnas_mostrar guarda tuplas (tabla, columna)
        # desde que "Fechas a mostrar" se extendió a otras tablas -- un
        # join() directo sobre tuplas revienta con "expected str instance,
        # tuple found". Hay que pasar cada una por su etiqueta de texto
        # primero (columna sola si es local, "Tabla.columna" si es externa).
        columnas_mostradas = ", ".join(
            self._etiqueta_clave(c) for c in sorted(self._columnas_mostrar, key=lambda c: (c[0] or "", c[1]))
        ) or self._columna_fecha
        if self._ventana_de_tiempo() is not None:
            n_en_rango = len(construir_serie_hitos(
                df, self._columna_fecha, dayfirst=self._dayfirst_de((None, self._columna_fecha)),
            ))
            self.lbl_estado.setText(
                f"{n_en_rango:,} de {self._total_fechas:,} fecha(s) de '{self._columna_fecha}' "
                f"dentro del período elegido (mostrando: {columnas_mostradas})."
            )
        else:
            self.lbl_estado.setText(
                f"{self._total_fechas:,} fecha(s) válida(s) en '{self._columna_fecha}' "
                f"(mostrando: {columnas_mostradas})."
            )

    def _dibujar_histograma(self, serie_hitos):
        if self._barra_histograma is not None:
            self.plot_histograma.removeItem(self._barra_histograma)
            self._barra_histograma = None
        if serie_hitos.empty:
            return
        bordes, conteos = construir_histograma(serie_hitos)
        if len(conteos) == 0:
            return
        centros = [
            (_timestamp_a_epoch(bordes[i]) + _timestamp_a_epoch(bordes[i + 1])) / 2
            for i in range(len(conteos))
        ]
        ancho = _timestamp_a_epoch(bordes[1]) - _timestamp_a_epoch(bordes[0])
        self._barra_histograma = pg.BarGraphItem(
            x=centros, height=list(conteos), width=ancho * 0.9,
            brush=COLOR_ACCENT,
        )
        self.plot_histograma.addItem(self._barra_histograma)

        t0, t1 = _timestamp_a_epoch(bordes[0]), _timestamp_a_epoch(bordes[-1])
        self.region_rango.blockSignals(True)
        self.region_rango.setBounds([t0, t1])
        if self.host.rango_tiempo is None:
            self.region_rango.setRegion([t0, t1])
        self.region_rango.blockSignals(False)

    def _dibujar_hitos_multi(self, df):
        for scatter in self._scatters_hitos.values():
            self.plot_principal.removeItem(scatter)
        self._scatters_hitos.clear()

        # "Fechas a mostrar" guarda claves (tabla_o_None, columna); los
        # pares de "Unir fechas" y las conexiones manuales de aquí para
        # abajo son siempre de la tabla activa, así que se normalizan a
        # esa misma forma. Una línea nunca puede apuntar a un punto
        # invisible, así que sus columnas también se agregan.
        columnas_deseadas = set(self._columnas_mostrar)
        for (col_a, col_b) in self._pares_conectar:
            columnas_deseadas.add((None, col_a))
            columnas_deseadas.add((None, col_b))
        for conexion in self._conexiones_manuales:
            columnas_deseadas.add(conexion["desde"][1])
            columnas_deseadas.add(conexion["hasta"][1])
        if not columnas_deseadas and self._columna_fecha:
            columnas_deseadas = {(None, self._columna_fecha)}

        # Cada clave trae su propia fuente: local = el df activo ya
        # filtrado que se está dibujando; externa = la tabla propia tal
        # cual está cargada, SIN los filtros de Datos de la tabla activa
        # (no correspondería aplicarle el filtro de una tabla a otra con
        # columnas distintas).
        series_por_clave = {}
        for clave in columnas_deseadas:
            tabla, columna = clave
            df_fuente = df if tabla is None else self.host.tablas.get(tabla)
            if df_fuente is None or columna not in df_fuente.columns:
                continue
            serie = construir_serie_hitos(df_fuente, columna, dayfirst=self._dayfirst_de(clave))
            if tabla is not None:
                # Una tabla externa no tiene filas en común con la activa, así
                # que el recorte del slider no la alcanza por fila: se aplica
                # directamente sobre sus fechas (el tiempo sí es común).
                ventana = self._ventana_de_tiempo()
                if ventana is not None:
                    serie = serie[(serie >= ventana[0]) & (serie <= ventana[1])]
            if not serie.empty:
                series_por_clave[clave] = serie

        self._posicion_punto = {}

        if not series_por_clave:
            self._hitos_xs_ordenados = np.array([])
            self._hitos_ys_ordenados = np.array([])
            self._hitos_indices_ordenados = []
            self._hitos_columna_ordenada = []
            self._actualizar_leyenda([])
            return

        usar_hora = any(tiene_componente_horario(s) for s in series_por_clave.values())
        if usar_hora:
            self._configurar_eje_y_hora()
        else:
            self._configurar_eje_y_orden()

        todos_x, todos_y, todos_idx, todos_col = [], [], [], []

        if usar_hora:
            # Y = hora del día, calculado por punto -- una columna sin hora
            # real simplemente cae en 00:00, que es honesto (no se sabe más
            # que eso), no hay que inventarle un ranking aparte.
            for clave, serie in series_por_clave.items():
                xs = [_timestamp_a_epoch(t) for t in serie]
                ys = minutos_desde_medianoche(serie).tolist()
                self._agregar_scatter_columna(clave, xs, ys)
                for indice, x, y in zip(serie.index, xs, ys):
                    self._posicion_punto[(indice, clave)] = (x, y)
                todos_x.extend(xs)
                todos_y.extend(ys)
                todos_idx.extend(serie.index)
                todos_col.extend([clave] * len(xs))
        else:
            # "Orden de aparición" es un ranking GLOBAL entre TODAS las
            # columnas mostradas juntas -- si cada columna tuviera su propio
            # ranking 0..n, dos columnas se pisarían en el mismo rango de Y
            # sin que signifique lo mismo.
            combinado = []
            for clave, serie in series_por_clave.items():
                for indice, ts in serie.items():
                    combinado.append((ts, indice, clave))
            combinado.sort(key=lambda t: t[0])

            puntos_por_columna = {clave: ([], []) for clave in series_por_clave}
            for orden, (ts, indice, clave) in enumerate(combinado):
                x = _timestamp_a_epoch(ts)
                puntos_por_columna[clave][0].append(x)
                puntos_por_columna[clave][1].append(orden)
                self._posicion_punto[(indice, clave)] = (x, orden)
                todos_x.append(x)
                todos_y.append(orden)
                todos_idx.append(indice)
                todos_col.append(clave)

            for clave, (xs, ys) in puntos_por_columna.items():
                self._agregar_scatter_columna(clave, xs, ys)

        # Para el clic: todos los puntos de todas las columnas, ordenados
        # juntos por X (búsqueda binaria sobre el conjunto combinado).
        orden_click = np.argsort(todos_x)
        self._hitos_xs_ordenados = np.array(todos_x)[orden_click]
        self._hitos_ys_ordenados = np.array(todos_y)[orden_click]
        self._hitos_indices_ordenados = [todos_idx[i] for i in orden_click]
        self._hitos_columna_ordenada = [todos_col[i] for i in orden_click]

        self._actualizar_leyenda(list(series_por_clave.keys()))

    def _agregar_scatter_columna(self, clave, xs, ys):
        # Círculo para columnas de la tabla activa, cuadrado para columnas
        # de otra tabla (base de datos relacional) -- el símbolo, no solo
        # el color, avisa que ese punto viene de otro origen.
        tabla, _ = clave
        simbolo = "o" if tabla is None else "s"
        color = self._color_por_columna.get(clave, COLOR_ACCENT)
        scatter = pg.ScatterPlotItem(x=xs, y=ys, size=8, symbol=simbolo, brush=pg.mkBrush(color))
        self.plot_principal.addItem(scatter)
        self._scatters_hitos[clave] = scatter

    @staticmethod
    def _etiqueta_clave(clave):
        """Texto legible de una clave (tabla_o_None, columna): solo el
        nombre si es de la tabla activa, "Tabla.columna" si es de otra."""
        tabla, col = clave
        return col if tabla is None else f"{tabla}.{col}"

    def _actualizar_leyenda(self, columnas_con_datos):
        # Leyenda propia (QLabel con un cuadradito de color) en vez de la
        # de PyQtGraph: con ScatterPlotItem sueltos, el ícono de la leyenda
        # nativa no siempre pinta el color correcto -- esto es más simple y
        # 100% predecible. Solo se muestra si hay 2+ columnas a la vista;
        # con una sola, el color no aporta nada que distinguir.
        while self.fila_leyenda.count():
            item = self.fila_leyenda.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if len(columnas_con_datos) <= 1:
            return
        for clave in columnas_con_datos:
            tabla, col = clave
            color = self._color_por_columna.get(clave, COLOR_ACCENT)
            cuadro = QLabel("■" if tabla is None else "▪")
            cuadro.setStyleSheet(f"color: {color}; font-size: 13px;")
            self.fila_leyenda.addWidget(cuadro)
            self.fila_leyenda.addWidget(QLabel(self._etiqueta_clave(clave)))
        self.fila_leyenda.addStretch()

    def _configurar_eje_y_hora(self):
        """La columna de fecha trae hora real: Y = hora del día (0-24h),
        X = fecha. Deja ver de un vistazo si algo se repite siempre a la
        misma hora (ej. fallas siempre de madrugada), algo que 'orden de
        aparición' nunca podría mostrar."""
        eje = self.plot_principal.getAxis("left")
        eje.setLabel("Hora del día")
        eje.setTicks([[(m, f"{m // 60:02d}:{m % 60:02d}") for m in range(0, 1441, 180)]])
        eje.setStyle(showValues=True)
        self.plot_principal.setYRange(0, 1440, padding=0.02)

    def _configurar_eje_y_orden(self):
        """La columna de fecha no trae hora real (todo a las 00:00:00) --
        mostrar 'hora del día' aquí mentiría (todo quedaría pegado en
        cero), así que se cae de vuelta a un eje sin unidad real, con eso
        explícito en la etiqueta."""
        eje = self.plot_principal.getAxis("left")
        eje.setLabel("Orden de aparición (sin agrupar)")
        eje.setTicks(None)
        eje.setStyle(showValues=False)
        self.plot_principal.enableAutoRange(axis="y")

    # ------------------------------------------------------------------
    # Conexiones entre puntos (automáticas por par, o manuales punto a punto)
    # ------------------------------------------------------------------
    def _redibujar_conexiones(self, df):
        for conexion in self._lineas_conexion:
            self.plot_principal.removeItem(conexion["item"])
        self._lineas_conexion.clear()
        self._conexion_seleccionada = None  # sus líneas acaban de desaparecer

        # Automáticas: un par tildado en "Unir fechas" conecta ambas
        # columnas en cada fila donde las dos fechas son válidas. Se
        # reutiliza construir_intervalos(), que ya calculaba exactamente
        # esto (inicio/fin/duración/incoherente) para el extinto Gantt.
        for clave in self._pares_conectar:
            par = self._pares_disponibles.get(clave)
            if par is None:
                continue
            intervalos = construir_intervalos(
                df, par,
                dayfirst=self._dayfirst_de((None, par.col_inicio)),
                dayfirst_fin=self._dayfirst_de((None, par.col_fin)),
            )
            color_normal = QColor(self._color_por_par.get(clave, COLOR_ACCENT))
            for indice, fila in intervalos.iterrows():
                p_ini = self._posicion_punto.get((indice, (None, par.col_inicio)))
                p_fin = self._posicion_punto.get((indice, (None, par.col_fin)))
                if p_ini is None or p_fin is None:
                    continue
                color = QColor(COLOR_DANGER) if fila["incoherente"] else color_normal
                self._agregar_conexion(
                    p_ini, p_fin, color, Qt.PenStyle.SolidLine,
                    desde=(indice, par.col_inicio), hasta=(indice, par.col_fin),
                    duracion=fila["duracion"], tipo="auto", etiqueta=par.etiqueta,
                )

        # Manuales: puntos elegidos a mano, de la misma fila o no. Si algún
        # extremo quedó fuera del rango del slider, la línea no se dibuja
        # (sigue guardada: reaparece al ampliar el rango).
        for conexion in self._conexiones_manuales:
            p_ini = self._posicion_punto.get(conexion["desde"])
            p_fin = self._posicion_punto.get(conexion["hasta"])
            if p_ini is None or p_fin is None:
                continue
            ts_ini = self._fecha_de_punto(df, conexion["desde"])
            ts_fin = self._fecha_de_punto(df, conexion["hasta"])
            duracion = (ts_fin - ts_ini) if (ts_ini is not None and ts_fin is not None) else None
            color = (
                QColor(COLOR_DANGER)
                if (duracion is not None and duracion.total_seconds() < 0)
                else QColor(_COLOR_CONEXION_MANUAL)
            )
            self._agregar_conexion(
                p_ini, p_fin, color, Qt.PenStyle.DashLine,
                desde=conexion["desde"], hasta=conexion["hasta"],
                duracion=duracion, tipo="manual", etiqueta="Conexión manual",
            )

        # Índice de segmentos en numpy para acertar clics sobre líneas (ver
        # _conexion_bajo_cursor).
        if self._lineas_conexion:
            self._segmentos = (
                np.array([c["p_ini"][0] for c in self._lineas_conexion], dtype=float),
                np.array([c["p_ini"][1] for c in self._lineas_conexion], dtype=float),
                np.array([c["p_fin"][0] for c in self._lineas_conexion], dtype=float),
                np.array([c["p_fin"][1] for c in self._lineas_conexion], dtype=float),
            )
        else:
            self._segmentos = None

    def _agregar_conexion(self, p_ini, p_fin, color, estilo, **datos):
        item = self._crear_linea(p_ini, p_fin, color, estilo)
        self._lineas_conexion.append({
            "item": item, "p_ini": p_ini, "p_fin": p_fin,
            "color": color, "estilo": estilo, **datos,
        })

    @staticmethod
    def _pintar_linea(item, color, estilo, grosor):
        pluma = QPen(color, grosor)
        # Cosmetic=True fija el grosor en PÍXELES de pantalla, sin importar
        # el zoom. Sin esto, un QGraphicsLineItem interpreta el grosor en
        # unidades de dato -- al acercar el slider de rango, esas mismas 2
        # unidades pasan a cubrir muchos más píxeles, y la línea se ve cada
        # vez más ancha hasta tapar la pantalla (justo lo que se veía).
        pluma.setCosmetic(True)
        pluma.setStyle(estilo)
        item.setPen(pluma)

    def _crear_linea(self, p_ini, p_fin, color, estilo):
        item = QGraphicsLineItem(p_ini[0], p_ini[1], p_fin[0], p_fin[1])
        self._pintar_linea(item, color, estilo, 2)
        item.setZValue(-1)  # detrás de los puntos, para no taparlos
        self.plot_principal.addItem(item)
        return item

    def _seleccionar_conexion(self, conexion):
        """Resalta la línea (más gruesa y del color del texto del tema) para
        que se vea cuál es la que Supr va a borrar."""
        self._deseleccionar_conexion()
        self._conexion_seleccionada = conexion
        self._pintar_linea(conexion["item"], QColor(self.colors["text"]), conexion["estilo"], 4)

    def _deseleccionar_conexion(self):
        conexion = self._conexion_seleccionada
        self._conexion_seleccionada = None
        # Si la vista se redibujó, esa línea ya no existe: no hay nada que restaurar.
        if conexion is not None and any(conexion is c for c in self._lineas_conexion):
            self._pintar_linea(conexion["item"], conexion["color"], conexion["estilo"], 2)

    def _fecha_de_punto(self, df, referencia):
        indice, clave = referencia
        tabla, columna = clave
        df_fuente = df if tabla is None else self.host.tablas.get(tabla)
        if df_fuente is None or columna not in df_fuente.columns or indice not in df_fuente.index:
            return None
        valor = parsear_fechas(pd.Series([df_fuente.loc[indice, columna]]), dayfirst=self._dayfirst_de(clave)).iloc[0]
        return valor if pd.notna(valor) else None

    def _on_toggle_conectar_manual(self, activo):
        self._modo_conectar_manual = activo
        self._primer_punto_manual = None
        self._ocultar_popups()
        if activo:
            self.lbl_estado.setText("Modo conectar activo: clic en el primer punto, luego en el segundo.")
        else:
            self._redibujar_todo()

    # ------------------------------------------------------------------
    # Slider de rango -> filtro global (solo Métricas/Indicadores/etc.)
    # ------------------------------------------------------------------
    def _on_region_cambiada(self):
        self._timer_debounce.start()

    def _aplicar_rango_activo(self):
        t0, t1 = self.region_rango.getRegion()
        # El cuarto valor es el formato con que se leyó la columna: el filtro
        # de main_window tiene que leer las fechas EXACTAMENTE igual que este
        # panel, o el recorte y el gráfico hablarían de días distintos.
        self.host.rango_tiempo = (
            _epoch_a_timestamp(t0), _epoch_a_timestamp(t1), self._columna_fecha,
            self._dayfirst_de((None, self._columna_fecha)),
        )
        # apply_table_filter() actualiza Datos/Métricas/Indicadores/etc. Su
        # aviso de "redibuja la Línea de Tiempo" se salta a propósito cuando
        # el cambio viene del slider (el histograma NO debe rehacerse en cada
        # arrastre); la vista de abajo sí se rehace a mano justo después.
        self._actualizando_desde_slider = True
        try:
            self.host.apply_table_filter()
        finally:
            self._actualizando_desde_slider = False
        self._redibujar_principal()

    def _dibujar_futuro(self):
        """Dibuja (o quita) las ventanas de «lo que se viene» según el interruptor. Si hay fechas futuras,
        amplía el eje para que se vean junto a los datos."""
        quitar_ventanas(self.plot_principal, self._items_futuro)
        self._items_futuro = []
        tablero = getattr(self.host, "tablero", None)
        ventanas = tablero.ventanas_futuras() if tablero is not None else []
        self.chk_futuro.setEnabled(bool(ventanas))
        if not ventanas:
            self.chk_futuro.setToolTip(
                "Todavía no hay fechas futuras que mostrar. Calcula un agotamiento, umbral, saturación o vida útil "
                "en Predicción > Preguntas y usa «Llevar al Panel».")
        if not ventanas or not self.chk_futuro.isChecked():
            self.plot_principal.enableAutoRange(axis="x")      # el eje vuelve a ajustarse a los datos
            return
        items, extremos = dibujar_ventanas(self.plot_principal, ventanas)
        self._items_futuro = items
        if extremos is None:
            return
        xs = getattr(self, "_hitos_xs_ordenados", None)
        lo, hi = extremos
        if xs is not None and len(xs):
            lo, hi = min(lo, float(np.min(xs))), max(hi, float(np.max(xs)))
        self.plot_principal.setXRange(lo, hi, padding=0.04)

    def refrescar_futuro(self):
        """Lo llama la ventana principal cuando se lleva un pronóstico nuevo al Panel (o se quita)."""
        try:
            self._dibujar_futuro()
        except Exception:
            pass

    def refrescar_vista(self):
        """Redibuja con los filtros de Datos actuales (buscador de filas,
        filtro por columna, clic en un gráfico), sin tocar los combos ni el
        menú de columnas -- eso es cosa de refrescar_datos(), pensado para
        cuando cambia el ESQUEMA, no los filtros. Se llama sola cada vez que
        cambia cualquier filtro de Datos (ver el hook en apply_table_filter),
        y también con el botón "Actualizar", por si acaso (ahí además se
        vuelve a detectar el formato de las fechas)."""
        self._formatos_detectados.clear()
        self._redibujar_todo()

    def limpiar_rango(self):
        self.host.rango_tiempo = None
        self.host.apply_table_filter()
        self._redibujar_todo()

    def _reiniciar_rango_y_redibujar(self):
        """Quita el recorte del slider y rehace todo. Si había un recorte,
        pasa por apply_table_filter() para que Datos/Métricas/etc. también
        lo pierdan (antes solo se olvidaba el rango y esas pestañas
        quedaban recortadas con un rango que ya no existía); ese mismo
        llamado redibuja este panel por el hook de main_window."""
        habia_rango = self.host.rango_tiempo is not None
        self.host.rango_tiempo = None
        if habia_rango:
            self.host.apply_table_filter()
        else:
            self._redibujar_todo()

    def _on_columna_fecha_cambiada(self, texto):
        if not texto:
            return
        self._columna_fecha = texto
        self._reiniciar_rango_y_redibujar()

    def _on_formato_fecha_cambiado(self, _indice):
        self._modo_formato = self.combo_formato_fecha.currentData()
        self._formatos_detectados.clear()
        # Todas las fechas ya dibujadas se leyeron con el formato viejo --
        # hay que rehacer el rango (los bordes cambian si, por ejemplo, lo
        # que se leía como "3 de abril" pasa a ser "4 de marzo") y todo el
        # dibujo, no solo refrescar_vista().
        self._reiniciar_rango_y_redibujar()

    # ------------------------------------------------------------------
    # Anomalías: bajo demanda, nunca en cada arrastre del slider
    # ------------------------------------------------------------------
    def _actualizar_anomalias_en_rango(self):
        df = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
        if df is None or df.empty or not self._columna_fecha:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        pares_personalizados = []
        if self.host.fingerprint_actual:
            pares_personalizados = self.host.memoria.obtener_pares_temporales(self.host.fingerprint_actual)

        detector = SemanticAnomalyDetector(df, pares_temporales_personalizados=pares_personalizados)
        anomalias = detector.detect_all()
        if self.host.fingerprint_actual:
            anomalias = self.host.memoria.enriquecer_con_memoria(self.host.fingerprint_actual, anomalias)

        cruces = anomalias_con_fecha(
            anomalias, df, self._columna_fecha, dayfirst=self._dayfirst_de((None, self._columna_fecha)),
        )
        self._cruces_anomalias = cruces
        self._pintar_marcadores_anomalias()
        if not cruces:
            self.lbl_estado.setText("Sin anomalías detectadas en este rango.")
            return
        self.lbl_estado.setText(f"{len(cruces)} anomalía(s) marcada(s) en este rango.")

    def _pintar_marcadores_anomalias(self):
        """Dibuja solo las anomalías (ya calculadas) que caen dentro del rango
        activo del slider. Así, al recortar el período, no quedan marcadores
        flotando fuera de la vista ni estirando el gráfico hacia fechas que
        ya no se están mirando."""
        visibles = self._cruces_anomalias
        ventana = self._ventana_de_tiempo()
        if ventana is not None:
            visibles = [c for c in visibles if ventana[0] <= c["fecha"] <= ventana[1]]
        self._marcadores_anomalias = visibles
        if not visibles:
            self._scatter_anomalias.setData([])
            return
        xs = [_timestamp_a_epoch(c["fecha"]) for c in visibles]
        y_base = float(self._hitos_ys_ordenados.max()) + 2 if len(self._hitos_ys_ordenados) else 1
        self._scatter_anomalias.setData(x=xs, y=[y_base] * len(xs))

    # ------------------------------------------------------------------
    # Clics sobre la vista principal: punto (info de fila / conectar
    # manualmente) o línea (duración). Búsqueda de puntos siempre binaria
    # (np.searchsorted, O(log n)) -- segura con 100 mil filas.
    # ------------------------------------------------------------------
    def _escala_pixeles(self):
        """(vista, píxeles por unidad de dato en X, ídem en Y) de la vista
        principal ahora mismo -- para medir distancias en píxeles de
        pantalla, que es lo que el usuario percibe al apuntar."""
        vb = self.plot_principal.getPlotItem().vb
        (x0, x1), (y0, y1) = vb.viewRange()
        rect_vista = vb.boundingRect()
        px_por_x = max(rect_vista.width(), 1) / max(x1 - x0, 1e-12)
        px_por_y = max(rect_vista.height(), 1) / max(y1 - y0, 1e-12)
        return vb, px_por_x, px_por_y

    def _punto_bajo_cursor(self, pos):
        """El punto más cercano al clic (a menos de _TOLERANCIA_PUNTO_PX
        píxeles), o None. Los puntos están ordenados por X, así que se
        acota con búsqueda binaria (O(log n)) la franja de X que cae dentro
        de la tolerancia y solo ahí se mide la distancia real -- segura con
        100 mil filas. (Antes se miraban solo los 10 vecinos por X, y en
        zonas densas el punto apuntado podía quedar fuera de esos 10.)"""
        if len(self._hitos_xs_ordenados) == 0:
            return None
        vb, px_por_x, px_por_y = self._escala_pixeles()
        punto = vb.mapSceneToView(pos)
        tolerancia_x = _TOLERANCIA_PUNTO_PX / px_por_x
        lo = int(np.searchsorted(self._hitos_xs_ordenados, punto.x() - tolerancia_x, side="left"))
        hi = int(np.searchsorted(self._hitos_xs_ordenados, punto.x() + tolerancia_x, side="right"))
        if lo >= hi:
            return None
        dx = (self._hitos_xs_ordenados[lo:hi] - punto.x()) * px_por_x
        dy = (self._hitos_ys_ordenados[lo:hi] - punto.y()) * px_por_y
        distancia2 = dx * dx + dy * dy
        mejor = int(np.argmin(distancia2))
        if distancia2[mejor] > _TOLERANCIA_PUNTO_PX ** 2:
            return None
        i = lo + mejor
        return self._hitos_indices_ordenados[i], self._hitos_columna_ordenada[i]

    def _conexion_bajo_cursor(self, pos):
        """La línea más cercana al clic (a menos de _TOLERANCIA_LINEA_PX
        píxeles), o None. Se mide la distancia punto-segmento en píxeles con
        numpy. Antes se usaba el picking de Qt (scene().items(pos)), que casi
        nunca acertaba: el 'grosor' de la línea para Qt está en unidades de
        DATO (segundos de fecha en X, minutos en Y), y en pantalla eso es
        una franja de una fracción de píxel -- imposible de clicar."""
        if self._segmentos is None:
            return None
        vb, px_por_x, px_por_y = self._escala_pixeles()
        punto = vb.mapSceneToView(pos)
        x0, y0, x1, y1 = self._segmentos
        # Todo relativo al cursor y en píxeles: el cursor queda en (0, 0).
        ax = (x0 - punto.x()) * px_por_x
        ay = (y0 - punto.y()) * px_por_y
        bx = (x1 - punto.x()) * px_por_x
        by = (y1 - punto.y()) * px_por_y
        dx, dy = bx - ax, by - ay
        largo2 = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(largo2 > 0, -(ax * dx + ay * dy) / largo2, 0.0)
        t = np.clip(t, 0.0, 1.0)
        cx, cy = ax + t * dx, ay + t * dy
        distancia2 = cx * cx + cy * cy
        mejor = int(np.argmin(distancia2))
        if distancia2[mejor] > _TOLERANCIA_LINEA_PX ** 2:
            return None
        return self._lineas_conexion[mejor]

    @staticmethod
    def _numero_fila(df, indice):
        """Número de fila tal como lo cuenta el buscador de filas de Datos:
        la posición dentro de la tabla completa, empezando en 1. Si el
        índice tiene etiquetas repetidas no hay una sola posición, y se
        muestra la etiqueta tal cual."""
        try:
            posicion = df.index.get_loc(indice)
        except KeyError:
            return indice
        if isinstance(posicion, (int, np.integer)):
            return int(posicion) + 1
        return indice

    def _on_click_principal(self, evento):
        # Solo clic izquierdo: el derecho abre el menú propio de PyQtGraph.
        if evento.button() != Qt.MouseButton.LeftButton:
            return
        # Foco en el gráfico: sin esto, Supr no llega al atajo de esta pestaña.
        self.plot_principal.setFocus()
        pos = evento.scenePos()
        if not self.plot_principal.sceneBoundingRect().contains(pos):
            self._ocultar_popups()
            return

        # Un punto siempre tiene su línea tocándolo justo en el extremo: si
        # las líneas se probaran primero, hacer clic en un punto conectado
        # mostraría la duración en vez de su fila, y esa fila sería inalcanzable.
        resultado_punto = self._punto_bajo_cursor(pos)

        if resultado_punto is not None and self._modo_conectar_manual:
            self._on_click_punto_conectando(resultado_punto)
            return

        if resultado_punto is not None:
            self._deseleccionar_conexion()
            self._popup_conexion.ocultar()
            self._mostrar_info_fila(resultado_punto)
            return

        conexion = self._conexion_bajo_cursor(pos)
        if conexion is not None:
            self._popup_info_fila.ocultar()
            self._seleccionar_conexion(conexion)
            self._mostrar_info_conexion(conexion)
            return

        # Clic en el vacío del gráfico: se cierra todo.
        self._ocultar_popups()

    def _on_click_punto_conectando(self, resultado_punto):
        indice_real, columna_origen = resultado_punto
        self._ocultar_popups()
        if self._primer_punto_manual is None:
            self._primer_punto_manual = (indice_real, columna_origen)
            numero = self._numero_fila(self._df_de_tabla(columna_origen[0]), indice_real)
            self.lbl_estado.setText(
                f"Primer punto: fila {numero}, '{self._etiqueta_clave(columna_origen)}'. Clic en el segundo punto."
            )
        elif self._primer_punto_manual == (indice_real, columna_origen):
            self.lbl_estado.setText("Elige un segundo punto distinto del primero.")
        else:
            self._conexiones_manuales.append({
                "desde": self._primer_punto_manual, "hasta": (indice_real, columna_origen),
            })
            self._primer_punto_manual = None
            self.lbl_estado.setText("Conexión creada. Puedes seguir conectando, o apagar el modo.")
            df = self._df_en_rango()
            self._dibujar_hitos_multi(df)  # por si el 2do punto no estaba visible aún
            self._redibujar_conexiones(df)

    def _mostrar_info_fila(self, resultado_punto):
        indice_real, columna_origen = resultado_punto
        tabla_origen, _ = columna_origen
        df_para_fila = self._df_de_tabla(tabla_origen)
        if df_para_fila is None or indice_real not in df_para_fila.index:
            self._popup_info_fila.ocultar()
            return
        fila = df_para_fila.loc[indice_real]
        numero = self._numero_fila(df_para_fila, indice_real)
        cabecera = f"<b>Fila {numero}</b>" + ("" if tabla_origen is None else f" · tabla {tabla_origen}")
        cuerpo = "<br>".join(f"<b>{col}:</b> {valor}" for col, valor in fila.items())
        texto = f"{cabecera}<br>{cuerpo}"
        if len(self._columnas_mostrar) > 1 or self._pares_conectar or self._conexiones_manuales:
            texto = f"{cabecera}<br><i>Fecha desde: {self._etiqueta_clave(columna_origen)}</i><br>{cuerpo}"
        self._popup_info_fila.mostrar(texto, QCursor.pos())

    def _mostrar_info_conexion(self, conexion):
        duracion = conexion["duracion"]
        if duracion is None or pd.isna(duracion):
            texto = f"<b>{conexion['etiqueta']}</b><br>No se pudo calcular la duración."
        else:
            texto = f"<b>{conexion['etiqueta']}</b><br>{formatear_duracion(duracion)}"
        on_eliminar = None
        if conexion["tipo"] == "manual":
            on_eliminar = lambda: self._eliminar_conexion_manual(conexion)
        self._popup_conexion.mostrar(texto, QCursor.pos(), on_eliminar=on_eliminar)

    def _borrar_conexion_seleccionada(self):
        """Supr / Retroceso: borra la conexión manual que está resaltada.
        Las automáticas no se borran una por una (vienen de un par tildado
        en 'Unir fechas'), así que ahí no hace nada."""
        conexion = self._conexion_seleccionada
        if conexion is None or conexion["tipo"] != "manual":
            return
        self._eliminar_conexion_manual(conexion)

    def _eliminar_conexion_manual(self, conexion):
        self._conexiones_manuales = [
            c for c in self._conexiones_manuales
            if not (c["desde"] == conexion["desde"] and c["hasta"] == conexion["hasta"])
        ]
        self._ocultar_popups()
        self._redibujar_conexiones(self._df_en_rango())

    def _on_click_marcador_anomalia(self, _scatter, puntos):
        if not puntos:
            return
        indice = puntos[0].index()
        if indice >= len(self._marcadores_anomalias):
            return
        anomalia = self._marcadores_anomalias[indice]["anomalia"]
        dialogo = _TarjetaAnomalia(anomalia, self.colors, self._enviar_anomalia_a_reporte, parent=self)
        dialogo.exec()

    def _enviar_anomalia_a_reporte(self, anomalia):
        # Reutiliza el mismo mecanismo con el que Narrativa ya manda
        # hallazgos al lienzo -- ver ReportBoxItem en main_window.py.
        self.host._reporte_agregar_anomalia_desde_linea_tiempo(anomalia)

    # ------------------------------------------------------------------
    # Métricas de tiempo (promedio/mínimo/máximo por conexión activa)
    # ------------------------------------------------------------------
    def _abrir_metricas_tiempo(self):
        df = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
        if df is None or df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        resumenes = []
        for clave in self._pares_conectar:
            par = self._pares_disponibles.get(clave)
            if par is None:
                continue
            intervalos = construir_intervalos(
                df, par,
                dayfirst=self._dayfirst_de((None, par.col_inicio)),
                dayfirst_fin=self._dayfirst_de((None, par.col_fin)),
            )
            if intervalos.empty:
                continue
            resumen = resumen_duraciones(intervalos["duracion"])
            if resumen:
                resumenes.append((par.etiqueta, resumen))

        if self._conexiones_manuales:
            duraciones_manuales = []
            for conexion in self._conexiones_manuales:
                ts_ini = self._fecha_de_punto(df, conexion["desde"])
                ts_fin = self._fecha_de_punto(df, conexion["hasta"])
                if ts_ini is not None and ts_fin is not None:
                    duraciones_manuales.append(ts_fin - ts_ini)
            resumen_manual = resumen_duraciones(duraciones_manuales)
            if resumen_manual:
                resumenes.append(("Conexiones manuales", resumen_manual))

        if not resumenes:
            QMessageBox.information(
                self, "Sin conexiones",
                "No hay ninguna conexión activa todavía -- tilda un par en "
                "'Unir fechas' o conecta puntos a mano primero."
            )
            return

        dialogo = _DialogoMetricasTiempo(resumenes, self._enviar_metricas_a_reporte, parent=self)
        dialogo.exec()

    def _enviar_metricas_a_reporte(self, titulo, html):
        self.host._reporte_agregar_metricas_tiempo(titulo, html)

    # ------------------------------------------------------------------
    # Envío a la pestaña Reporte (como imagen, con botón de actualizar)
    # ------------------------------------------------------------------
    def snapshot(self) -> QPixmap:
        """Foto de la Línea de Tiempo (histograma + vista principal, con
        líneas de conexión incluidas) para la pestaña Reporte. Usa el
        exportador nativo de PyQtGraph -- igual que ChartPanel.snapshot()
        -- porque un .grab() de un widget que no está visible en pantalla
        en ese momento (ej. estás parado en Reporte, no en Línea de
        Tiempo) sale en blanco."""
        import pyqtgraph.exporters as pg_exporters

        def _exportar(plot_widget):
            try:
                exportador = pg_exporters.ImageExporter(plot_widget.getPlotItem())
                imagen = exportador.export(toBytes=True)
                pixmap = QPixmap.fromImage(imagen)
                if not pixmap.isNull():
                    return pixmap
            except Exception:
                pass
            return plot_widget.grab()

        pix_hist = _exportar(self.plot_histograma)
        pix_main = _exportar(self.plot_principal)
        ancho = max(pix_hist.width(), pix_main.width(), 1)
        alto = pix_hist.height() + pix_main.height()
        combinado = QPixmap(ancho, alto)
        combinado.fill(Qt.GlobalColor.white)
        pintor = QPainter(combinado)
        pintor.drawPixmap(0, 0, pix_hist)
        pintor.drawPixmap(0, pix_hist.height(), pix_main)
        pintor.end()
        return combinado

    def nombre(self) -> str:
        return "Línea de Tiempo"

    def etiqueta(self) -> str:
        cuantas_conexiones = len(self._pares_conectar) + len(self._conexiones_manuales)
        sufijo = f", {cuantas_conexiones} conexión(es)" if cuantas_conexiones else ""
        return f"Línea de Tiempo ({self._columna_fecha or 'sin columna'}{sufijo})"