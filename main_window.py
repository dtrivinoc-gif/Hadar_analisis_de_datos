"""
Ventana principal de Hadar Data Analytics (HadarApp): arma la barra
lateral y las pestañas (Datos, Gráficos, Métricas, Indicadores,
Frecuencias, Narrativa, Reporte), y conecta entre sí todos los módulos
del resto de la app.
"""
import os
import hashlib
import json

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QThread, QTimer, QRectF, QVariantAnimation, QEasingCurve
from PySide6.QtGui import (
    QColor, QFont, QIcon, QPixmap, QPainter, QPen, QBrush,
    QPageSize, QTextCharFormat, QTextDocument,
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QTableView, QTabWidget, QScrollArea, QFrame,
    QLineEdit, QMessageBox, QFileDialog, QRadioButton, QButtonGroup,
    QHeaderView, QInputDialog, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QTabBar, QDialog, QApplication, QTextBrowser,
    QDialogButtonBox, QToolButton, QMenu, QFormLayout,
    QTreeWidget, QTreeWidgetItem, QSizePolicy, QSpinBox,
)
from PySide6.QtPrintSupport import QPrinter

from .config import (
    THEMES, COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_3, COLOR_HOVER,
    COLOR_DANGER, ICON_PATH, LOGO_PNG_PATH,
    URL_POS_SINCRONIZACION, API_KEY_POS_SINCRONIZACION,
)
from .io_datos import (
    SqlMultipleTablesError, read_sql_file, read_sqlite_file, load_data, _excel_sheet_names,
    SqlServerNoDisponible, listar_tablas_sql_server, leer_tabla_sql_server,
    PosNoDisponible, verificar_conexion_pos, listar_tablas_pos, leer_tabla_pos,
    cast_valor_a_dtype,
)
from .table_model import PandasTableModel
from .graficos import UfWorker, ChartPanel, render_boxplot, build_stylesheet, create_stat_card, set_card_alarma
from .indicadores import (
    Indicador, sugerir_indicadores_por_defecto, IndicadorCard, DialogoIndicador,
)
from .excel_transform import DialogoTransformacionExcel
from .linaje_grafo import construir_grafo_linaje
from .linaje_ui import PanelLinaje
from .procedencia import (
    BitacoraProcedencia, TIPO_COLUMNA_CALCULADA, TIPO_ARCHIVO_ORIGEN,
    ORIGEN_ARCHIVO, ORIGEN_SQL_SERVER, evento_de_origen,
    estado_de_actualizacion, frase_desactualizacion,
)
from .report_widgets import (
    ReportBoxItem, REPORTE_PAPEL_BG, REPORTE_PAPEL_BORDE, REPORTE_PAPEL_TINTA,
    REPORTE_COLOR_ROJO, REPORTE_FUENTES, REPORTE_TAMANOS,
)
from .memoria import MemoriaHadar
from .proyecto import guardar_proyecto, abrir_proyecto, ruta_carpeta_proyectos, EXTENSION
from .aprendizaje_adaptativo import actualizar_linea_base
from .anomalias import SemanticAnomalyDetector, anomalias_a_notas_celda, detectar_columnas_fecha
from .deteccion_multivariada import evaluar_confiabilidad
from .dialogos_temporales import ConfigurarRelacionesTemporalesDialog
from .narrativa import QuestionAssistant, NarrativeGenerator, _decodificar_identidad_anomalia
from .dialogo_nota import DialogoNota
from .ontologia_ui import DialogoEsquemaOntologia
from .ontologia import inferir_relaciones, enriquecer_tabla_con_relaciones
from .contagio import explorar_linaje, columna_clave_de_tabla
from .alarmas import AlarmaHadar, AlarmaCard, DialogoAlarma, calcular_valor_actual, evaluar_regla
from .notificaciones import DialogoConfiguracionCorreo, disparar_envio_correo, cargar_configuracion
from .linea_tiempo_ui import LineaTiempoPanel
from .panel import PanelControl
from .prediccion_ui import PanelPrediccion
from .linea_tiempo import parsear_fechas
from .limpieza import (
    detectar_limpieza_sugerida, agrupar_por_tipo,
    aplicar_eliminar_duplicados, aplicar_espacios_en_blanco,
    aplicar_formato_inconsistente, aplicar_quitar_caracteres,
    aplicar_numeros_como_texto,
)
from .dialogos_limpieza import DialogoRevisarCaracteres

# Texto que representa "valores nulos/vacíos" en el filtro por columna de
# Datos y en la pestaña Frecuencias. Se trata siempre aparte (nunca como
# texto real que alguien pueda haber escrito) para no confundir un dato
# real llamado, por ejemplo, "vacío" con una celda efectivamente vacía.
VALOR_VACIO_FILTRO = "(vacíos)"

# Etiquetas legibles para cada "tipo" de anomalía que devuelve
# SemanticAnomalyDetector.detect_all() (ver anomalias.py) -- se usan tanto
# en el combo "Filtrar por anomalía" de la pestaña Datos como en el resumen
# de anomalías de la pestaña Frecuencias, para no mostrarle al usuario la
# clave interna en snake_case.
ETIQUETAS_TIPO_ANOMALIA = {
    "valores_nulos": "Valores nulos",
    "duplicados": "Filas duplicadas",
    "regla_negocio": "Regla de negocio incumplida",
    "temporal": "Inconsistencia temporal",
    "quiebre_patron": "Quiebre de patrón (outlier)",
    "deriva_historica": "Deriva histórica",
    "patron_multivariado": "Combinación inusual (ML)",
}
# Diccionario inverso: etiqueta legible -> clave interna, para traducir de
# vuelta lo que el usuario elige en el combo.
_ETIQUETAS_TIPO_ANOMALIA_INVERSA = {v: k for k, v in ETIQUETAS_TIPO_ANOMALIA.items()}


def _todos_los_indices_anomalos(anomalias_lista):
    """Índices de fila (los mismos que usa el .index del DataFrame de esa
    tabla) de TODAS las anomalías de una lista, sin importar el tipo.
    Usado para armar self._anomalias_por_tabla, que es lo que la
    sub-pestaña Linaje usa para marcar qué filas están conectadas con
    alguna anomalía."""
    indices = set()
    for a in anomalias_lista:
        idxs = a.get("indices_atipicos")
        if idxs is None:
            idx_unico = a.get("fila_indice")
            idxs = [idx_unico] if idx_unico is not None else []
        indices.update(idxs)
    return indices


class _DialogoElegirHojas(QDialog):
    """
    Cuando un Excel tiene varias hojas, pregunta cuáles cargar:
      - Marcar 1 sola hoja: se carga como una sola tabla, y se sigue
        pudiendo cambiar/unir hojas después (como funcionaba antes).
      - Marcar varias hojas: cada una se carga como una tabla DISTINTA
        y relacionada (ej. Clientes, Pedidos, Productos en el mismo
        libro), para que el esquema (Ver esquema) las cruce entre sí.
    """

    def __init__(self, hojas, nombre_archivo, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Elegir hoja(s) — {nombre_archivo}")
        self.resize(380, 320)

        layout = QVBoxLayout(self)

        ayuda = QLabel(
            "Este archivo tiene varias hojas.\n\n"
            "• Marca 1 sola si es una sola tabla (podrás cambiar de hoja "
            "más adelante).\n"
            "• Marca varias si cada hoja es una tabla distinta y "
            "relacionada (ej. Clientes, Pedidos, Productos) — Hadar armará "
            "el esquema sugerido entre ellas."
        )
        ayuda.setWordWrap(True)
        layout.addWidget(ayuda)

        self.lista = QListWidget()
        self.lista.setSelectionMode(QAbstractItemView.NoSelection)
        for h in hojas:
            item = QListWidgetItem(h)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.lista.addItem(item)
        if hojas:
            self.lista.item(0).setCheckState(Qt.Checked)
        layout.addWidget(self.lista, stretch=1)

        fila_botones = QHBoxLayout()
        btn_todas = QPushButton("Marcar todas")
        btn_todas.clicked.connect(self._marcar_todas)
        btn_ninguna = QPushButton("Marcar solo la primera")
        btn_ninguna.clicked.connect(self._marcar_solo_primera)
        fila_botones.addWidget(btn_todas)
        fila_botones.addWidget(btn_ninguna)
        fila_botones.addStretch()
        layout.addLayout(fila_botones)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _marcar_todas(self):
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(Qt.Checked)

    def _marcar_solo_primera(self):
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(Qt.Checked if i == 0 else Qt.Unchecked)

    def _validar_y_aceptar(self):
        if not self.hojas_elegidas():
            QMessageBox.warning(self, "Elige al menos una hoja", "Marca al menos una hoja para continuar.")
            return
        self.accept()

    def hojas_elegidas(self):
        return [
            self.lista.item(i).text()
            for i in range(self.lista.count())
            if self.lista.item(i).checkState() == Qt.Checked
        ]


class _DialogoElegirTablasSql(QDialog):
    """
    Mismo patrón que _DialogoElegirHojas, pero para bases de datos SQL con
    varias tablas: marcar 1 sola tabla se comporta como siempre (una tabla
    suelta); marcar varias las carga todas de una vez, cada una como una
    tabla relacionada distinta, para que el esquema (Ver esquema) las
    cruce entre sí -- antes de este cambio, SQL solo dejaba elegir 1 tabla
    a la vez, así que una base de datos relacional completa por SQL nunca
    armaba el esquema, aunque el mismo archivo con Excel sí lo hacía.
    """

    def __init__(self, tablas, nombre_archivo, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Elegir tabla(s) — {nombre_archivo}")
        self.resize(380, 320)

        layout = QVBoxLayout(self)

        ayuda = QLabel(
            "Esta base de datos SQL tiene varias tablas.\n\n"
            "• Marca 1 sola si solo quieres esa tabla.\n"
            "• Marca varias si son tablas relacionadas (ej. Clientes, Pedidos, "
            "Productos) — Hadar armará el esquema sugerido entre ellas."
        )
        ayuda.setWordWrap(True)
        layout.addWidget(ayuda)

        self.lista = QListWidget()
        self.lista.setSelectionMode(QAbstractItemView.NoSelection)
        for t in tablas:
            item = QListWidgetItem(t)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.lista.addItem(item)
        if tablas:
            self.lista.item(0).setCheckState(Qt.Checked)
        layout.addWidget(self.lista, stretch=1)

        fila_botones = QHBoxLayout()
        btn_todas = QPushButton("Marcar todas")
        btn_todas.clicked.connect(self._marcar_todas)
        btn_ninguna = QPushButton("Marcar solo la primera")
        btn_ninguna.clicked.connect(self._marcar_solo_primera)
        fila_botones.addWidget(btn_todas)
        fila_botones.addWidget(btn_ninguna)
        fila_botones.addStretch()
        layout.addLayout(fila_botones)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _marcar_todas(self):
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(Qt.Checked)

    def _marcar_solo_primera(self):
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(Qt.Checked if i == 0 else Qt.Unchecked)

    def _validar_y_aceptar(self):
        if not self.tablas_elegidas():
            QMessageBox.warning(self, "Elige al menos una tabla", "Marca al menos una tabla para continuar.")
            return
        self.accept()

    def tablas_elegidas(self):
        return [
            self.lista.item(i).text()
            for i in range(self.lista.count())
            if self.lista.item(i).checkState() == Qt.Checked
        ]


class _DialogoConexionSqlServer(QDialog):
    """Pide los datos para conectarse a un SQL Server DE VERDAD -- local o
    de otro PC en la red -- a diferencia de "Cargar Archivo → .sql", que
    solo lee un archivo de texto con instrucciones SQL, sin conectarse a
    ningún servidor. Por seguridad, la contraseña NUNCA se guarda en el
    proyecto -- este mismo diálogo se reusa también para "Actualizar desde
    la fuente", pidiéndola de nuevo cada vez."""

    def __init__(self, parent=None, valores_iniciales=None):
        super().__init__(parent)
        self.setWindowTitle("Conectar a SQL Server")
        self.resize(360, 260)
        valores_iniciales = valores_iniciales or {}

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.txt_servidor = QLineEdit(valores_iniciales.get("servidor", ""))
        self.txt_servidor.setPlaceholderText("ej. 192.168.1.15 o NOMBRE-PC")
        form.addRow("Servidor:", self.txt_servidor)

        self.txt_puerto = QLineEdit(str(valores_iniciales.get("puerto", 1433)))
        form.addRow("Puerto:", self.txt_puerto)

        self.txt_base_datos = QLineEdit(valores_iniciales.get("base_datos", ""))
        form.addRow("Base de datos:", self.txt_base_datos)

        self.txt_usuario = QLineEdit(valores_iniciales.get("usuario", ""))
        form.addRow("Usuario:", self.txt_usuario)

        self.txt_password = QLineEdit()
        self.txt_password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Contraseña:", self.txt_password)

        layout.addLayout(form)

        aviso = QLabel(
            "La contraseña no se guarda -- se vuelve a pedir cada vez que "
            "conectas o actualizas."
        )
        aviso.setWordWrap(True)
        aviso.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(aviso)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def valores(self):
        try:
            puerto = int(self.txt_puerto.text().strip())
        except ValueError:
            puerto = 1433
        return {
            "servidor": self.txt_servidor.text().strip(),
            "puerto": puerto,
            "base_datos": self.txt_base_datos.text().strip(),
            "usuario": self.txt_usuario.text().strip(),
            "password": self.txt_password.text(),
        }


class _DialogoConexionPos(QDialog):
    """Pide (o confirma) la dirección del POS en la red local. A diferencia
    de _DialogoConexionSqlServer, acá NO se pide contraseña/API key en el
    diálogo -- esa clave vive en config.py de Analytics (variable de
    entorno ANALYTICS_API_KEY_POS), fija para este PC, igual que la clave
    de Turso vive en el config.py del POS. Solo la URL es editable acá,
    por si la IP o el nombre de red del PC del POS cambia."""

    def __init__(self, parent=None, valores_iniciales=None):
        super().__init__(parent)
        self.setWindowTitle("Conectar al POS")
        self.resize(360, 200)
        valores_iniciales = valores_iniciales or {}

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.txt_url = QLineEdit(valores_iniciales.get("url_base", URL_POS_SINCRONIZACION))
        self.txt_url.setPlaceholderText("ej. http://192.168.1.100:8000")
        form.addRow("Dirección del POS:", self.txt_url)
        layout.addLayout(form)

        if not API_KEY_POS_SINCRONIZACION:
            aviso_clave = QLabel(
                "⚠ No hay una API key configurada en este PC (variable de "
                "entorno ANALYTICS_API_KEY_POS). La conexión va a fallar "
                "hasta que se configure."
            )
            aviso_clave.setWordWrap(True)
            aviso_clave.setStyleSheet("color: #F59E0B; font-size: 11px;")
            layout.addWidget(aviso_clave)

        self.lbl_estado_prueba = QLabel("")
        self.lbl_estado_prueba.setWordWrap(True)
        layout.addWidget(self.lbl_estado_prueba)

        btn_probar = QPushButton("Probar conexión")
        btn_probar.clicked.connect(self._probar_conexion)
        layout.addWidget(btn_probar)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _probar_conexion(self):
        url = self.txt_url.text().strip()
        if verificar_conexion_pos(url):
            self.lbl_estado_prueba.setText("✓ El POS responde en esa dirección.")
            self.lbl_estado_prueba.setStyleSheet("color: #34D399;")
        else:
            self.lbl_estado_prueba.setText(
                "✗ No se pudo contactar al POS ahí. Revisa que esté encendido "
                "y conectado a la misma red wifi."
            )
            self.lbl_estado_prueba.setStyleSheet("color: #EF4444;")

    def valores(self):
        return {"url_base": self.txt_url.text().strip()}


class HadarApp(QMainWindow):
    def __init__(self, modo="nuevo"):
        super().__init__()
        self.setWindowTitle("Hadar Data Analytics Pro")
        ANCHO_DISENO, ALTO_DISENO = 1300, 850
        pantalla = QApplication.primaryScreen()
        geo_disponible = pantalla.availableGeometry() if pantalla else None
        if geo_disponible is not None and (
            geo_disponible.width() < ANCHO_DISENO or geo_disponible.height() < ALTO_DISENO
        ):
            # La ventana se diseñó pensando en 1300x850, pero si la pantalla
            # es más chica que eso (ej. muchos notebooks vienen en 1280x720),
            # pedirle ese tamaño obliga a Windows a achicarla por su cuenta
            # al abrirla -- y el contenido interno, que ya se armó pensando
            # en el tamaño grande, queda con partes cortadas hasta forzar un
            # resize real (minimizar/maximizar). Para evitar ese primer
            # desajuste, en pantallas chicas se abre directamente maximizada:
            # así el primer tamaño que ve el contenido ya es el definitivo.
            self.resize(
                min(ANCHO_DISENO, geo_disponible.width()),
                min(ALTO_DISENO, geo_disponible.height()),
            )
            self.setWindowState(Qt.WindowMaximized)
        else:
            self.resize(ANCHO_DISENO, ALTO_DISENO)
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        # De dónde viene esta sesión: "nuevo" (proyecto pensado para
        # guardarse y usarse seguido), "casual" (análisis rápido y
        # desechable, nunca se guarda ni tiene ML) o "continuar" (ver
        # abrir_proyecto_desde_ruta, que llega desde la Pantalla de Inicio).
        self.modo_sesion = modo
        # None = todavía no se decide si este proyecto tiene aprendizaje
        # adaptativo. Se pregunta una sola vez, en el primer "Guardar
        # Proyecto" (ver _guardar_proyecto), y de ahí en adelante queda
        # fijo para este proyecto -- o se carga directo del archivo si el
        # modo es "continuar" (ver abrir_proyecto_desde_ruta).
        self.ml_activado = None
        # Línea base adaptativa acumulada de este proyecto (ver
        # aprendizaje_adaptativo.py) -- vacía hasta que se guarda o
        # reabre un proyecto con ml_activado=True.
        self.linea_base_ml = {}
        # Detección multivariada (Isolation Forest, ver deteccion_multivariada.py):
        # opt-in por dataset, se decide en la pestaña Narrativa (botón con el
        # termómetro de confiabilidad) -- a diferencia de ml_activado, no se
        # pregunta una sola vez: se puede prender/apagar en cualquier momento.
        self.ml_multivariado_activado = False
        # Configuración de la pestaña Predicción (qué predecir, qué columnas usar).
        # Se guarda en el .hadarproy; el modelo entrenado no (ver prediccion.py).
        self.prediccion_config = {}
        # De dónde vino cada tabla ({nombre_tabla: {"tipo": "archivo"|"sql_server", ...}})
        # -- para poder "Actualizar" sin volver a preguntar todo. Nunca
        # incluye contraseñas (ver _DialogoConexionSqlServer).
        self.fuentes_datos = {}
        # Historia de las columnas (de dónde salió cada una): hoy, cuáles
        # calculó Hadar y con qué fórmula. Se guarda dentro del .hadarproy y
        # alimenta el capítulo "Origen de los datos" de Narrativa. Ver procedencia.py.
        self.procedencia = BitacoraProcedencia()

        self.theme_name = "tactical"
        self.colors = THEMES[self.theme_name]

        self.df = None
        self.filtered_df = None
        self.filtro_grafico = None
        # Filtro "por anomalía" de la pestaña Datos: None = sin filtro,
        # "todas" = cualquier anomalía, o la clave interna de un tipo
        # puntual (ver ETIQUETAS_TIPO_ANOMALIA). Se apoya en
        # self._ultimas_anomalias, que solo tiene contenido después de
        # generar el informe en Narrativa -- ver _indices_filtro_anomalia().
        self.filtro_tipo_anomalia = None

        # Ontología liviana (Paso B/C): además de self.df (la tabla activa,
        # que sigue alimentando Gráficos/Métricas/Indicadores/etc. IGUAL que
        # antes), se guardan todas las tablas cargadas en self.tablas para
        # poder mostrar el esquema sugerido entre ellas cuando hay 2 o más.
        # Con 1 sola tabla, self.tablas queda con 1 solo elemento y nada de
        # esto se nota: el botón "Ver esquema" queda oculto.
        self.tablas = {}
        self.nombre_tabla_activa = None
        # Relaciones del esquema, tal como quedaron la última vez que el
        # usuario abrió "Ver esquema" (con lo que haya agregado/borrado a
        # mano incluido). None = todavía no se ha abierto esa ventana en
        # esta carga; Narrativa, en ese caso, las infiere de nuevo solo
        # para el cruce entre tablas, sin guardar nada.
        self.relaciones_ontologia = None

        # Alarmas: reglas persistentes (SQLite) sobre Indicadores, Métricas
        # y Frecuencias. Ver alarmas.py.
        self.alarma_memoria = AlarmaHadar()
        self.reglas_alarma = self.alarma_memoria.listar_reglas()
        self.estado_previo_alarmas = {}  # id de regla -> estaba en alarma sí/no
        self._hilos_correo = []  # referencias vivas a hilos de envío de correo

        self.excel_path = None
        self.excel_sheet_names = []
        self.excel_hojas_activas = []
        self.uf_val = 37000.0
        self.chart_panels = []
        self._uf_thread = None
        self._uf_worker = None

        # Memoria organizacional persistente (ver clase MemoriaHadar arriba).
        # Vive en la carpeta de datos del usuario, no junto al ejecutable.
        self.memoria = MemoriaHadar()
        self.fingerprint_actual = None

        # Rango activo de la Línea de Tiempo: None = sin recortar, o
        # (t0: Timestamp, t1: Timestamp, columna: str, dayfirst: bool). Lo
        # consume _recortar_por_rango_tiempo() (Datos, Métricas, etc. vía
        # apply_table_filter) y la vista de puntos de la propia Línea de
        # Tiempo. dayfirst es el formato con que la Línea de Tiempo leyó esa
        # columna: se guarda acá para que el filtro lea las fechas IGUAL.
        self.rango_tiempo = None
        # Fechas ya interpretadas de la columna del rango (ver _fechas_de_columna).
        self._cache_fechas = None

        # Filas específicas buscadas a mano en Datos (ej. "2000, 3000, 27, 8")
        # para comparar registros puntuales. None = sin buscador activo.
        # Mientras esté activo, ignora a propósito los demás filtros de la
        # pestaña Datos (columna, gráfico, línea de tiempo) para no mezclar
        # dos criterios de filtrado distintos sin que el usuario lo note.
        self.filas_buscadas = None

        # Firma del estado del proyecto al último guardar/abrir -- se
        # compara contra el estado actual para saber si hay cambios sin
        # guardar (ver _hay_cambios_sin_guardar), sin tener que llevar la
        # cuenta manual de cada edición por separado.
        self._ultimo_guardado_firma = None

        central = QWidget()
        self.setCentralWidget(central)
        self._central_widget = central
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._build_sidebar(root_layout)
        self._build_main_area(root_layout)
        self.apply_theme(self.theme_name)

        if self.modo_sesion == "casual":
            # Un "Análisis Casual" es desechable a propósito: sin botón de
            # Guardar Proyecto no hay tentación de guardarlo a medias, y
            # así nunca termina teniendo un archivo .hadarproy ni ML.
            self.btn_guardar_proyecto.setVisible(False)

    # ------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------
    ANCHO_SIDEBAR_EXPANDIDO = 280
    ANCHO_SIDEBAR_COLAPSADO = 40

    def _build_sidebar(self, root_layout):
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(self.ANCHO_SIDEBAR_EXPANDIDO)

        # El contenido real vive dentro de un QScrollArea: así, si a futuro se
        # agregan más secciones (o la ventana queda muy baja), la barra hace
        # scroll en vez de apretujar o cortar controles.
        outer_layout = QVBoxLayout(self.sidebar)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # --- Header: logo + título -----------------------------------
        # A propósito FUERA del QScrollArea de abajo: así queda siempre
        # arriba y fijo, sin importar cuántas secciones plegables estén
        # abiertas/cerradas ni cuánto se scrollee el resto. Se oculta junto
        # con self.sidebar_scroll cuando la barra se colapsa (ver
        # _toggle_sidebar) porque a 40px de ancho el texto no entra igual.
        self.sidebar_header = QWidget()
        header_layout = QHBoxLayout(self.sidebar_header)
        header_layout.setContentsMargins(20, 20, 20, 10)
        header_layout.setSpacing(10)
        if os.path.exists(LOGO_PNG_PATH):
            self.lbl_logo_header = QLabel()
            header_layout.addWidget(self.lbl_logo_header)
            self._actualizar_logo_tema()
        else:
            # Si falta el archivo del logo, se muestra el texto de siempre
            title = QLabel("HADAR ANALYTICS")
            title.setObjectName("title")
            header_layout.addWidget(title)
        header_layout.addStretch()
        outer_layout.addWidget(self.sidebar_header)
        self.sidebar_scroll = QScrollArea()
        scroll = self.sidebar_scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        # Botón para plegar/desplegar. A propósito NO va metido en
        # outer_layout junto al resto (eso lo empujaba a distinta altura
        # según si el contenido estaba visible u oculto) -- es hijo directo
        # de self.sidebar, posicionado a mano, para que quede siempre
        # centrado verticalmente y en el mismo lugar exacto en los dos
        # estados (da la sensación de ser un solo botón que solo cambia
        # de flecha, no dos botones distintos).
        self.btn_colapsar_sidebar = QPushButton("◀", self._central_widget)
        self.btn_colapsar_sidebar.setObjectName("sidebarToggleBtn")
        self.btn_colapsar_sidebar.setFixedSize(20, 44)
        self.btn_colapsar_sidebar.setToolTip("Ocultar/mostrar la barra lateral")
        self.btn_colapsar_sidebar.clicked.connect(self._toggle_sidebar)
        self.btn_colapsar_sidebar.raise_()
        self._reposicionar_boton_sidebar()

        contenido = QWidget()
        scroll.setWidget(contenido)
        sidebar_layout = QVBoxLayout(contenido)
        sidebar_layout.setContentsMargins(20, 10, 20, 20)
        sidebar_layout.setSpacing(10)
        # Layout "raíz" de la barra: las secciones plegables (ver
        # _crear_seccion_plegable) reasignan la variable local
        # sidebar_layout a su propio contenido mientras se arman, así que
        # el código de cada sección de más abajo no cambia ni una línea --
        # solo hay que volver a este layout_raiz al terminar cada sección.
        sidebar_layout_raiz = sidebar_layout
        sidebar_layout_raiz.setAlignment(Qt.AlignTop)

        # --- Sección: Datos ---------------------------------------------
        sidebar_layout = self._crear_seccion_plegable(sidebar_layout_raiz, "DATOS", abierta=True)

        btn_load = QPushButton("Cargar Archivo")
        btn_load.setObjectName("accentButton")
        btn_load.clicked.connect(self.on_load_file)
        sidebar_layout.addWidget(btn_load)

        fila_sql_server = QHBoxLayout()
        btn_conectar_sql_server = QPushButton("Conectar a SQL Server...")
        btn_conectar_sql_server.setToolTip(
            "Se conecta EN VIVO a un SQL Server (local o de otro PC en la "
            "red) -- distinto de cargar un archivo .sql suelto."
        )
        btn_conectar_sql_server.clicked.connect(self._conectar_sql_server)
        fila_sql_server.addWidget(btn_conectar_sql_server)

        self.btn_actualizar_fuente = QPushButton("↻ Actualizar")
        self.btn_actualizar_fuente.setToolTip(
            "Vuelve a traer los datos de donde vinieron (archivo, SQL "
            "Server o POS) para la tabla activa. Disponible solo si esa "
            "tabla vino de un archivo o de una conexión en vivo."
        )
        self.btn_actualizar_fuente.clicked.connect(self._actualizar_desde_la_fuente)
        self.btn_actualizar_fuente.setEnabled(False)
        fila_sql_server.addWidget(self.btn_actualizar_fuente)
        sidebar_layout.addLayout(fila_sql_server)

        btn_conectar_pos = QPushButton("Conectar al POS...")
        btn_conectar_pos.setToolTip(
            "Se conecta EN VIVO al PC del POS en la misma red wifi, para "
            "traer sus tablas de ventas, productos, fiado, etc."
        )
        btn_conectar_pos.clicked.connect(self._conectar_pos)
        sidebar_layout.addWidget(btn_conectar_pos)

        # "Chip" con el archivo cargado: reemplaza el texto suelto por una
        # pequeña tarjeta, para que se lea como un dato de estado y no como
        # una nota perdida entre botones.
        archivo_chip = QFrame()
        archivo_chip.setObjectName("fileChip")
        archivo_chip_layout = QHBoxLayout(archivo_chip)
        archivo_chip_layout.setContentsMargins(10, 8, 10, 8)
        archivo_chip_layout.setSpacing(8)
        self.lbl_archivo = QLabel("Ningún archivo cargado.")
        self.lbl_archivo.setObjectName("muted")
        self.lbl_archivo.setWordWrap(True)
        archivo_chip_layout.addWidget(self.lbl_archivo, stretch=1)
        sidebar_layout.addWidget(archivo_chip)

        # --- Proyecto: guardar/continuar el trabajo en curso -------------
        fila_proyecto = QHBoxLayout()
        btn_guardar_proyecto = QPushButton("Guardar Proyecto")
        btn_guardar_proyecto.setToolTip(
            "Guarda tus tablas, filtro, esquema, indicadores y notas en un "
            "archivo .hadarproy para poder seguir otro día donde quedaste."
        )
        btn_guardar_proyecto.clicked.connect(self._guardar_proyecto)
        fila_proyecto.addWidget(btn_guardar_proyecto)
        self.btn_guardar_proyecto = btn_guardar_proyecto

        btn_abrir_proyecto = QPushButton("Abrir Proyecto")
        btn_abrir_proyecto.setToolTip("Continúa un análisis guardado antes con \"Guardar Proyecto\".")
        btn_abrir_proyecto.clicked.connect(self._abrir_proyecto)
        fila_proyecto.addWidget(btn_abrir_proyecto)
        sidebar_layout.addLayout(fila_proyecto)

        # Hojas de Excel (solo visible con archivos .xlsx/.xls de varias
        # hojas): sub-panel con acento a la izquierda para que se lea como
        # contenido desplegado de "Cargar Archivo", no como una sección aparte.
        self.excel_frame = QFrame()
        self.excel_frame.setObjectName("subPanel")
        excel_layout = QVBoxLayout(self.excel_frame)
        excel_layout.setContentsMargins(10, 10, 10, 10)
        excel_layout.setSpacing(6)

        lbl_hoja = QLabel("Hoja de Excel")
        lbl_hoja.setObjectName("muted")
        excel_layout.addWidget(lbl_hoja)
        self.combo_hoja_excel = QComboBox()
        self.combo_hoja_excel.currentTextChanged.connect(self._on_cambiar_hoja_excel)
        excel_layout.addWidget(self.combo_hoja_excel)

        self.btn_unir_hoja = QPushButton("+ Unir otra hoja")
        self.btn_unir_hoja.clicked.connect(self._on_unir_hoja_excel)
        excel_layout.addWidget(self.btn_unir_hoja)

        self.lbl_hojas_estado = QLabel("")
        self.lbl_hojas_estado.setObjectName("muted")
        self.lbl_hojas_estado.setWordWrap(True)
        excel_layout.addWidget(self.lbl_hojas_estado)

        self.excel_frame.setVisible(False)
        sidebar_layout.addWidget(self.excel_frame)

        # --- Sección: Herramientas ---------------------------------------
        sidebar_layout = self._crear_seccion_plegable(sidebar_layout_raiz, "HERRAMIENTAS", abierta=False)

        btn_excel = QPushButton("Calcular nueva columna")
        btn_excel.setObjectName("accentButton")
        btn_excel.setToolTip(
            "Crea una columna nueva combinando otras columnas con una fórmula "
            "(como en Excel). Así cualquier cálculo queda disponible para "
            "graficar, no solo para verse como tarjeta."
        )
        btn_excel.clicked.connect(self.abrir_transformacion_excel)
        sidebar_layout.addWidget(btn_excel)

        # Conversor UF/CLP
        self.uf_toggle_btn = QPushButton("Conversor UF/CLP: Desactivado")
        self.uf_toggle_btn.setCheckable(True)
        self.uf_toggle_btn.clicked.connect(self._on_uf_toggle)
        sidebar_layout.addWidget(self.uf_toggle_btn)

        self.uf_frame = QFrame()
        self.uf_frame.setObjectName("subPanel")
        uf_layout = QVBoxLayout(self.uf_frame)
        uf_layout.setContentsMargins(10, 10, 10, 10)
        uf_layout.setSpacing(6)

        mode_row = QHBoxLayout()
        self.radio_auto = QRadioButton("Auto")
        self.radio_manual = QRadioButton("Manual")
        self.radio_auto.setChecked(True)
        self.uf_mode_group = QButtonGroup(self)
        self.uf_mode_group.addButton(self.radio_auto)
        self.uf_mode_group.addButton(self.radio_manual)
        self.radio_auto.toggled.connect(self._on_uf_mode_change)
        mode_row.addWidget(self.radio_auto)
        mode_row.addWidget(self.radio_manual)
        uf_layout.addLayout(mode_row)

        self.uf_entry = QLineEdit("37000")
        self.uf_entry.setVisible(False)
        uf_layout.addWidget(self.uf_entry)

        self.uf_col_combo = QComboBox()
        uf_layout.addWidget(self.uf_col_combo)

        dir_row = QHBoxLayout()
        self.radio_clp_uf = QRadioButton("CLP a UF")
        self.radio_uf_clp = QRadioButton("UF a CLP")
        self.radio_clp_uf.setChecked(True)
        self.uf_dir_group = QButtonGroup(self)
        self.uf_dir_group.addButton(self.radio_clp_uf)
        self.uf_dir_group.addButton(self.radio_uf_clp)
        dir_row.addWidget(self.radio_clp_uf)
        dir_row.addWidget(self.radio_uf_clp)
        uf_layout.addLayout(dir_row)

        self.btn_uf_apply = QPushButton("Convertir")
        self.btn_uf_apply.clicked.connect(self.apply_uf_conversion)
        uf_layout.addWidget(self.btn_uf_apply)

        self.lbl_uf_status = QLabel("")
        self.lbl_uf_status.setObjectName("muted")
        uf_layout.addWidget(self.lbl_uf_status)

        self.uf_frame.setVisible(False)
        sidebar_layout.addWidget(self.uf_frame)

        sidebar_layout.addStretch()

        # --- Sección: Buscar filas (ancla arriba de Apariencia) -----------
        sidebar_layout = self._crear_seccion_plegable(sidebar_layout_raiz, "BUSCAR FILAS", abierta=False)

        self.datos_buscar_filas = QLineEdit()
        self.datos_buscar_filas.setPlaceholderText("Ej: 2000, 3000, 27, 8")
        self.datos_buscar_filas.setToolTip(
            "Escribe números de fila separados por coma -- los mismos que ves "
            "en la columna de la izquierda de la tabla de Datos -- y presiona "
            "Enter. Sirve para comparar registros puntuales lejos entre sí "
            "(ej. la fila 2000 y la fila 27) sin desplazarte manualmente. "
            "Afecta a toda la app (Datos, Métricas, Indicadores, "
            "Frecuencias), no solo a la pestaña Datos.\n\n"
            "Mientras esté activo, reemplaza el filtro por columna y el de "
            "línea de tiempo, para no mezclar dos criterios a la vez."
        )
        self.datos_buscar_filas.returnPressed.connect(self._buscar_filas_especificas)
        sidebar_layout.addWidget(self.datos_buscar_filas)

        fila_botones_busqueda = QHBoxLayout()
        btn_buscar_filas = QPushButton("Buscar")
        btn_buscar_filas.clicked.connect(self._buscar_filas_especificas)
        fila_botones_busqueda.addWidget(btn_buscar_filas)
        self.btn_limpiar_busqueda_filas = QPushButton("Quitar")
        self.btn_limpiar_busqueda_filas.setVisible(False)
        self.btn_limpiar_busqueda_filas.clicked.connect(self._limpiar_busqueda_filas)
        fila_botones_busqueda.addWidget(self.btn_limpiar_busqueda_filas)
        sidebar_layout.addLayout(fila_botones_busqueda)

        # --- Sección: Limpieza sugerida -----------------------------------
        # A propósito NO toca ninguna anomalía de anomalias.py/Narrativa --
        # esto es solo suciedad técnica (duplicados, espacios de más,
        # formatos inconsistentes). Nunca modifica el dataset: solo señala
        # y deja navegar, la corrección la hace la persona a mano.
        sidebar_layout = self._crear_seccion_plegable(sidebar_layout_raiz, "LIMPIEZA SUGERIDA", abierta=False)

        self.btn_limpieza_sugerida = QPushButton("Buscar suciedad en los datos")
        self.btn_limpieza_sugerida.setToolTip(
            "Busca duplicados exactos, espacios de más, formatos inconsistentes "
            "(ej. 'Chile' / 'chile ' / 'CHILE'), caracteres especiales y números "
            "escritos como texto (ej. 'cuatro', '6 unidades') en la tabla activa. "
            "Marca las celdas afectadas en rojo -- distinto del rojo de anomalías "
            "de Narrativa, que puede ser un problema real de negocio, no basura "
            "para limpiar.\n\n"
            "Nunca modifica los datos solo: señala, y podés elegir aplicar cada "
            "categoría por separado."
        )
        self.btn_limpieza_sugerida.clicked.connect(self._buscar_limpieza_sugerida)
        sidebar_layout.addWidget(self.btn_limpieza_sugerida)

        # Contenedor donde se arman, dinámicamente, los botones "Aplicar"
        # por categoría (duplicados / espacios / formato / caracteres) --
        # se reconstruye cada vez que se busca o se aplica una corrección.
        self.limpieza_categorias_widget = QWidget()
        self.limpieza_categorias_layout = QVBoxLayout(self.limpieza_categorias_widget)
        self.limpieza_categorias_layout.setContentsMargins(0, 4, 0, 4)
        self.limpieza_categorias_layout.setSpacing(4)
        sidebar_layout.addWidget(self.limpieza_categorias_widget)

        self.limpieza_lista = QListWidget()
        self.limpieza_lista.setVisible(False)
        self.limpieza_lista.setToolTip("Haz clic en un hallazgo para ir directo a esa fila en Datos.")
        self.limpieza_lista.itemClicked.connect(self._ir_a_hallazgo_limpieza)
        sidebar_layout.addWidget(self.limpieza_lista)

        self.lbl_limpieza_estado = QLabel("")
        self.lbl_limpieza_estado.setObjectName("muted")
        self.lbl_limpieza_estado.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_limpieza_estado)

        self.btn_limpiar_marcas_limpieza = QPushButton("Quitar marcas")
        self.btn_limpiar_marcas_limpieza.setVisible(False)
        self.btn_limpiar_marcas_limpieza.clicked.connect(self._quitar_marcas_limpieza)
        sidebar_layout.addWidget(self.btn_limpiar_marcas_limpieza)

        # --- Sección: Apariencia (anclada abajo) --------------------------
        sidebar_layout = self._crear_seccion_plegable(sidebar_layout_raiz, "APARIENCIA", abierta=False)

        self.botones_tema = {}
        for clave, etiqueta in (
            ("light", "Blanco"),
            ("gray", "Gris Claro"),
            ("dark", "Azul Oscuro"),
            ("tactical", "Negro Táctico"),
        ):
            btn_tema = QPushButton(etiqueta)
            btn_tema.setCheckable(True)
            btn_tema.setChecked(clave == self.theme_name)
            btn_tema.clicked.connect(lambda _checked, clave=clave: self._seleccionar_tema(clave))
            sidebar_layout.addWidget(btn_tema)
            self.botones_tema[clave] = btn_tema

        # "Salir" queda SIEMPRE visible, fuera de cualquier acordeón --
        # tenerlo escondido detrás de "Apariencia" plegada sería mala idea.
        # El addStretch() de acá es lo que ancla Salir siempre abajo del
        # todo y, de paso, mantiene las secciones pegadas arriba (sin él,
        # con la mayoría de las secciones plegadas, el contenido quedaba
        # más corto que el alto de la barra y Qt lo repartía raro).
        sidebar_layout_raiz.addStretch(1)
        sidebar_layout_raiz.addWidget(self._divider())
        btn_salir = QPushButton("✕  Salir")
        btn_salir.setObjectName("dangerButton")
        btn_salir.setToolTip("Cierra Hadar. Si hay cambios sin guardar, te lo va a preguntar antes.")
        btn_salir.clicked.connect(self.close)
        sidebar_layout_raiz.addWidget(btn_salir)

        root_layout.addWidget(self.sidebar)

    def _crear_seccion_plegable(self, layout_padre, titulo, abierta=False):
        """Crea una sección plegable de la barra lateral (ej. "APARIENCIA"):
        un encabezado en el que se hace clic para mostrar/ocultar su
        contenido. Devuelve el QVBoxLayout donde va el contenido de la
        sección -- se agrega ya a layout_padre, así que el código que la
        llama solo necesita reasignar su variable local `sidebar_layout` a
        lo que esto devuelve y seguir agregando widgets normalmente."""
        layout_padre.addWidget(self._divider())

        header = QPushButton(f"▸  {titulo}")
        header.setObjectName("sectionHeader")
        header.setCheckable(True)
        header.setChecked(abierta)
        header.setCursor(Qt.PointingHandCursor)
        layout_padre.addWidget(header)

        contenedor = QWidget()
        contenido_layout = QVBoxLayout(contenedor)
        contenido_layout.setContentsMargins(4, 6, 0, 6)
        contenido_layout.setSpacing(8)
        contenedor.setVisible(abierta)
        layout_padre.addWidget(contenedor)

        def _alternar(checked, header=header, contenedor=contenedor, titulo=titulo):
            contenedor.setVisible(checked)
            header.setText(f"{'▾' if checked else '▸'}  {titulo}")

        header.toggled.connect(_alternar)
        return contenido_layout

    def _seleccionar_tema(self, theme_name):
        self.apply_theme(theme_name)
        for clave, boton in self.botones_tema.items():
            boton.setChecked(clave == theme_name)

    def _refrescar_todas_las_pestanas(self):
        """Fuerza que los PlotWidget de TODAS las pestañas (la visible y las
        ocultas) recalculen su viewport. Qt no entrega resizeEvent a un
        widget mientras está oculto -- así que cuando la barra lateral
        cambia de ancho, la ventana recién se abre, o cambia de tamaño
        mientras el usuario está en otra pestaña, los gráficos de las
        pestañas ocultas quedan con su geometría interna desactualizada
        hasta que algo los "despierta" a mano.

        OJO: redimensionar un widget a su MISMO tamaño (ej. plot.resize(
        plot.size())) no sirve -- Qt detecta que el tamaño no cambió y ni
        siquiera dispara el resizeEvent interno, así que no arregla nada.
        Por eso acá se cambia el ancho en 1px y se vuelve al original: eso
        sí garantiza dos resizeEvent reales, que es lo que pyqtgraph
        necesita para recalcular su viewport."""
        for i in range(self.tabview.count()):
            widget = self.tabview.widget(i)
            if widget is None:
                continue
            widget.updateGeometry()
            for plot in widget.findChildren(pg.PlotWidget):
                tam = plot.size()
                plot.resize(tam.width() + 1, tam.height())
                plot.resize(tam)
                plot.getViewBox().updateAutoRange()
        QApplication.processEvents()

    def showEvent(self, event):
        super().showEvent(event)
        # El primer acomodo de tamaño real de la ventana (al abrir Hadar,
        # o al restaurarla desde minimizada) puede dejar los gráficos de
        # las pestañas que no están activas con el viewport mal calculado
        # -- se ve igual que el "se corta" reportado, pero sin que el
        # usuario haya tocado nada todavía. QTimer.singleShot(0, ...) para
        # que corra apenas la ventana termine de asentar su geometría.
        QTimer.singleShot(0, self._refrescar_todas_las_pestanas)

    def _reposicionar_boton_sidebar(self):
        """Mantiene el botón de plegar pegado al borde derecho de la barra
        (asomándose un poco hacia el área de contenido) y un poco más abajo
        del centro vertical -- se llama al construir, en cada cuadro de la
        animación (el ancho cambia) y al redimensionar la ventana (el alto
        cambia). Así siempre queda en el mismo lugar relativo, sea cual sea
        el estado, dando la sensación de un solo botón.

        Es hijo del widget central (no de la barra) a propósito: así puede
        asomarse un poco más allá del borde de la barra sin que esa mitad
        quede recortada -- un widget hijo nunca se pinta fuera de los
        límites de su padre."""
        if not hasattr(self, "btn_colapsar_sidebar"):
            return
        borde_derecho_sidebar = self.sidebar.x() + self.sidebar.width()
        x = borde_derecho_sidebar - self.btn_colapsar_sidebar.width() // 2
        y = max(0, (self.sidebar.height() - self.btn_colapsar_sidebar.height()) // 2) + 40
        self.btn_colapsar_sidebar.move(x, y)
        self.btn_colapsar_sidebar.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposicionar_boton_sidebar()

    def _firma_estado_actual(self):
        """Firma liviana del estado actual del proyecto (tablas + filtro +
        notas + indicadores + relaciones), para saber si hay cambios sin
        guardar sin tener que llevar la cuenta manual de cada edición por
        separado -- se compara contra la firma del último guardado/abierto."""
        partes = []
        for nombre in sorted(self.tablas.keys()):
            df = self.tablas[nombre]
            try:
                hash_tabla = int(pd.util.hash_pandas_object(df, index=True).sum())
            except Exception:
                hash_tabla = df.shape  # mejor un valor aproximado que reventar acá
            partes.append(f"{nombre}:{df.shape}:{hash_tabla}")
        partes.append(f"activa:{self.nombre_tabla_activa}")
        if hasattr(self, "datos_filter_col"):
            partes.append(f"filtro:{self.datos_filter_col.currentText()}")
        if hasattr(self, "table_model"):
            partes.append(f"notas:{sorted(self.table_model.notas_manuales().items(), key=str)}")
        partes.append(f"indicadores:{[ind.to_dict() for ind in self.indicadores]}")
        partes.append(f"relaciones:{self.relaciones_ontologia}")
        partes.append(f"procedencia:{self.procedencia.a_lista()}")
        texto = "|".join(str(p) for p in partes)
        return hashlib.sha256(texto.encode("utf-8", errors="ignore")).hexdigest()

    def _hay_cambios_sin_guardar(self):
        if not self.tablas:
            return False  # nada cargado, nada que perder
        return self._firma_estado_actual() != self._ultimo_guardado_firma

    def closeEvent(self, event):
        """Antes de cerrar Hadar de verdad (por el botón "✕ Salir" de la
        barra, la X de la ventana, o Alt+F4 -- las tres formas pasan por
        acá), si hay cambios sin guardar como proyecto, se lo pregunta."""
        if self._hay_cambios_sin_guardar():
            respuesta = QMessageBox.question(
                self, "Cambios sin guardar",
                "Este análisis tiene cambios que no se guardaron como proyecto.\n\n"
                "¿Querés guardarlos antes de salir?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if respuesta == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if respuesta == QMessageBox.StandardButton.Save:
                if not self._guardar_proyecto():
                    event.ignore()  # se canceló el diálogo de guardar, o falló -- no salir
                    return
        event.accept()

    def _toggle_sidebar(self):
        """Pliega/despliega la barra lateral con una animación deslizante.
        Al plegar, el contenido se oculta de inmediato (antes de que termine
        de encogerse, para no ver texto apretujado a mitad de camino); al
        desplegar, el contenido reaparece recién cuando termina de crecer."""
        # Si el usuario hace clic de nuevo mientras la animación anterior
        # todavía está corriendo (doble clic, clics rápidos), sin este freno
        # quedaban dos QVariantAnimation compitiendo por el mismo ancho al
        # mismo tiempo -- el resultado final dependía de cuál terminara al
        # último, y a veces la barra quedaba a mitad de camino, achicando
        # el área de las pestañas de forma permanente hasta reiniciar Hadar.
        if getattr(self, "_animacion_sidebar", None) is not None:
            self._animacion_sidebar.stop()

        expandido_ahora = getattr(self, "_sidebar_expandido", True)
        self._sidebar_expandido = not expandido_ahora
        self.btn_colapsar_sidebar.setText("◀" if self._sidebar_expandido else "▶")

        # El fondo de la barra cambia mientras está colapsada: cuando no
        # queda contenido adentro, que se vea como una extensión lisa del
        # fondo de la ventana (azul oscuro / blanco), no como un panel
        # aparte -- ver QFrame#sidebar[colapsado="true"] en graficos.py.
        self.sidebar.setProperty("colapsado", "false" if self._sidebar_expandido else "true")
        self.sidebar.style().unpolish(self.sidebar)
        self.sidebar.style().polish(self.sidebar)

        if not self._sidebar_expandido:
            self.sidebar_scroll.setVisible(False)
            self.sidebar_header.setVisible(False)

        animacion = QVariantAnimation(self)
        animacion.setStartValue(self.sidebar.width())
        animacion.setEndValue(
            self.ANCHO_SIDEBAR_EXPANDIDO if self._sidebar_expandido else self.ANCHO_SIDEBAR_COLAPSADO
        )
        animacion.setDuration(180)
        animacion.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def _en_cada_cuadro(v):
            self.sidebar.setFixedWidth(int(v))
            self._reposicionar_boton_sidebar()

        animacion.valueChanged.connect(_en_cada_cuadro)
        if self._sidebar_expandido:
            animacion.finished.connect(lambda: self.sidebar_scroll.setVisible(True))
            animacion.finished.connect(lambda: self.sidebar_header.setVisible(True))
        animacion.finished.connect(self._refrescar_todas_las_pestanas)
        animacion.start()
        self._animacion_sidebar = animacion  # referencia viva -- si no, Python la destruye a mitad de camino

    def _section_label(self, texto):
        lbl = QLabel(texto)
        lbl.setObjectName("sectionLabel")
        return lbl

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {self.colors['border']};")
        line.setFixedHeight(1)
        return line

    def _on_uf_toggle(self):
        active = self.uf_toggle_btn.isChecked()
        self.uf_frame.setVisible(active)
        self.uf_toggle_btn.setText(f"Conversor UF/CLP: {'Activado' if active else 'Desactivado'}")
        if active:
            self._on_uf_mode_change()
            if self.df is not None:
                num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                self.uf_col_combo.clear()
                self.uf_col_combo.addItems(num_cols)

    def _on_uf_mode_change(self):
        if self.radio_auto.isChecked():
            self.uf_entry.setVisible(False)
            self.lbl_uf_status.setText("Consultando UF online...")
            self._start_uf_fetch()
        else:
            self.uf_entry.setVisible(True)
            self.lbl_uf_status.setText("")

    def _start_uf_fetch(self):
        self._uf_thread = QThread()
        self._uf_worker = UfWorker()
        self._uf_worker.moveToThread(self._uf_thread)
        self._uf_thread.started.connect(self._uf_worker.run)
        self._uf_worker.finished.connect(self._on_uf_fetched)
        self._uf_worker.finished.connect(self._uf_thread.quit)
        self._uf_worker.finished.connect(self._uf_worker.deleteLater)
        self._uf_thread.finished.connect(self._uf_thread.deleteLater)
        self._uf_thread.start()

    def _on_uf_fetched(self, val):
        if val is not None:
            self.uf_val = val
            self.lbl_uf_status.setText(f"UF hoy: ${val:,.2f}")
        else:
            self.lbl_uf_status.setText("Sin conexión. Usa Manual.")

    def apply_uf_conversion(self):
        if self.df is None:
            return
        if self.radio_manual.isChecked():
            try:
                self.uf_val = float(self.uf_entry.text())
            except ValueError:
                QMessageBox.critical(self, "Error", "Valor UF inválido.")
                return

        col = self.uf_col_combo.currentText()
        if not col or col not in self.df.columns:
            return

        if self.radio_clp_uf.isChecked():
            self.df[f"{col}_UF"] = (self.df[col] / self.uf_val).round(2)
        else:
            self.df[f"{col}_CLP"] = (self.df[col] * self.uf_val).round(0)

        self.refresh_all_column_lists()
        self.apply_table_filter()
        QMessageBox.information(self, "Éxito", f"Columna '{col}' convertida.")

    def _actualizar_logo_tema(self):
        """El logo es blanco sobre fondo transparente; aquí se pinta del
        color de texto del tema activo (blanco en Azul/Negro, casi negro
        en Blanco/Gris) para que siempre se vea."""
        if not hasattr(self, "lbl_logo_header"):
            return
        dpr = self.devicePixelRatioF()
        pix = QPixmap(LOGO_PNG_PATH).scaledToWidth(
            int(240 * dpr), Qt.TransformationMode.SmoothTransformation)
        tinte = QPixmap(pix.size())
        tinte.fill(Qt.GlobalColor.transparent)
        p = QPainter(tinte)
        p.drawPixmap(0, 0, pix)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(tinte.rect(), QColor(self.colors["text"]))
        p.end()
        tinte.setDevicePixelRatio(dpr)
        self.lbl_logo_header.setPixmap(tinte)

    def apply_theme(self, theme_name):
        self.theme_name = theme_name
        self.colors = THEMES[theme_name]
        self._actualizar_logo_tema()
        QApplication.instance().setStyleSheet(build_stylesheet(self.colors))
        for panel in self.chart_panels:
            panel.apply_theme(self.colors)
            if self.filtered_df is not None:
                panel.render(self.filtered_df)
        if hasattr(self, "panel_linea_tiempo"):
            self.panel_linea_tiempo.aplicar_tema(self.colors)
        if hasattr(self, "panel_control"):
            self.panel_control.aplicar_tema(self.colors)
        if hasattr(self, "panel_prediccion"):
            self.panel_prediccion.aplicar_tema(self.colors)
        if hasattr(self, "reporte_view"):
            self.reporte_view.setBackgroundBrush(QBrush(QColor(self.colors["bg"])))
        if hasattr(self, "panel_linaje"):
            self.panel_linaje.mostrar(self.panel_linaje._grafo, self.colors)
        if self.filtered_df is not None:
            self._actualizar_metricas()
            self._update_frecuencias()
            self._actualizar_alarmas()

    # ------------------------------------------------------------------
    # CARGA DE ARCHIVO
    # ------------------------------------------------------------------
    def _leer_tablas_de_archivo(self, path):
        """
        Lee UN archivo y devuelve una LISTA de tablas encontradas en él:
        [{'nombre','df','hojas_libro','hoja','motor'}, ...]
        - CSV/Parquet/tabla SQL única: siempre 1 tabla en la lista.
        - Excel de 1 sola hoja: 1 tabla.
        - Excel de varias hojas: si el usuario marca 1 sola hoja -> 1 tabla
          (con 'hojas_libro' con la lista completa, para que el panel de
          "cambiar/unir hoja" de la barra lateral siga funcionando igual
          que antes); si marca VARIAS hojas -> una tabla POR CADA hoja
          marcada, nombradas como la hoja (ej. "Clientes", "Pedidos"), con
          'hojas_libro' en None (esas ya no usan el panel de hojas: cada
          una quedó como una tabla relacionada aparte para el esquema).
        Devuelve None si el usuario canceló el diálogo de selección.
        """
        ext = os.path.splitext(path)[1].lower()
        nombre_base = os.path.splitext(os.path.basename(path))[0]
        hojas = _excel_sheet_names(path) if ext in (".xlsx", ".xls") else None

        if hojas and len(hojas) > 1:
            dialogo = _DialogoElegirHojas(hojas, os.path.basename(path), self)
            if dialogo.exec() != QDialog.DialogCode.Accepted:
                return None
            elegidas = dialogo.hojas_elegidas()

            if len(elegidas) == 1:
                hoja = elegidas[0]
                df = pd.read_excel(path, sheet_name=hoja)
                return [{"nombre": nombre_base, "df": df, "hojas_libro": hojas,
                         "hoja": hoja, "motor": "pandas"}]

            resultado = []
            for hoja in elegidas:
                df = pd.read_excel(path, sheet_name=hoja)
                resultado.append({"nombre": hoja, "df": df, "hojas_libro": None,
                                   "hoja": hoja, "motor": "pandas"})
            return resultado

        df, motor = load_data(path)
        hoja_unica = hojas[0] if hojas else None
        return [{"nombre": nombre_base, "df": df, "hojas_libro": hojas,
                  "hoja": hoja_unica, "motor": motor}]

    def _sincronizar_tabla_activa(self):
        """Cuando self.df cambia por una acción sobre la tabla activa
        (cambiar de hoja, unir otra hoja, o 'Calcular nueva columna'),
        refleja ese mismo cambio en self.tablas, para que el esquema
        (Ver esquema) siempre muestre la versión más reciente."""
        if self.nombre_tabla_activa is not None:
            self.tablas[self.nombre_tabla_activa] = self.df

    @staticmethod
    def _nombre_tabla_disponible(nombre_base, tablas_existentes):
        """Evita choques de nombre cuando 2 archivos se llaman parecido
        (ej. 'Pedidos.xlsx' y 'Pedidos_2024.xlsx' -> 'Pedidos', 'Pedidos_2024';
        si de verdad se repite el mismo nombre, usa 'Pedidos', 'Pedidos (2)')."""
        nombre = nombre_base
        i = 2
        while nombre in tablas_existentes:
            nombre = f"{nombre_base} ({i})"
            i += 1
        return nombre

    def on_load_file(self):
        # getOpenFileNames (con "s") deja elegir 1 o VARIOS archivos a la
        # vez: 1 archivo se comporta exactamente igual que antes; 2 o más
        # arman la ontología (esquema sugerido entre tablas).
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Cargar archivo(s) de datos", "",
            "Archivos de Datos (*.csv *.xlsx *.xls *.parquet *.sql *.sqlite *.db *.sqlite3)"
            ";;Todos los archivos (*.*)"
        )
        if not paths:
            return

        nuevas_tablas = {}      # {nombre_tabla: DataFrame}
        nuevas_fuentes = {}     # {nombre_tabla: {"tipo": "archivo", "ruta": ...}}
        nuevos_origenes = {}    # {nombre_tabla: datos del archivo/hoja} para la bitácora de procedencia
        info_principal = None   # (path, hojas, hoja_elegida, motor) del 1er archivo cargado OK
        errores = []

        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            try:
                tablas_del_archivo = self._leer_tablas_de_archivo(path)
            except SqlMultipleTablesError as e:
                dialogo = _DialogoElegirTablasSql(e.tables, os.path.basename(path), self)
                if dialogo.exec() != QDialog.DialogCode.Accepted:
                    continue
                elegidas = dialogo.tablas_elegidas()

                leer_tabla = read_sqlite_file if ext in (".sqlite", ".db", ".sqlite3") else read_sql_file
                tablas_del_archivo = []
                for nombre_tabla_sql in elegidas:
                    try:
                        df_sql = leer_tabla(path, table_name=nombre_tabla_sql)
                    except Exception as ex:
                        errores.append(f"{os.path.basename(path)} · {nombre_tabla_sql}: {ex}")
                        continue
                    # Si se marcó 1 sola tabla, se nombra como el archivo (igual que
                    # antes); si se marcaron varias, cada una se nombra como la
                    # tabla SQL, igual que Excel nombra cada hoja por separado.
                    nombre_entrada = (
                        os.path.splitext(os.path.basename(path))[0]
                        if len(elegidas) == 1 else nombre_tabla_sql
                    )
                    tablas_del_archivo.append({
                        "nombre": nombre_entrada, "df": df_sql,
                        "hojas_libro": None, "hoja": None, "motor": "pandas",
                        "tabla_sql": nombre_tabla_sql,
                    })
                if not tablas_del_archivo:
                    continue
            except Exception as e:
                errores.append(f"{os.path.basename(path)}: {e}")
                continue

            if tablas_del_archivo is None:   # el usuario canceló "elegir hoja(s)" para este archivo
                continue

            for entrada in tablas_del_archivo:
                nombre_tabla = self._nombre_tabla_disponible(entrada["nombre"], nuevas_tablas)
                nuevas_tablas[nombre_tabla] = entrada["df"]
                # Un .sql es un dump de texto ya congelado -- "actualizarlo"
                # no tiene el mismo sentido que un Excel/CSV que alguien más
                # puede seguir editando, así que no se ofrece el botón para
                # esas tablas (sí para csv/xlsx/xls/parquet).
                if ext != ".sql":
                    nuevas_fuentes[nombre_tabla] = {"tipo": "archivo", "ruta": path}
                # A diferencia de nuevas_fuentes, el origen se anota también para
                # un .sql: aunque no se pueda "actualizar", sí se sabe de dónde vino.
                nuevos_origenes[nombre_tabla] = {
                    "nombre_archivo": os.path.basename(path), "ruta": path,
                    "hoja": entrada.get("hoja"), "tabla_sql": entrada.get("tabla_sql"),
                }
                if info_principal is None:
                    info_principal = (path, entrada["hojas_libro"], entrada["hoja"], entrada["motor"])

        if not nuevas_tablas:
            if errores:
                QMessageBox.critical(self, "Error al leer archivo(s)", "\n".join(errores))
            return
        if errores:
            QMessageBox.warning(
                self, "Algunos archivos no se pudieron cargar",
                "Los demás sí se cargaron con normalidad.\n\n" + "\n".join(errores)
            )

        # El primer archivo cargado con éxito queda como "tabla activa": es
        # el que ve la pestaña Datos y el que sigue usando toda la
        # maquinaria que ya existía (hojas de Excel, motor, UF, etc.), sin
        # ningún cambio de comportamiento respecto de antes.
        self.tablas = nuevas_tablas
        self.fuentes_datos = nuevas_fuentes
        self.procedencia = BitacoraProcedencia()   # datos recién cargados: sin historia previa
        for nombre_tabla, o in nuevos_origenes.items():
            self.procedencia.registrar_origen(
                nombre_tabla, ORIGEN_ARCHIVO,
                filas=len(nuevas_tablas[nombre_tabla]), columnas=nuevas_tablas[nombre_tabla].shape[1],
                **o,
            )
        self.nombre_tabla_activa = next(iter(nuevas_tablas))
        self.df = nuevas_tablas[self.nombre_tabla_activa]
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        try:
            self.table_model.limpiar_todas_las_notas()
            self._quitar_marcas_limpieza()
            self.fingerprint_actual = self.memoria.registrar_carga(
                self.df, nombre_proceso=os.path.basename(info_principal[0])
            )

            path_principal, hojas, hoja_elegida, motor = info_principal
            self.excel_path = path_principal if hojas else None
            self.excel_sheet_names = hojas or []
            self.excel_hojas_activas = [hoja_elegida] if hoja_elegida else []
            self._actualizar_panel_hojas_excel()

            motor_label = "Polars" if motor == "polars" else "Pandas"
            sufijo_multi_tabla = (
                f"  (+{len(nuevas_tablas) - 1} tabla{'s' if len(nuevas_tablas) > 2 else ''} más)"
                if len(nuevas_tablas) > 1 else ""
            )
            self.lbl_archivo.setText(
                f"{os.path.basename(path_principal)}{sufijo_multi_tabla}\n"
                f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col. · Motor: {motor_label}"
            )
            self.refresh_all_column_lists()
            self.apply_table_filter()
            self.lbl_placeholder.setVisible(False)
            self._actualizar_boton_esquema()
            self._actualizar_boton_fuente()
        except Exception as e:
            # Antes, un error acá cortaba la función a medias en silencio:
            # "Ver diagrama"/el selector de tablas quedaban sin actualizar,
            # o Datos quedaba desincronizado, sin ningún aviso -- peor
            # todavía en un .exe compilado sin consola, donde el traceback
            # no se ve en ningún lado. Ahora se ve el error real, para
            # diagnosticar la causa de raíz en vez de adivinarla.
            QMessageBox.critical(
                self, "Error al terminar de cargar",
                "Los datos se leyeron, pero algo falló al terminar de prepararlos "
                f"para mostrarlos:\n\n{type(e).__name__}: {e}\n\n"
                "Prueba tocar un filtro de Datos para refrescar la vista; si el "
                "problema sigue, esto ayuda a encontrar la causa exacta."
            )
            import traceback
            traceback.print_exc()

    def _actualizar_panel_hojas_excel(self):
        """Muestra/oculta y refresca el panel de Hojas de Excel en la barra
        lateral, según si el archivo cargado tiene más de una hoja."""
        hojas = self.excel_sheet_names
        activas = self.excel_hojas_activas
        tiene_multiples_hojas = len(hojas) > 1
        self.excel_frame.setVisible(tiene_multiples_hojas)
        if not tiene_multiples_hojas:
            return

        self.combo_hoja_excel.blockSignals(True)
        self.combo_hoja_excel.clear()
        self.combo_hoja_excel.addItems(hojas)
        if activas:
            self.combo_hoja_excel.setCurrentText(activas[0])
        self.combo_hoja_excel.blockSignals(False)

        self.lbl_hojas_estado.setText("Usando: " + " + ".join(activas) if activas else "")

    def _on_cambiar_hoja_excel(self, nombre_hoja):
        """Cambia la hoja principal cargada (descarta cualquier unión previa)."""
        if not nombre_hoja or not self.excel_path:
            return
        if self.excel_hojas_activas == [nombre_hoja]:
            return
        try:
            df = pd.read_excel(self.excel_path, sheet_name=nombre_hoja)
        except Exception as e:
            QMessageBox.critical(self, "Error al leer hoja", str(e))
            return

        self.df = df
        self._sincronizar_tabla_activa()
        self.procedencia.olvidar_tabla(self.nombre_tabla_activa)   # otra hoja: lo anotado de la anterior ya no aplica
        self.procedencia.registrar_origen(
            self.nombre_tabla_activa, ORIGEN_ARCHIVO,
            filas=len(df), columnas=df.shape[1],
            nombre_archivo=os.path.basename(self.excel_path), ruta=self.excel_path, hoja=nombre_hoja,
        )
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        self.table_model.limpiar_todas_las_notas()
        self._quitar_marcas_limpieza()
        self.fingerprint_actual = self.memoria.registrar_carga(
            self.df, nombre_proceso=f"{os.path.basename(self.excel_path)} · {nombre_hoja}"
        )
        self.excel_hojas_activas = [nombre_hoja]
        self._actualizar_panel_hojas_excel()

        self.lbl_archivo.setText(
            f"{os.path.basename(self.excel_path)}\n"
            f"{df.shape[0]:,} filas, {df.shape[1]} col. · Motor: Pandas"
        )
        self.refresh_all_column_lists()
        self.apply_table_filter()

    def _on_unir_hoja_excel(self):
        """Une otra hoja del mismo archivo a los datos actuales, siempre que
        tenga exactamente las mismas columnas (sin importar el orden)."""
        if self.df is None or not self.excel_path:
            return
        disponibles = [h for h in self.excel_sheet_names if h not in self.excel_hojas_activas]
        if not disponibles:
            QMessageBox.information(self, "Sin más hojas", "Ya están todas las hojas del archivo unidas.")
            return

        elegida, ok = QInputDialog.getItem(
            self, "Unir hoja", "¿Qué hoja quieres unir a los datos actuales?", disponibles, 0, False
        )
        if not ok or not elegida:
            return
        try:
            nueva_df = pd.read_excel(self.excel_path, sheet_name=elegida)
        except Exception as e:
            QMessageBox.critical(self, "Error al leer hoja", str(e))
            return

        cols_actuales = set(self.df.columns)
        cols_nuevas = set(nueva_df.columns)
        if cols_actuales != cols_nuevas:
            solo_actuales = cols_actuales - cols_nuevas
            solo_nuevas = cols_nuevas - cols_actuales
            detalle = []
            if solo_actuales:
                detalle.append("Solo en los datos actuales: " + ", ".join(sorted(map(str, solo_actuales))))
            if solo_nuevas:
                detalle.append(f"Solo en \"{elegida}\": " + ", ".join(sorted(map(str, solo_nuevas))))
            QMessageBox.warning(
                self, "Columnas distintas",
                "No se pudo unir la hoja porque sus columnas no coinciden con las actuales.\n\n"
                + "\n".join(detalle)
            )
            return

        # Mismas columnas (aunque estén en otro orden): se unen las filas.
        nueva_df = nueva_df[list(self.df.columns)]
        self.df = pd.concat([self.df, nueva_df], ignore_index=True)
        self._sincronizar_tabla_activa()
        self.procedencia.registrar_hoja_unida(
            self.nombre_tabla_activa, elegida, filas=len(self.df), filas_agregadas=len(nueva_df),
        )
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        self.excel_hojas_activas = self.excel_hojas_activas + [elegida]
        self._actualizar_panel_hojas_excel()

        self.lbl_archivo.setText(
            f"{os.path.basename(self.excel_path)}\n"
            f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col. · Motor: Pandas"
        )
        self.refresh_all_column_lists()
        self.apply_table_filter()

    def abrir_transformacion_excel(self):
        """Abre el diálogo de transformación estilo Excel. La columna que
        resulte queda como una columna normal de self.df, así que aparece de
        inmediato en Gráficos, Métricas, Frecuencias, Narrativa y Reporte."""
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Sin datos", "Primero carga un archivo de datos.")
            return

        # Para avisar, dentro del diálogo, si se va a reemplazar una columna
        # de la que otras columnas calculadas salieron.
        dependientes = {}
        for col in self.df.columns:
            usan = self.procedencia.dependientes_de(self.nombre_tabla_activa, col)
            if usan:
                dependientes[col] = usan

        dialogo = DialogoTransformacionExcel(self.df, self, dependientes=dependientes)
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            df_resultado, nombre_columna = dialogo.get_resultado()
            if df_resultado is not None and nombre_columna:
                self.df = df_resultado
                self._sincronizar_tabla_activa()
                formula_usada = dialogo.get_formula_usada()
                if formula_usada:
                    # Se anota ANTES de refrescar las listas, para que el
                    # encabezado de la columna nueva ya salga con su marca "ƒx".
                    self.procedencia.registrar_columna_calculada(
                        self.nombre_tabla_activa, nombre_columna,
                        formula_texto=formula_usada[0], expresion=formula_usada[1],
                    )
                self.refresh_all_column_lists()
                self.apply_table_filter()

                QMessageBox.information(
                    self, "Éxito",
                    f"Columna '{nombre_columna}' creada y aplicada a {len(self.df):,} filas.\n\n"
                    + (f"Fórmula: {formula_usada[0]}\n\n" if formula_usada else "")
                    + "Pasa el mouse sobre su encabezado (lleva la marca ƒx) para recordar cómo "
                    "se calculó. También queda en Narrativa, en «Origen de los datos»."
                )

    def _actualizar_marcas_columnas_calculadas(self):
        """Marca con "ƒx" (y tooltip con la fórmula) los encabezados de Datos
        de las columnas que Hadar calculó en la tabla activa."""
        if not hasattr(self, "table_model"):
            return
        mapa = {}
        avisos = {}
        if self.df is not None and self.nombre_tabla_activa is not None:
            todos = self.procedencia.eventos_de_tabla(self.nombre_tabla_activa)
            for e in todos:
                if e.tipo == TIPO_COLUMNA_CALCULADA and e.columna in self.df.columns:
                    mapa[str(e.columna)] = e.detalle.get("formula_texto", "")
                    aviso = frase_desactualizacion(estado_de_actualizacion(todos, e.columna), larga=False)
                    if aviso:
                        avisos[str(e.columna)] = aviso
        self.table_model.set_columnas_calculadas(mapa, avisos)
        # El encabezado con "ƒx ⚠" es más largo que el nombre: se ensancha la columna
        # después de que la tabla termine de refrescarse (si no, se corta el texto).
        QTimer.singleShot(0, self._ensanchar_encabezados_marcados)

    def _ensanchar_encabezados_marcados(self):
        if self.df is None or not hasattr(self, "table_view"):
            return
        cabecera = self.table_view.horizontalHeader()
        columnas = [str(c) for c in self.table_model._df.columns]
        for nombre in self.table_model._columnas_calculadas:
            if nombre in columnas:
                i = columnas.index(nombre)
                cabecera.resizeSection(i, max(cabecera.sectionSize(i), cabecera.sectionSizeHint(i)))

    def refresh_all_column_lists(self):
        if self.df is None:
            return
        self._actualizar_marcas_columnas_calculadas()
        all_cols = self.df.columns.tolist()
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.datos_filter_col.blockSignals(True)
        current = self.datos_filter_col.currentText()
        self.datos_filter_col.clear()
        self.datos_filter_col.addItems(["Sin filtro"] + all_cols)
        self.datos_filter_col.setCurrentText(current if current in all_cols else "Sin filtro")
        self.datos_filter_col.blockSignals(False)

        for panel in self.chart_panels:
            panel.update_axis_choices(all_cols, num_cols)

        # Poda los bloques de Métricas cuya columna ya no exista en el
        # dataset actual (ej. se cargó un archivo distinto) -- si no se
        # quitaran, quedarían mostrando el último valor calculado como si
        # todavía fuera válido.
        if hasattr(self, "metricas_bloques"):
            for bloque in list(self.metricas_bloques):
                if bloque["columna"] not in num_cols:
                    self._quitar_bloque_metrica(bloque)

        self.freq_col_list.blockSignals(True)
        selected_before = [item.text() for item in self.freq_col_list.selectedItems()]
        self.freq_col_list.clear()
        for c in all_cols:
            item = QListWidgetItem(c)
            self.freq_col_list.addItem(item)
        if selected_before:
            for i in range(self.freq_col_list.count()):
                item = self.freq_col_list.item(i)
                if item.text() in selected_before:
                    item.setSelected(True)
        elif self.freq_col_list.count():
            self.freq_col_list.item(0).setSelected(True)
        self.freq_col_list.blockSignals(False)

        if hasattr(self, "panel_linea_tiempo"):
            self.panel_linea_tiempo.refrescar_datos()

    # ------------------------------------------------------------------
    # ÁREA PRINCIPAL
    # ------------------------------------------------------------------
    def _build_main_area(self, root_layout):
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(25, 20, 25, 20)

        # Barra de filtro interactivo
        self.filtro_bar = QFrame()
        self.filtro_bar.setObjectName("card")
        filtro_layout = QHBoxLayout(self.filtro_bar)
        self.lbl_filtro_activo = QLabel("")
        self.lbl_filtro_activo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        filtro_layout.addWidget(self.lbl_filtro_activo, stretch=1)
        filtro_layout.addStretch()
        self.btn_quitar_filtro = QPushButton("Quitar Filtro")
        self.btn_quitar_filtro.clicked.connect(self._quitar_filtro_grafico)
        filtro_layout.addWidget(self.btn_quitar_filtro)
        self.filtro_bar.setVisible(False)
        main_layout.addWidget(self.filtro_bar)

        self.lbl_placeholder = QLabel("Carga una BD desde la barra lateral para empezar.")
        self.lbl_placeholder.setObjectName("muted")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_placeholder)

        self.tabview = QTabWidget()
        main_layout.addWidget(self.tabview)
        # Misma lógica que el refresco tras animar la barra lateral: al
        # cambiar de pestaña, la que recién se muestra puede traer
        # geometría vieja en sus PlotWidget si algo cambió mientras estaba
        # oculta (ver _refrescar_todas_las_pestanas).
        self.tabview.currentChanged.connect(lambda _i: self._refrescar_todas_las_pestanas())

        tab_datos = QWidget()
        tab_graficos = QWidget()
        tab_metricas = QWidget()
        tab_indicadores = QWidget()
        tab_frecuencias = QWidget()
        tab_narrativa = QWidget()
        tab_reporte = QWidget()
        tab_alarma = QWidget()
        tab_linea_tiempo = QWidget()
        tab_panel = QWidget()
        tab_prediccion = QWidget()
        self.tabview.addTab(tab_datos, "Datos")
        self.tabview.addTab(tab_panel, "Panel")
        self.tabview.addTab(tab_narrativa, "Narrativa")
        self.tabview.addTab(tab_graficos, "Gráficos")
        self.tabview.addTab(tab_metricas, "Métricas")
        self.tabview.addTab(tab_indicadores, "Indicadores")
        self.tabview.addTab(tab_frecuencias, "Frecuencias")
        self.tabview.addTab(tab_alarma, "Alarma")
        self.tabview.addTab(tab_linea_tiempo, "Línea de Tiempo")
        self.tabview.addTab(tab_prediccion, "Predicción")
        self.tabview.addTab(tab_reporte, "Reporte")

        self._build_tab_datos(tab_datos)
        self._build_tab_graficos(tab_graficos)
        self._build_tab_metricas(tab_metricas)
        self._build_tab_indicadores(tab_indicadores)
        self._build_tab_frecuencias(tab_frecuencias)
        self._build_tab_narrativa(tab_narrativa)
        self._build_tab_reporte(tab_reporte)
        self._build_tab_alarma(tab_alarma)
        self._build_tab_linea_tiempo(tab_linea_tiempo)
        self._build_tab_panel(tab_panel)
        self._build_tab_prediccion(tab_prediccion)

        root_layout.addWidget(main)

    def _refrescar_barra_filtro(self):
        if self.filtro_grafico:
            self.filtro_bar.setVisible(True)
            self.lbl_filtro_activo.setText(
                f"Filtro Interactivo: {self.filtro_grafico['columna']} = {self.filtro_grafico['valor']}"
            )
        else:
            self.filtro_bar.setVisible(False)

    def _quitar_filtro_grafico(self):
        self.filtro_grafico = None
        self._refrescar_barra_filtro()
        self.apply_table_filter()

    # ------------------------------------------------------------------
    # TAB PANEL (ver panel.py)
    # ------------------------------------------------------------------
    def _build_tab_panel(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel_control = PanelControl(host=self)
        layout.addWidget(self.panel_control)

    def _notificar_panel(self):
        if hasattr(self, "panel_control"):
            self.panel_control.notificar_cambio()
        if hasattr(self, "panel_prediccion"):
            self.panel_prediccion.notificar_cambio()

    # ------------------------------------------------------------------
    # TAB PREDICCIÓN (ver prediccion.py y prediccion_ui.py)
    # ------------------------------------------------------------------
    def _build_tab_prediccion(self, tab):
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel_prediccion = PanelPrediccion(host=self)
        layout.addWidget(self.panel_prediccion)

    # ------------------------------------------------------------------
    # TAB LÍNEA DE TIEMPO
    # ------------------------------------------------------------------
    def _build_tab_linea_tiempo(self, tab):
        layout = QVBoxLayout(tab)
        self.panel_linea_tiempo = LineaTiempoPanel(host=self)
        layout.addWidget(self.panel_linea_tiempo)

    # ------------------------------------------------------------------
    # TAB DATOS
    # ------------------------------------------------------------------
    def _build_tab_datos(self, tab):
        layout = QVBoxLayout(tab)

        # Selector de tabla activa: solo se ve cuando hay 2+ tablas cargadas
        # (una base relacional). Con 1 sola tabla queda oculto y todo se ve
        # exactamente igual que siempre.
        fila_tabla = QHBoxLayout()
        fila_tabla.addWidget(QLabel("Tabla:"))
        self.combo_tabla_activa = QComboBox()
        self.combo_tabla_activa.setToolTip(
            "Elige qué tabla de tu base relacional quieres ver de cerca en "
            "esta pestaña (y en Gráficos, Métricas, Indicadores, Frecuencias)."
        )
        self.combo_tabla_activa.currentTextChanged.connect(self._on_cambiar_tabla_activa)
        fila_tabla.addWidget(self.combo_tabla_activa)
        fila_tabla.addStretch()
        self.fila_selector_tabla = QWidget()
        self.fila_selector_tabla.setLayout(fila_tabla)
        self.fila_selector_tabla.setVisible(False)
        layout.addWidget(self.fila_selector_tabla)

        top = QHBoxLayout()
        top.addWidget(QLabel("Filtrar por Columna:"))
        self.datos_filter_col = QComboBox()
        self.datos_filter_col.addItem("Sin filtro")
        self.datos_filter_col.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.datos_filter_col.setMinimumContentsLength(14)
        self.datos_filter_col.setMaximumWidth(180)
        self.datos_filter_col.currentTextChanged.connect(self._on_datos_filter_col_change)
        top.addWidget(self.datos_filter_col)

        self.datos_values_list = QListWidget()
        self.datos_values_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.datos_values_list.setMaximumHeight(70)
        self.datos_values_list.itemSelectionChanged.connect(self.apply_table_filter)
        top.addWidget(self.datos_values_list, stretch=1)
        layout.addLayout(top)

        # Fila aparte para Anomalía + los 3 botones: en la fila de arriba,
        # el único elemento que se puede achicar es la lista de valores
        # (datos_values_list) -- si aun achicándola al mínimo no alcanzaba
        # el espacio, estos botones quedaban recortados contra el borde de
        # la ventana sin ninguna forma de verlos ni hacer scroll (ver
        # "Editar Datos" cortado a media palabra). Con su propia fila,
        # nunca compiten por el mismo espacio.
        fila_acciones = QHBoxLayout()
        fila_acciones.addWidget(QLabel("Anomalía:"))
        self.datos_filter_anomalia = QComboBox()
        self.datos_filter_anomalia.addItem("Sin filtro")
        self.datos_filter_anomalia.setEnabled(False)
        self.datos_filter_anomalia.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.datos_filter_anomalia.setMinimumContentsLength(14)
        self.datos_filter_anomalia.setMaximumWidth(180)
        self.datos_filter_anomalia.setToolTip(
            "Filtra la tabla a solo las filas que Narrativa marcó como "
            "anomalía (opcionalmente de un tipo en particular). Se habilita "
            "después de presionar \"Generar Narrativa\"."
        )
        self.datos_filter_anomalia.currentTextChanged.connect(self._on_datos_filter_anomalia_change)
        fila_acciones.addWidget(self.datos_filter_anomalia)

        self.btn_editar_datos = QPushButton("Editar Datos: Desactivado")
        self.btn_editar_datos.setCheckable(True)
        self.btn_editar_datos.toggled.connect(self._toggle_edicion_datos)
        fila_acciones.addWidget(self.btn_editar_datos)

        self.btn_nota_celda = QPushButton("Agregar Nota")
        self.btn_nota_celda.setToolTip(
            "Selecciona una celda y escribe una nota. Las celdas que Narrativa "
            "marca como anomalía también quedan anotadas automáticamente aquí."
        )
        self.btn_nota_celda.clicked.connect(self._abrir_dialogo_nota)
        fila_acciones.addWidget(self.btn_nota_celda)

        self.btn_ver_esquema = QPushButton("Ver esquema")
        self.btn_ver_esquema.setToolTip(
            "Muestra cómo Hadar cree que se relacionan las tablas que cargaste "
            "(clientes, pedidos, productos, etc). Aparece solo cuando cargaste "
            "2 o más tablas a la vez desde \"Cargar Archivo\"."
        )
        self.btn_ver_esquema.clicked.connect(self._abrir_ventana_esquema)
        self.btn_ver_esquema.setVisible(False)
        fila_acciones.addWidget(self.btn_ver_esquema)
        fila_acciones.addStretch()
        layout.addLayout(fila_acciones)

        self.lbl_edicion_hint = QLabel(
            "Doble clic en una celda para editarla, o en el encabezado de una columna para renombrarla. "
            "Los cambios se aplican de inmediato a todas las pestañas."
        )
        self.lbl_edicion_hint.setObjectName("muted")
        self.lbl_edicion_hint.setVisible(False)
        layout.addWidget(self.lbl_edicion_hint)

        self.table_model = PandasTableModel()
        self.table_model.on_edit = self._on_table_cell_edited
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.horizontalHeader().setSectionsClickable(True)
        self.table_view.horizontalHeader().sectionDoubleClicked.connect(self._rename_column)
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.clicked.connect(self._mostrar_celda_en_status)
        layout.addWidget(self.table_view, stretch=1)

        bottom = QHBoxLayout()
        self.lbl_total_reg = QLabel("Total Registros: -")
        self.lbl_total_reg.setObjectName("muted")
        self.lbl_total_cols = QLabel("Total Columnas: -")
        self.lbl_total_cols.setObjectName("muted")
        bottom.addWidget(self.lbl_total_reg)
        bottom.addWidget(self.lbl_total_cols)
        bottom.addStretch()
        layout.addLayout(bottom)

    def _buscar_filas_especificas(self):
        """Filtra la tabla de Datos a exactamente las filas que el usuario
        escribió (por su número tal como se ve en la tabla), para comparar
        registros puntuales sin importar qué tan lejos estén entre sí."""
        texto = self.datos_buscar_filas.text().strip()
        if not texto:
            self._limpiar_busqueda_filas()
            return
        if self.df is None:
            return

        numeros, invalidos = [], []
        for token in texto.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                numeros.append(int(token))
            except ValueError:
                invalidos.append(token)
        if invalidos:
            QMessageBox.warning(
                self, "Números no reconocidos",
                f"No entendí esto como número de fila: {', '.join(invalidos)}."
            )

        total_filas = len(self.df)
        fuera_de_rango = [n for n in numeros if not (1 <= n <= total_filas)]
        if fuera_de_rango:
            QMessageBox.warning(
                self, "Fuera de rango",
                f"Esta tabla tiene {total_filas:,} fila(s) -- no existe la fila "
                f"{', '.join(map(str, fuera_de_rango))}."
            )

        posiciones = sorted({n - 1 for n in numeros if 1 <= n <= total_filas})
        if not posiciones:
            return

        self.filas_buscadas = posiciones
        self.datos_filter_col.setEnabled(False)
        self.datos_values_list.setEnabled(False)
        self.btn_limpiar_busqueda_filas.setVisible(True)
        self.apply_table_filter()

    def _limpiar_busqueda_filas(self):
        self.filas_buscadas = None
        self.datos_buscar_filas.clear()
        self.datos_filter_col.setEnabled(True)
        self.datos_values_list.setEnabled(True)
        self.btn_limpiar_busqueda_filas.setVisible(False)
        self.apply_table_filter()

    def _buscar_limpieza_sugerida(self):
        """Busca suciedad técnica (duplicados, espacios de más, formatos
        inconsistentes, caracteres especiales) en la tabla activa completa
        -- no en la vista filtrada, para no pasar por alto algo que un
        filtro esté ocultando. Marca las celdas afectadas y arma la lista
        navegable y los botones de 'Aplicar' por categoría. A propósito NO
        toca self.table_model.anomalias/notas: es un sistema aparte (ver
        comentario en table_model.py)."""
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        self.table_model.limpiar_marcas_de_limpieza()
        hallazgos = detectar_limpieza_sugerida(self.df)
        self._ultimos_hallazgos_limpieza = hallazgos

        self.limpieza_lista.clear()
        for h in hallazgos:
            if h.tipo == "duplicado":
                for fila in h.filas:
                    for col in self.df.columns:
                        self.table_model.marcar_limpieza(fila, col, h.descripcion, emitir=False)
            else:
                for fila in h.filas:
                    self.table_model.marcar_limpieza(fila, h.columna, h.descripcion, emitir=False)

            item = QListWidgetItem(h.descripcion)
            item.setData(Qt.UserRole, h.filas[0])
            item.setToolTip(h.descripcion)
            self.limpieza_lista.addItem(item)

        if self.table_model.rowCount() and self.table_model.columnCount():
            self.table_model.layoutChanged.emit()

        self._reconstruir_categorias_limpieza()

        hay_hallazgos = bool(hallazgos)
        self.limpieza_lista.setVisible(hay_hallazgos)
        self.btn_limpiar_marcas_limpieza.setVisible(hay_hallazgos)
        if hay_hallazgos:
            self.lbl_limpieza_estado.setText(f"{len(hallazgos)} hallazgo(s) marcado(s) en rojo en Datos.")
        else:
            self.lbl_limpieza_estado.setText("No se encontró suciedad técnica evidente en esta tabla.")

    def _reconstruir_categorias_limpieza(self):
        """Arma, de nuevo desde cero, un botón 'Aplicar' por cada
        categoría de suciedad encontrada -- se llama después de buscar y
        después de cada corrección aplicada (para que la categoría ya
        resuelta desaparezca de la lista)."""
        while self.limpieza_categorias_layout.count():
            item = self.limpieza_categorias_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        etiquetas = {
            "duplicado": "Duplicados exactos",
            "espacio_en_blanco": "Espacios de más",
            "formato_inconsistente": "Formatos inconsistentes",
            "caracter_especial": "Caracteres especiales",
            "numero_en_texto": "Números escritos como texto",
        }
        grupos = agrupar_por_tipo(getattr(self, "_ultimos_hallazgos_limpieza", []))
        for tipo, hallazgos_tipo in grupos.items():
            fila = QHBoxLayout()
            etiqueta = etiquetas.get(tipo, tipo)
            fila.addWidget(QLabel(f"{etiqueta} ({len(hallazgos_tipo)})"))
            btn = QPushButton("Aplicar")
            btn.clicked.connect(lambda _=False, t=tipo: self._aplicar_correccion_limpieza(t))
            fila.addWidget(btn)
            contenedor = QWidget()
            contenedor.setLayout(fila)
            self.limpieza_categorias_layout.addWidget(contenedor)

    def _aplicar_correccion_limpieza(self, tipo):
        """Aplica UNA categoría de corrección a self.df (nunca todas
        juntas). Para caracteres especiales, primero abre un diálogo de
        revisión carácter por carácter -- para las demás, una
        confirmación simple con el conteo alcanza, porque la corrección
        (recortar un espacio, unificar 'Chile'/'chile ') no tiene
        ambigüedad real como sí la tiene borrar un símbolo a ciegas."""
        hallazgos_tipo = [h for h in getattr(self, "_ultimos_hallazgos_limpieza", []) if h.tipo == tipo]
        if not hallazgos_tipo:
            return

        caracteres_quitados = None   # solo para "caracteres especiales" (cuáles eligió la persona)
        if tipo == "caracter_especial":
            dialogo = DialogoRevisarCaracteres(hallazgos_tipo, self)
            if dialogo.exec() != QDialog.DialogCode.Accepted or not dialogo.claves_elegidas:
                return
            df_nuevo = aplicar_quitar_caracteres(self.df, hallazgos_tipo, dialogo.claves_elegidas)
            caracteres_quitados = sorted({
                k[1] for k in dialogo.claves_elegidas if isinstance(k, tuple) and len(k) == 2
            })
        else:
            descripciones = {
                "duplicado": (
                    f"¿Eliminar los {len(hallazgos_tipo)} grupo(s) de filas duplicadas? "
                    f"Se conserva la primera fila de cada grupo."
                ),
                "espacio_en_blanco": f"¿Recortar espacios de más en {len(hallazgos_tipo)} celda(s)?",
                "formato_inconsistente": f"¿Unificar el formato en {len(hallazgos_tipo)} celda(s)?",
                "numero_en_texto": (
                    f"¿Convertir {len(hallazgos_tipo)} celda(s) con números escritos como texto "
                    f"(ej. 'cuatro', '6 unidades') a su valor numérico?"
                ),
            }
            respuesta = QMessageBox.question(
                self, "Confirmar limpieza",
                f"{descripciones.get(tipo, '¿Aplicar esta corrección?')}\n\n"
                f"Esto modifica los datos cargados en Hadar (no el archivo original "
                f"en tu computador).",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if respuesta != QMessageBox.StandardButton.Yes:
                return
            if tipo == "duplicado":
                df_nuevo = aplicar_eliminar_duplicados(self.df, hallazgos_tipo)
            elif tipo == "espacio_en_blanco":
                df_nuevo = aplicar_espacios_en_blanco(self.df, hallazgos_tipo)
            elif tipo == "formato_inconsistente":
                df_nuevo = aplicar_formato_inconsistente(self.df, hallazgos_tipo)
            elif tipo == "numero_en_texto":
                df_nuevo = aplicar_numeros_como_texto(self.df, hallazgos_tipo)
            else:
                return

        df_antes = self.df
        self.df = df_nuevo
        self.tablas[self.nombre_tabla_activa] = df_nuevo
        # Se anota lo que REALMENTE cambió (comparando antes y después), antes de
        # refrescar las listas para que los encabezados ya muestren los avisos.
        self.procedencia.registrar_limpieza(
            self.nombre_tabla_activa, tipo, df_antes, df_nuevo, caracteres=caracteres_quitados,
        )
        self.refresh_all_column_lists()
        self.apply_table_filter()
        # Se vuelve a calcular desde cero: lo que se acaba de corregir ya
        # no debería aparecer, y las marcas rojas tienen que recalcularse
        # porque las filas/posiciones pueden haber cambiado (ej. al borrar
        # duplicados).
        self._buscar_limpieza_sugerida()

    def _ir_a_hallazgo_limpieza(self, item):
        """Selecciona y desplaza la vista de Datos hasta la fila del
        hallazgo elegido en la lista. Si esa fila no está visible con el
        filtro actual de Datos, avisa en vez de fallar en silencio."""
        row_label = item.data(Qt.UserRole)
        try:
            posicion = self.table_model._df.index.get_loc(row_label)
        except KeyError:
            self.statusBar().showMessage(
                f"La fila {row_label} no está visible con el filtro actual de Datos.", 5000
            )
            return
        if isinstance(posicion, slice) or not isinstance(posicion, (int,)):
            return  # índice repetido/no único: no hay una única fila que seleccionar
        self.tabview.setCurrentIndex(0)  # pestaña Datos
        self.table_view.selectRow(posicion)
        self.table_view.scrollTo(self.table_model.index(posicion, 0))

    def _quitar_marcas_limpieza(self):
        self.table_model.limpiar_marcas_de_limpieza()
        self._ultimos_hallazgos_limpieza = []
        self.limpieza_lista.clear()
        self.limpieza_lista.setVisible(False)
        self.btn_limpiar_marcas_limpieza.setVisible(False)
        self.lbl_limpieza_estado.setText("")
        self._reconstruir_categorias_limpieza()

    def _mostrar_celda_en_status(self, index):
        """Muestra el contenido COMPLETO de la celda clickeada en la barra
        de estado -- útil cuando la columna es angosta y el texto se ve
        cortado. Complementa el tooltip al pasar el mouse (que desaparece
        apenas te movés): esto queda a la vista mientras seguís
        trabajando en otra parte de la pantalla."""
        if not index.isValid():
            return
        modelo = index.model()
        valor = modelo.data(index, Qt.DisplayRole)
        columna = modelo.headerData(index.column(), Qt.Horizontal, Qt.EditRole)   # nombre real, sin la marca ƒx
        fila = modelo.headerData(index.row(), Qt.Vertical)
        if valor in (None, ""):
            self.statusBar().clearMessage()
            return
        self.statusBar().showMessage(f"Fila {fila}, columna «{columna}»:  {valor}")

    def _actualizar_boton_esquema(self):
        hay_varias_tablas = len(self.tablas) > 1
        self.btn_ver_esquema.setVisible(hay_varias_tablas)
        self._actualizar_selector_tabla_activa()

    def _actualizar_selector_tabla_activa(self):
        hay_varias_tablas = len(self.tablas) > 1
        self.fila_selector_tabla.setVisible(hay_varias_tablas)
        if not hay_varias_tablas:
            return
        self.combo_tabla_activa.blockSignals(True)
        self.combo_tabla_activa.clear()
        self.combo_tabla_activa.addItems(list(self.tablas.keys()))
        if self.nombre_tabla_activa:
            self.combo_tabla_activa.setCurrentText(self.nombre_tabla_activa)
        self.combo_tabla_activa.blockSignals(False)

    def _on_cambiar_tabla_activa(self, nombre_tabla):
        """Cambia cuál tabla de la base relacional se ve como 'tabla activa'
        en Datos, Gráficos, Métricas, Indicadores y Frecuencias. No borra ni
        modifica ninguna tabla — solo cambia el 'lente' con el que se mira."""
        if not nombre_tabla or nombre_tabla not in self.tablas:
            return
        if nombre_tabla == self.nombre_tabla_activa:
            return

        self.nombre_tabla_activa = nombre_tabla
        self.df = self.tablas[nombre_tabla]
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        # OJO: las notas de celda (incluidas las automáticas de Narrativa) se
        # limpian al cambiar de tabla, igual que al cargar un archivo nuevo,
        # porque hoy se guardan sin distinguir de qué tabla son. Si esto
        # llega a molestar en el uso real, se puede guardar las notas por
        # tabla más adelante.
        self.table_model.limpiar_todas_las_notas()
        self._quitar_marcas_limpieza()

        # El panel de "Hoja de Excel" (cambiar/unir hoja) es propio de la
        # tabla que se cargó como archivo único con varias hojas; al mirar
        # otra tabla de la base relacional no aplica, así que se oculta.
        self.excel_path = None
        self.excel_sheet_names = []
        self.excel_hojas_activas = []
        self._actualizar_panel_hojas_excel()

        self.lbl_archivo.setText(
            f"{nombre_tabla}  (tabla {list(self.tablas.keys()).index(nombre_tabla) + 1} de {len(self.tablas)})\n"
            f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col."
        )
        self.refresh_all_column_lists()
        self.apply_table_filter()
        self._actualizar_boton_fuente()

    def _actualizar_boton_fuente(self):
        """Habilita '↻ Actualizar' solo si la tabla activa vino de un
        archivo o de una conexión SQL Server -- una tabla nueva sin
        guardar todavía, o de un .sql (dump estático), no tiene fuente."""
        info = self.fuentes_datos.get(self.nombre_tabla_activa)
        self.btn_actualizar_fuente.setEnabled(info is not None)
        if info is None:
            self.btn_actualizar_fuente.setToolTip(
                "Esta tabla no tiene una fuente para actualizar (no vino de "
                "un archivo ni de una conexión SQL Server)."
            )
        elif info["tipo"] == "archivo":
            self.btn_actualizar_fuente.setToolTip(f"Vuelve a leer: {info['ruta']}")
        else:
            self.btn_actualizar_fuente.setToolTip(
                f"Se vuelve a conectar a {info['servidor']} y trae la tabla "
                f"'{info['tabla']}' de nuevo (pide la contraseña otra vez)."
            )

    def _conectar_sql_server(self):
        """Conexión EN VIVO a un SQL Server -- distinto de 'Cargar Archivo
        → .sql', que solo lee un dump de texto sin conectarse a nada."""
        dialogo = _DialogoConexionSqlServer(self)
        if dialogo.exec() != QDialog.DialogCode.Accepted:
            return
        datos_conexion = dialogo.valores()
        if not datos_conexion["servidor"] or not datos_conexion["base_datos"]:
            QMessageBox.warning(self, "Faltan datos", "Servidor y base de datos son obligatorios.")
            return

        try:
            tablas_disponibles = listar_tablas_sql_server(
                datos_conexion["servidor"], datos_conexion["puerto"],
                datos_conexion["base_datos"], datos_conexion["usuario"],
                datos_conexion["password"],
            )
        except SqlServerNoDisponible as e:
            QMessageBox.critical(self, "No se pudo conectar", str(e))
            return

        if not tablas_disponibles:
            QMessageBox.information(self, "Sin tablas", "Esa base de datos no tiene tablas.")
            return

        selector = _DialogoElegirTablasSql(tablas_disponibles, datos_conexion["base_datos"], self)
        if selector.exec() != QDialog.DialogCode.Accepted:
            return
        elegidas = selector.tablas_elegidas()

        nuevas_tablas = {}
        nuevas_fuentes = {}
        errores = []
        for nombre_tabla_sql in elegidas:
            try:
                df = leer_tabla_sql_server(
                    datos_conexion["servidor"], datos_conexion["puerto"],
                    datos_conexion["base_datos"], datos_conexion["usuario"],
                    datos_conexion["password"], nombre_tabla_sql,
                )
            except SqlServerNoDisponible as e:
                errores.append(str(e))
                continue
            nombre_tabla = self._nombre_tabla_disponible(nombre_tabla_sql, nuevas_tablas)
            nuevas_tablas[nombre_tabla] = df
            # Se guarda todo menos la contraseña -- ver _DialogoConexionSqlServer.
            nuevas_fuentes[nombre_tabla] = {
                "tipo": "sql_server", "servidor": datos_conexion["servidor"],
                "puerto": datos_conexion["puerto"], "base_datos": datos_conexion["base_datos"],
                "usuario": datos_conexion["usuario"], "tabla": nombre_tabla_sql,
            }

        if not nuevas_tablas:
            QMessageBox.critical(self, "No se pudo traer ninguna tabla", "\n".join(errores))
            return
        if errores:
            QMessageBox.warning(self, "Algunas tablas no se pudieron traer", "\n".join(errores))

        self.tablas = nuevas_tablas
        self.fuentes_datos = nuevas_fuentes
        self.procedencia = BitacoraProcedencia()   # datos recién cargados: sin historia previa
        for nombre_tabla, f in nuevas_fuentes.items():
            self.procedencia.registrar_origen(
                nombre_tabla, ORIGEN_SQL_SERVER,
                filas=len(nuevas_tablas[nombre_tabla]), columnas=nuevas_tablas[nombre_tabla].shape[1],
                servidor=f["servidor"], base_datos=f["base_datos"], tabla_sql=f["tabla"],
            )
        self.nombre_tabla_activa = next(iter(nuevas_tablas))
        self.df = nuevas_tablas[self.nombre_tabla_activa]
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        self.table_model.limpiar_todas_las_notas()
        self._quitar_marcas_limpieza()
        self.excel_path = None
        self.excel_sheet_names = []
        self.excel_hojas_activas = []
        self._actualizar_panel_hojas_excel()

        sufijo_multi_tabla = (
            f"  (+{len(nuevas_tablas) - 1} tabla{'s' if len(nuevas_tablas) > 2 else ''} más)"
            if len(nuevas_tablas) > 1 else ""
        )
        self.lbl_archivo.setText(
            f"SQL Server: {datos_conexion['base_datos']}{sufijo_multi_tabla}\n"
            f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col."
        )
        self.refresh_all_column_lists()
        self.apply_table_filter()
        self.lbl_placeholder.setVisible(False)
        self._actualizar_boton_esquema()
        self._actualizar_boton_fuente()

    def _conectar_pos(self):
        """Conexión EN VIVO al POS en la red local, vía la API de
        sincronización (api_pos.py del lado del POS)."""
        dialogo = _DialogoConexionPos(self)
        if dialogo.exec() != QDialog.DialogCode.Accepted:
            return
        url_base = dialogo.valores()["url_base"]
        if not url_base:
            QMessageBox.warning(self, "Falta la dirección", "Ingresa la dirección del POS.")
            return
        if not API_KEY_POS_SINCRONIZACION:
            QMessageBox.critical(
                self, "Falta la API key",
                "No hay una API key configurada en este PC (variable de "
                "entorno ANALYTICS_API_KEY_POS). Configúrala antes de conectar."
            )
            return

        try:
            tablas_disponibles = listar_tablas_pos(url_base, API_KEY_POS_SINCRONIZACION)
        except PosNoDisponible as e:
            QMessageBox.critical(self, "No se pudo conectar", str(e))
            return

        if not tablas_disponibles:
            QMessageBox.information(self, "Sin tablas", "El POS no tiene tablas disponibles.")
            return

        selector = _DialogoElegirTablasSql(tablas_disponibles, "POS by Hadar", self)
        if selector.exec() != QDialog.DialogCode.Accepted:
            return
        elegidas = selector.tablas_elegidas()

        nuevas_tablas = {}
        nuevas_fuentes = {}
        errores = []
        for nombre_tabla_pos in elegidas:
            try:
                df = leer_tabla_pos(url_base, API_KEY_POS_SINCRONIZACION, nombre_tabla_pos)
            except PosNoDisponible as e:
                errores.append(str(e))
                continue
            nombre_tabla = self._nombre_tabla_disponible(nombre_tabla_pos, nuevas_tablas)
            nuevas_tablas[nombre_tabla] = df
            # Igual criterio que sql_server: se guarda de dónde vino, pero
            # nunca la API key (esa vive en config.py de este PC).
            nuevas_fuentes[nombre_tabla] = {
                "tipo": "api_pos", "url_base": url_base, "tabla": nombre_tabla_pos,
            }

        if not nuevas_tablas:
            QMessageBox.critical(self, "No se pudo traer ninguna tabla", "\n".join(errores))
            return
        if errores:
            QMessageBox.warning(self, "Algunas tablas no se pudieron traer", "\n".join(errores))

        self.tablas = nuevas_tablas
        self.fuentes_datos = nuevas_fuentes
        self.procedencia = BitacoraProcedencia()   # datos recién cargados: sin historia previa
        for nombre_tabla, f in nuevas_fuentes.items():
            self.procedencia.registrar_origen(
                nombre_tabla, ORIGEN_SQL_SERVER,  # ver nota en el mensaje de esta respuesta
                filas=len(nuevas_tablas[nombre_tabla]), columnas=nuevas_tablas[nombre_tabla].shape[1],
                servidor=f["url_base"], base_datos="POS by Hadar", tabla_sql=f["tabla"],
            )
        self.nombre_tabla_activa = next(iter(nuevas_tablas))
        self.df = nuevas_tablas[self.nombre_tabla_activa]
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        self.table_model.limpiar_todas_las_notas()
        self._quitar_marcas_limpieza()
        self.excel_path = None
        self.excel_sheet_names = []
        self.excel_hojas_activas = []
        self._actualizar_panel_hojas_excel()

        sufijo_multi_tabla = (
            f"  (+{len(nuevas_tablas) - 1} tabla{'s' if len(nuevas_tablas) > 2 else ''} más)"
            if len(nuevas_tablas) > 1 else ""
        )
        self.lbl_archivo.setText(
            f"POS by Hadar{sufijo_multi_tabla}\n"
            f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col."
        )
        self.refresh_all_column_lists()
        self.apply_table_filter()
        self.lbl_placeholder.setVisible(False)
        self._actualizar_boton_esquema()
        self._actualizar_boton_fuente()

    def _releer_archivo_de_origen(self, ruta):
        """Vuelve a leer el archivo de la tabla activa y devuelve (DataFrame,
        [hojas leídas]). Si es un Excel, relee la MISMA hoja (o las mismas hojas
        unidas) de la que salió la tabla, en vez de la primera hoja del archivo:
        así 'Actualizar' no trae datos de otra hoja sin avisar."""
        ext = os.path.splitext(ruta)[1].lower()
        if ext in (".xlsx", ".xls"):
            origen = evento_de_origen(self.procedencia.eventos_de_tabla(self.nombre_tabla_activa))
            hojas = []
            if origen is not None:
                if origen.detalle.get("hoja"):
                    hojas.append(origen.detalle["hoja"])
                hojas.extend(origen.detalle.get("hojas_unidas") or [])
            if not hojas:
                nombres = _excel_sheet_names(ruta)
                hojas = [nombres[0]] if nombres else []
            if hojas:
                partes = [pd.read_excel(ruta, sheet_name=h) for h in hojas]
                df = partes[0] if len(partes) == 1 else pd.concat(partes, ignore_index=True)
                return df, hojas
        df, _motor = load_data(ruta)
        return df, []

    def _actualizar_desde_la_fuente(self):
        """Vuelve a traer los datos de la tabla ACTIVA desde donde vinieron
        -- un archivo (Excel/CSV/Parquet compartido que alguien más pudo
        haber editado) o una conexión SQL Server en vivo."""
        info = self.fuentes_datos.get(self.nombre_tabla_activa)
        if info is None:
            return

        hojas_releidas = []
        if info["tipo"] == "archivo":
            try:
                df_nuevo, hojas_releidas = self._releer_archivo_de_origen(info["ruta"])
            except Exception as e:
                QMessageBox.critical(self, "No se pudo actualizar", str(e))
                return
        elif info["tipo"] == "api_pos":
            if not API_KEY_POS_SINCRONIZACION:
                QMessageBox.critical(
                    self, "Falta la API key",
                    "No hay una API key configurada en este PC (variable de "
                    "entorno ANALYTICS_API_KEY_POS)."
                )
                return
            try:
                df_nuevo = leer_tabla_pos(
                    info["url_base"], API_KEY_POS_SINCRONIZACION, info["tabla"],
                )
            except PosNoDisponible as e:
                QMessageBox.critical(self, "No se pudo actualizar", str(e))
                return
        else:
            password, ok = QInputDialog.getText(
                self, "Contraseña de SQL Server",
                f"Usuario: {info['usuario']} @ {info['servidor']}\nContraseña:",
                QLineEdit.EchoMode.Password,
            )
            if not ok:
                return
            try:
                df_nuevo = leer_tabla_sql_server(
                    info["servidor"], info["puerto"], info["base_datos"],
                    info["usuario"], password, info["tabla"],
                )
            except SqlServerNoDisponible as e:
                QMessageBox.critical(self, "No se pudo actualizar", str(e))
                return

        self.tablas[self.nombre_tabla_activa] = df_nuevo
        self.df = df_nuevo
        # Se volvió a leer desde la MISMA fuente: se conserva de dónde vino (y cuándo
        # se cargó por primera vez), pero lo calculado antes ya no aplica.
        self.procedencia.olvidar_tabla(self.nombre_tabla_activa, conservar=(TIPO_ARCHIVO_ORIGEN,))
        if not self.procedencia.registrar_actualizacion(
            self.nombre_tabla_activa, filas=len(df_nuevo), columnas=df_nuevo.shape[1],
            hoja=hojas_releidas[0] if hojas_releidas else None,
        ):
            # Sin origen anotado (ej. proyecto antiguo sin fuente completa): se anota
            # ahora lo que sí se sabe, sin inventar cuándo fue la primera carga.
            if info["tipo"] == "archivo":
                self.procedencia.registrar_origen(
                    self.nombre_tabla_activa, ORIGEN_ARCHIVO, es_actualizacion=True,
                    filas=len(df_nuevo), columnas=df_nuevo.shape[1],
                    nombre_archivo=os.path.basename(info["ruta"]), ruta=info["ruta"],
                    hoja=hojas_releidas[0] if hojas_releidas else None,
                )
            elif info["tipo"] == "api_pos":
                self.procedencia.registrar_origen(
                    self.nombre_tabla_activa, ORIGEN_SQL_SERVER, es_actualizacion=True,
                    filas=len(df_nuevo), columnas=df_nuevo.shape[1],
                    servidor=info["url_base"], base_datos="POS by Hadar", tabla_sql=info["tabla"],
                )
            else:
                self.procedencia.registrar_origen(
                    self.nombre_tabla_activa, ORIGEN_SQL_SERVER, es_actualizacion=True,
                    filas=len(df_nuevo), columnas=df_nuevo.shape[1],
                    servidor=info["servidor"], base_datos=info["base_datos"], tabla_sql=info["tabla"],
                )
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        self.refresh_all_column_lists()
        self.apply_table_filter()
        self.lbl_archivo.setText(
            f"{self.nombre_tabla_activa}  (actualizado)\n"
            f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col."
        )
        QMessageBox.information(
            self, "Actualizado",
            f"'{self.nombre_tabla_activa}' se actualizó: {self.df.shape[0]:,} filas."
        )

    def _abrir_ventana_esquema(self):
        if len(self.tablas) < 2:
            QMessageBox.information(
                self, "Sin tablas para relacionar",
                "Carga 2 o más tablas al mismo tiempo (selecciona varios archivos "
                "en \"Cargar Archivo\") para ver el esquema sugerido entre ellas."
            )
            return
        dialogo = DialogoEsquemaOntologia(self.tablas, self.colors, self,
                                           relaciones_iniciales=self.relaciones_ontologia)
        dialogo.exec()
        self.relaciones_ontologia = dialogo.obtener_relaciones_confirmadas()
        # Los datos no cambiaron, pero el esquema sí -- si ya había una
        # narrativa generada, el capítulo de cruces quedaría desactualizado.
        self._marcar_narrativa_desactualizada()

    def _guardar_proyecto(self):
        """Devuelve True si el proyecto quedó efectivamente guardado en
        disco, False si el usuario canceló el diálogo o falló el guardado
        -- closeEvent() usa este valor para decidir si es seguro cerrar."""
        if not self.tablas:
            QMessageBox.information(self, "Sin datos", "Carga datos primero para poder guardar un proyecto.")
            return False
        sugerencia = os.path.join(
            ruta_carpeta_proyectos(), f"{self.nombre_tabla_activa or 'proyecto'}{EXTENSION}"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar proyecto", sugerencia, f"Proyecto Hadar (*{EXTENSION})"
        )
        if not path:
            return False
        if not path.lower().endswith(EXTENSION):
            path += EXTENSION

        filtro_columna = self.datos_filter_col.currentText()
        filtro_valores = [item.text() for item in self.datos_values_list.selectedItems()]

        if self.ml_activado is None:
            # Se pregunta una sola vez por proyecto -- de ahí en adelante
            # queda fijo, no se vuelve a preguntar en guardados siguientes.
            respuesta = QMessageBox.question(
                self, "Aprendizaje continuo",
                "¿Quieres activar el aprendizaje continuo para este proyecto?\n\n"
                "Con cada guardado, Hadar va aprendiendo el rango normal de "
                "TUS propios datos en vez de usar un umbral genérico -- pero "
                "solo tiene sentido si vas a seguir actualizando este mismo "
                "proyecto en el tiempo, no para análisis puntuales.",
                QMessageBox.Yes | QMessageBox.No,
            )
            self.ml_activado = (respuesta == QMessageBox.Yes)

        if self.ml_activado:
            # Foto de los datos ACTUALES, justo antes de guardar -- así
            # la línea base queda al día con lo que se está guardando,
            # no con lo que había la última vez que se abrió el proyecto.
            self.linea_base_ml = actualizar_linea_base(self.linea_base_ml, self.tablas)

        self.procedencia.podar(self.tablas)   # nada de columnas que ya no existen
        try:
            guardar_proyecto(
                path,
                tablas=self.tablas,
                nombre_tabla_activa=self.nombre_tabla_activa,
                relaciones_ontologia=self.relaciones_ontologia,
                filtro_columna=filtro_columna,
                filtro_valores=filtro_valores,
                notas_manuales=self.table_model.notas_manuales(),
                indicadores=self.indicadores,
                ml_activado=self.ml_activado,
                linea_base_ml=self.linea_base_ml,
                fuentes_datos=self.fuentes_datos,
                ml_multivariado_activado=self.ml_multivariado_activado,
                procedencia=self.procedencia.a_lista(),
                prediccion=(self.panel_prediccion.config_a_dict()
                            if hasattr(self, "panel_prediccion") else self.prediccion_config),
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar el proyecto", str(e))
            return False
        QMessageBox.information(self, "Proyecto guardado", f"Guardado en:\n{path}")
        self._ultimo_guardado_firma = self._firma_estado_actual()
        return True

    def _abrir_proyecto(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir proyecto", ruta_carpeta_proyectos(), f"Proyecto Hadar (*{EXTENSION})"
        )
        if not path:
            return
        self.abrir_proyecto_desde_ruta(path)

    def abrir_proyecto_desde_ruta(self, path):
        """La lógica real de 'continuar un proyecto', separada del botón de
        la barra lateral para que la Pantalla de Inicio también pueda
        llamarla directo con una ruta ya elegida (sin mostrar otro
        diálogo de archivo encima)."""
        try:
            datos_proyecto = abrir_proyecto(path)
        except Exception as e:
            QMessageBox.critical(self, "Error al abrir el proyecto", str(e))
            return
        if not datos_proyecto.tablas:
            QMessageBox.critical(self, "Proyecto vacío", "Este archivo no tiene ninguna tabla guardada.")
            return

        self.tablas = datos_proyecto.tablas
        self.nombre_tabla_activa = (
            datos_proyecto.nombre_tabla_activa if datos_proyecto.nombre_tabla_activa in self.tablas
            else next(iter(self.tablas))
        )
        self.df = self.tablas[self.nombre_tabla_activa]
        self.filtro_grafico = None
        self._resetear_filtro_anomalias()
        self.relaciones_ontologia = datos_proyecto.relaciones_ontologia
        self.ml_activado = datos_proyecto.ml_activado
        self.linea_base_ml = datos_proyecto.linea_base_ml
        self.fuentes_datos = datos_proyecto.fuentes_datos
        self.procedencia = BitacoraProcedencia.desde_lista(datos_proyecto.procedencia)
        # Proyectos guardados antes de que existiera el registro de origen: se
        # completa con lo que la fuente guardada sí dice (archivo/base), sin fecha.
        self.procedencia.completar_origen_desde_fuentes(self.tablas, self.fuentes_datos)
        self.ml_multivariado_activado = datos_proyecto.ml_multivariado_activado
        if hasattr(self, "btn_ml_multivariado"):
            # blockSignals: evita que restaurar el estado dispare de nuevo el
            # diálogo de confirmación y el auto-test de _alternar_ml_multivariado
            # -- acá solo se restaura lo que el proyecto ya tenía guardado.
            self.btn_ml_multivariado.blockSignals(True)
            self.btn_ml_multivariado.setChecked(self.ml_multivariado_activado)
            self.btn_ml_multivariado.blockSignals(False)
            self.btn_ml_multivariado.setText(
                "Detección por combinación (ML): Activado" if self.ml_multivariado_activado
                else "Detección por combinación (ML): Desactivado"
            )
        if self.ml_activado:
            # Foto de los datos tal como llegan al abrir -- útil sobre
            # todo cuando el proyecto se actualizó afuera (ej. una BD que
            # cambió) entre una sesión y la siguiente.
            self.linea_base_ml = actualizar_linea_base(self.linea_base_ml, self.tablas)

        # No hay un archivo Excel real detrás de un proyecto reabierto,
        # así que el panel de "Hoja de Excel" no aplica acá.
        self.excel_path = None
        self.excel_sheet_names = []
        self.excel_hojas_activas = []
        self._actualizar_panel_hojas_excel()

        self.table_model.limpiar_todas_las_notas()
        self._quitar_marcas_limpieza()
        self.table_model.cargar_notas_manuales(datos_proyecto.notas_manuales)

        self.indicadores = [Indicador.from_dict(d) for d in datos_proyecto.indicadores_dict]
        self.prediccion_config = datos_proyecto.prediccion
        if hasattr(self, "panel_prediccion"):
            self.panel_prediccion.cargar_config(self.prediccion_config)

        nombre_proyecto = os.path.splitext(os.path.basename(path))[0]
        sufijo_multi_tabla = (
            f"  (+{len(self.tablas) - 1} tabla{'s' if len(self.tablas) > 2 else ''} más)"
            if len(self.tablas) > 1 else ""
        )
        self.lbl_archivo.setText(
            f"{nombre_proyecto}{sufijo_multi_tabla}\n"
            f"{self.df.shape[0]:,} filas, {self.df.shape[1]} col. · Proyecto abierto"
        )
        self.refresh_all_column_lists()

        # Reponer el filtro de Datos exactamente como quedó guardado.
        if datos_proyecto.filtro_columna and datos_proyecto.filtro_columna != "Sin filtro":
            self.datos_filter_col.blockSignals(True)
            self.datos_filter_col.setCurrentText(datos_proyecto.filtro_columna)
            self.datos_filter_col.blockSignals(False)
            self.datos_values_list.clear()
            if datos_proyecto.filtro_columna in self.df.columns:
                columna = self.df[datos_proyecto.filtro_columna]
                vals = sorted(columna.dropna().unique().tolist(), key=str)
                for v in vals:
                    self.datos_values_list.addItem(QListWidgetItem(str(v)))
                if columna.isna().any():
                    self.datos_values_list.addItem(QListWidgetItem(VALOR_VACIO_FILTRO))
                self.datos_values_list.blockSignals(True)
                valores_guardados = {str(v) for v in datos_proyecto.filtro_valores}
                for i in range(self.datos_values_list.count()):
                    item = self.datos_values_list.item(i)
                    item.setSelected(item.text() in valores_guardados)
                self.datos_values_list.blockSignals(False)

        self.apply_table_filter()
        self._actualizar_indicadores()
        self.lbl_placeholder.setVisible(False)
        self._actualizar_boton_esquema()
        self._actualizar_boton_fuente()

        self._ultimo_guardado_firma = self._firma_estado_actual()
        QMessageBox.information(self, "Proyecto abierto", f"Continuando: {nombre_proyecto}")

    def _toggle_edicion_datos(self, activo: bool):
        self.table_model.editable = activo
        self.btn_editar_datos.setText(
            f"Editar Datos: {'Activado' if activo else 'Desactivado'}"
        )
        self.lbl_edicion_hint.setVisible(activo)
        self.table_view.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
            if activo else QAbstractItemView.NoEditTriggers
        )

    def _on_table_cell_edited(self, row_label, col_name, nuevo_valor, valor_anterior=None):
        """Propaga una edición hecha en la tabla (que puede estar mostrando
        self.filtered_df) de vuelta al DataFrame maestro self.df, y refresca
        el resto de la app (stats, indicadores, frecuencias)."""
        if self.df is None or row_label not in self.df.index:
            return
        self.df.at[row_label, col_name] = nuevo_valor
        self._cache_fechas = None   # por si la celda editada era una fecha
        # Solo cuenta como cambio a mano si el valor de verdad cambió (Qt llama a
        # setData también cuando se acepta la celda sin tocar nada).
        if not self._valores_iguales(valor_anterior, nuevo_valor):
            self.procedencia.registrar_edicion_manual(self.nombre_tabla_activa, str(col_name))
            self._actualizar_marcas_columnas_calculadas()
        self.apply_table_filter()

    @staticmethod
    def _valores_iguales(a, b):
        try:
            if a is None and b is None:
                return True
            if a != a and b != b:   # ambos NaN
                return True
            return bool(a == b)
        except Exception:
            return str(a) == str(b)

    def _abrir_dialogo_nota(self):
        seleccion = self.table_view.selectionModel().currentIndex() if self.table_view.selectionModel() else None
        if seleccion is None or not seleccion.isValid():
            QMessageBox.information(
                self, "Selecciona una celda",
                "Haz clic en una celda de la tabla y luego presiona \"Agregar Nota\"."
            )
            return

        row, col = seleccion.row(), seleccion.column()
        df_mostrado = self.table_model._df
        try:
            row_label = df_mostrado.index[row]
            col_name = df_mostrado.columns[col]
        except IndexError:
            return
        valor = df_mostrado.iat[row, col]
        valor_texto = "" if pd.isna(valor) else str(valor)

        clave = (row_label, col_name)
        nota_existente = self.table_model.notas.get(clave, "")
        era_anomalia = clave in self.table_model.anomalias

        dialogo = DialogoNota(row, col_name, valor_texto, nota_existente, era_anomalia, parent=self)
        if dialogo.exec() != QDialog.DialogCode.Accepted:
            return

        texto = dialogo.get_nota()
        if not texto:
            self.table_model.quitar_nota(row_label, col_name)
            return
        self.table_model.agregar_nota(
            row_label, col_name, texto, es_anomalia=dialogo.es_anomalia(), origen_narrativa=False
        )

    def _rename_column(self, section):
        """Renombra una columna al hacer doble clic en su encabezado. Solo
        disponible con 'Editar Datos' activado, igual que la edición de celdas."""
        if not self.table_model.editable:
            QMessageBox.information(
                self, "Edición desactivada",
                "Activa \"Editar Datos\" para poder renombrar columnas."
            )
            return
        if self.df is None or section < 0 or section >= len(self.table_model._df.columns):
            return

        nombre_actual = str(self.table_model._df.columns[section])
        nuevo_nombre, ok = QInputDialog.getText(
            self, "Renombrar columna", "Nuevo nombre de columna:",
            QLineEdit.Normal, nombre_actual
        )
        if not ok:
            return
        nuevo_nombre = nuevo_nombre.strip()
        if not nuevo_nombre or nuevo_nombre == nombre_actual:
            return
        if nuevo_nombre in self.df.columns:
            QMessageBox.warning(self, "Nombre repetido", "Ya existe una columna con ese nombre.")
            return

        # Recordar qué widgets apuntaban a la columna vieja para restaurar
        # la selección/valor después de refrescar todas las listas.
        filtro_col_antes = self.datos_filter_col.currentText()
        panels_antes = [(p, p.combo_x.currentText(), p.combo_y.currentText()) for p in self.chart_panels]
        seleccion_frecuencias = [item.text() for item in self.freq_col_list.selectedItems()]

        self.df.rename(columns={nombre_actual: nuevo_nombre}, inplace=True)
        self._cache_fechas = None
        self.procedencia.renombrar_columna(self.nombre_tabla_activa, nombre_actual, nuevo_nombre)

        if self.filtro_grafico and self.filtro_grafico.get("columna") == nombre_actual:
            self.filtro_grafico["columna"] = nuevo_nombre

        for ind in getattr(self, "indicadores", []):
            if ind.columna == nombre_actual:
                ind.columna = nuevo_nombre
            if ind.formula:
                ind.formula = ind.formula.replace(f"'{nombre_actual}'", f"'{nuevo_nombre}'")
                ind.formula = ind.formula.replace(f'"{nombre_actual}"', f'"{nuevo_nombre}"')

        for bloque in getattr(self, "metricas_bloques", []):
            if bloque["columna"] == nombre_actual:
                bloque["columna"] = nuevo_nombre
                bloque["lbl_titulo"].setText(f"<b>Métricas: {nuevo_nombre}</b>")

        self.refresh_all_column_lists()

        if filtro_col_antes == nombre_actual:
            self.datos_filter_col.setCurrentText(nuevo_nombre)
        for panel, x_antes, y_antes in panels_antes:
            if x_antes == nombre_actual:
                panel.combo_x.setCurrentText(nuevo_nombre)
            if y_antes == nombre_actual:
                panel.combo_y.setCurrentText(nuevo_nombre)
        if nombre_actual in seleccion_frecuencias:
            objetivo = [nuevo_nombre if c == nombre_actual else c for c in seleccion_frecuencias]
            for i in range(self.freq_col_list.count()):
                item = self.freq_col_list.item(i)
                item.setSelected(item.text() in objetivo)

        self.apply_table_filter()

    def _on_datos_filter_col_change(self, value):
        self.datos_values_list.clear()
        if value and value != "Sin filtro" and self.df is not None:
            columna = self.df[value]
            vals = sorted(columna.dropna().unique().tolist(), key=str)
            for v in vals:
                item = QListWidgetItem(str(v))
                self.datos_values_list.addItem(item)
            if columna.isna().any():
                self.datos_values_list.addItem(QListWidgetItem(VALOR_VACIO_FILTRO))
            self.datos_values_list.selectAll()
        self.apply_table_filter()

    def _on_datos_filter_anomalia_change(self, texto):
        """El combo 'Anomalía' del filtro de Datos: traduce la etiqueta
        legible elegida de vuelta a la clave interna que usa
        self._ultimas_anomalias, y refiltra la tabla."""
        if not texto or texto == "Sin filtro":
            self.filtro_tipo_anomalia = None
        elif texto == "Cualquier anomalía":
            self.filtro_tipo_anomalia = "todas"
        else:
            self.filtro_tipo_anomalia = _ETIQUETAS_TIPO_ANOMALIA_INVERSA.get(texto, texto)
        self.apply_table_filter()

    def _repoblar_filtro_anomalias(self):
        """Repuebla el combo 'Anomalía' de Datos con los tipos que
        aparecen en self._ultimas_anomalias (calculado por Narrativa la
        última vez que se generó el informe). Sin anomalías detectadas
        (nunca se generó Narrativa, o el dataset no tiene ninguna), el
        combo queda deshabilitado en 'Sin filtro' -- igual que estaba
        antes de que existiera este filtro."""
        if not hasattr(self, "datos_filter_anomalia"):
            return
        tipos_presentes = sorted({
            a.get("tipo") for a in getattr(self, "_ultimas_anomalias", []) if a.get("tipo")
        })

        self.datos_filter_anomalia.blockSignals(True)
        self.datos_filter_anomalia.clear()
        self.datos_filter_anomalia.addItem("Sin filtro")
        if tipos_presentes:
            self.datos_filter_anomalia.addItem("Cualquier anomalía")
            for tipo in tipos_presentes:
                self.datos_filter_anomalia.addItem(ETIQUETAS_TIPO_ANOMALIA.get(tipo, tipo))
        self.datos_filter_anomalia.setEnabled(bool(tipos_presentes))
        self.datos_filter_anomalia.setCurrentText("Sin filtro")
        self.datos_filter_anomalia.blockSignals(False)
        self.filtro_tipo_anomalia = None

    def _indices_filtro_anomalia(self):
        """Índices reales (los mismos que usa self.df.index) de las filas
        que cumplen el tipo de anomalía elegido en el filtro de Datos.
        Se apoya en self._ultimas_anomalias -- si nunca se generó Narrativa,
        o el tipo elegido ya no aparece, devuelve un conjunto vacío (el
        filtro simplemente no deja pasar ninguna fila, en vez de fallar)."""
        tipo = self.filtro_tipo_anomalia
        indices = set()
        for a in getattr(self, "_ultimas_anomalias", []):
            if tipo != "todas" and a.get("tipo") != tipo:
                continue
            idxs = a.get("indices_atipicos")
            if idxs is None:
                idx_unico = a.get("fila_indice")
                idxs = [idx_unico] if idx_unico is not None else []
            indices.update(idxs)
        return indices

    def _resetear_filtro_anomalias(self):
        """Se llama junto con 'self.filtro_grafico = None' en cada punto
        donde se carga un dataset nuevo o se cambia de tabla activa: los
        índices que guarda self._ultimas_anomalias son de OTRO dataset, así
        que seguir mostrando ese filtro llevaría a resultados sin sentido
        (o directamente vacíos). Se limpia en vez de arrastrarlo. Por la
        misma razón, la sub-pestaña Linaje (que también depende de la
        última Narrativa generada) se desactiva hasta que se vuelva a
        generar sobre los datos nuevos."""
        self._ultimas_anomalias = []
        self._anomalias_por_tabla = {}
        self._linaje_disponible = False
        self._repoblar_filtro_anomalias()
        self._actualizar_resumen_anomalias_frecuencias()
        self._actualizar_estado_linaje()

    def _construir_df_filtrado_base(self):
        """DataFrame filtrado por TODO lo de la pestaña Datos (buscador de
        filas puntuales, filtro por columna, filtro por clic en un gráfico)
        EXCEPTO el rango de la Línea de Tiempo. Lo usa apply_table_filter()
        como punto de partida antes de sumar el rango, y lo usa también
        LineaTiempoPanel para dibujar su propio histograma -- si el
        histograma leyera self.filtered_df (que YA incluye el rango), cada
        arrastre del slider se estaría mordiendo la cola: la vista se
        achicaría sobre sí misma en vez de mostrar siempre el total
        disponible para recortar."""
        if self.df is None:
            return None
        # Sin .copy(): los filtros de abajo nunca modifican `df`, solo arman
        # máscaras y al final se recorta UNA vez (df[mascara] ya devuelve una
        # tabla nueva). Antes se copiaba la tabla entera en cada cambio de
        # filtro, incluso sin ningún filtro activo.
        df = self.df

        if self.filas_buscadas:
            # Modo "comparar filas puntuales": a propósito se ignoran los
            # demás filtros de Datos mientras este esté activo (ver tooltip
            # del buscador) -- así nunca hay dos criterios mezclados sin que
            # el usuario lo sepa.
            posiciones_validas = [p for p in self.filas_buscadas if p < len(df)]
            return df.iloc[posiciones_validas]

        mascara = None   # np.ndarray de bool alineado con las filas de `df`

        col = self.datos_filter_col.currentText()
        if col and col != "Sin filtro":
            selected_vals = [item.text() for item in self.datos_values_list.selectedItems()]
            if selected_vals:
                incluir_vacios = VALOR_VACIO_FILTRO in selected_vals
                valores_normales = [v for v in selected_vals if v != VALOR_VACIO_FILTRO]
                m = df[col].astype(str).isin(valores_normales)
                if incluir_vacios:
                    m = m | df[col].isna()
                mascara = m.to_numpy()

        if self.filtro_grafico:
            col_f = self.filtro_grafico["columna"]
            val_f = self.filtro_grafico["valor"]
            if col_f in df.columns:
                m = (df[col_f].astype(str) == str(val_f)).to_numpy()
                mascara = m if mascara is None else (mascara & m)

        if self.filtro_tipo_anomalia:
            indices_anomalos = self._indices_filtro_anomalia()
            m = np.asarray(df.index.isin(indices_anomalos))
            mascara = m if mascara is None else (mascara & m)

        if mascara is None:
            # Sin ningún filtro activo: una copia "superficial" (comparte los
            # datos, cuesta casi nada) para que el resultado siga siendo un
            # objeto distinto de self.df, igual que antes con .copy().
            return df.copy(deep=False)
        return df[mascara]

    def _recortar_por_rango_tiempo(self, df):
        """Aplica a `df` el rango del slider de la Línea de Tiempo (si hay
        uno). Lo usan apply_table_filter() y LineaTiempoPanel para dibujar
        su vista de puntos, así los dos recortan exactamente las mismas
        filas. Con el buscador de filas puntuales activo el rango se ignora
        (ver el comentario de self.filas_buscadas)."""
        if self.filas_buscadas or not self.rango_tiempo:
            return df
        t0, t1, col_t, *resto = self.rango_tiempo
        if col_t not in df.columns:
            return df
        dayfirst = resto[0] if resto else True
        fechas = self._fechas_de_columna(df, col_t, dayfirst)
        return df[np.asarray((fechas >= t0) & (fechas <= t1))]

    def _fechas_de_columna(self, df, col, dayfirst):
        """Fechas ya interpretadas de la columna `col`, en el mismo orden
        que las filas de `df`. Interpretar fechas es lo más caro del recorte
        por rango: antes se rehacía completo en cada movimiento del slider.
        Ahora se interpreta la columna de self.df UNA vez y se guarda
        (self._cache_fechas); solo se vuelve a hacer si cambia la tabla, la
        columna, el formato dayfirst o el tamaño de la tabla, o si se editó/
        renombró algo a mano (esos dos casos limpian el caché a propósito).
        Si `df` no es un subconjunto de self.df, se interpreta directo, como
        antes, sin usar el caché."""
        base = self.df
        if base is not None and col in base.columns and col in df.columns:
            cache = getattr(self, "_cache_fechas", None)
            if (
                cache is None or cache["df"] is not base or cache["col"] != col
                or cache["dayfirst"] != dayfirst or cache["forma"] != base.shape
            ):
                resultado = parsear_fechas(base[col], dayfirst=dayfirst)
                if hasattr(resultado, "to_numpy"):
                    resultado = resultado.to_numpy()
                cache = {
                    "df": base, "col": col, "dayfirst": dayfirst, "forma": base.shape,
                    "fechas": pd.Series(np.asarray(resultado), index=base.index),
                }
                self._cache_fechas = cache
            if df is base:
                return cache["fechas"]
            if base.index.is_unique:
                posiciones = base.index.get_indexer(df.index)
                if (posiciones >= 0).all():
                    return cache["fechas"].iloc[posiciones]
        return parsear_fechas(df[col], dayfirst=dayfirst)

    def apply_table_filter(self):
        if self.df is None:
            return
        df = self._recortar_por_rango_tiempo(self._construir_df_filtrado_base())

        self.filtered_df = df
        self._refrescar_barra_filtro()
        self.table_model.set_dataframe(df)
        self.lbl_total_reg.setText(f"Total Registros: {len(df):,}")
        self.lbl_total_cols.setText(f"Total Columnas: {len(df.columns)}")

        self._actualizar_metricas()
        self._actualizar_indicadores()
        self._update_frecuencias()
        self._actualizar_alarmas()
        self.lbl_placeholder.setVisible(False)

        # Se salta a propósito cuando el propio slider de la Línea de
        # Tiempo es quien llamó a apply_table_filter (ver _actualizando_desde_slider
        # en linea_tiempo_ui.py): la base que usa este panel para dibujar
        # (_construir_df_filtrado_base) no incluye rango_tiempo, así que
        # arrastrar el slider nunca cambia lo que hay que redibujar --
        # solo redibuja cuando el cambio viene de OTRO lado (buscador de
        # filas, filtro por columna, clic en un gráfico).
        if hasattr(self, "panel_linea_tiempo") and not getattr(
            self.panel_linea_tiempo, "_actualizando_desde_slider", False
        ):
            self.panel_linea_tiempo.refrescar_vista()

        if hasattr(self, "narrativa_browser"):
            self._marcar_narrativa_desactualizada()

    # ------------------------------------------------------------------
    # TAB GRÁFICOS
    # ------------------------------------------------------------------
    def _build_tab_graficos(self, tab):
        layout = QVBoxLayout(tab)

        self.charts_scroll = QScrollArea()
        self.charts_scroll.setWidgetResizable(True)
        self.charts_container = QWidget()
        self.charts_layout = QVBoxLayout(self.charts_container)
        self.charts_layout.addStretch()
        self.charts_scroll.setWidget(self.charts_container)
        layout.addWidget(self.charts_scroll)

        btn_add = QPushButton("+ Agregar Gráfico")
        btn_add.setObjectName("accentButton")
        btn_add.clicked.connect(self.add_chart_panel)
        layout.addWidget(btn_add)

        self.add_chart_panel()

    def add_chart_panel(self):
        panel = ChartPanel(self.colors)
        panel.set_render_callback(lambda p: p.render(self.filtered_df))
        panel.removeRequested.connect(self.remove_chart_panel)
        panel.categoryClicked.connect(self._on_category_clicked)
        self.charts_layout.insertWidget(self.charts_layout.count() - 1, panel)
        self.chart_panels.append(panel)

        if self.df is not None:
            all_cols = self.df.columns.tolist()
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            panel.update_axis_choices(all_cols, num_cols)

        if self.filtered_df is not None:
            panel.render(self.filtered_df)

    def remove_chart_panel(self, panel):
        if len(self.chart_panels) <= 1:
            QMessageBox.warning(self, "Atención", "Debes mantener al menos un gráfico.")
            return
        self.chart_panels.remove(panel)
        panel.setParent(None)
        panel.deleteLater()

    def _on_category_clicked(self, columna, valor):
        self.filtro_grafico = {"columna": columna, "valor": valor}
        self._refrescar_barra_filtro()
        self.apply_table_filter()

    # ------------------------------------------------------------------
    # TAB MÉTRICAS
    # ------------------------------------------------------------------
    def _build_tab_metricas(self, tab):
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        btn_agregar = QPushButton("+ Agregar métrica de otra columna")
        btn_agregar.setObjectName("accentButton")
        btn_agregar.clicked.connect(lambda: self._agregar_bloque_metrica())
        toolbar.addWidget(btn_agregar)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contenedor = QWidget()
        self.metricas_layout = QVBoxLayout(contenedor)
        self.metricas_layout.addStretch()  # empuja los bloques hacia arriba
        scroll.setWidget(contenedor)
        layout.addWidget(scroll, stretch=1)

        self.metricas_bloques = []  # cada uno: dict con columna/frame/cards/lbls/plot
        self._actualizar_metricas()

    def _columnas_numericas_disponibles_para_metricas(self):
        if self.df is None:
            return []
        ya_agregadas = {b["columna"] for b in self.metricas_bloques}
        return [
            str(c) for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c]) and str(c) not in ya_agregadas
        ]

    def _agregar_bloque_metrica(self, columna=None):
        if self.df is None:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        if columna is None:
            disponibles = self._columnas_numericas_disponibles_para_metricas()
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
        frame.setObjectName("card")
        frame_layout = QVBoxLayout(frame)

        encabezado = QHBoxLayout()
        lbl_titulo = QLabel(f"<b>Métricas: {columna}</b>")
        encabezado.addWidget(lbl_titulo)
        encabezado.addStretch()
        btn_quitar = QPushButton("Quitar")
        encabezado.addWidget(btn_quitar)
        frame_layout.addLayout(encabezado)

        cuerpo = QHBoxLayout()

        columna_tarjetas = QVBoxLayout()
        card_media, lbl_media = create_stat_card("Media")
        card_mediana, lbl_mediana = create_stat_card("Mediana")
        card_moda, lbl_moda = create_stat_card("Moda")
        card_std, lbl_std = create_stat_card("Desviación Std")
        for card in (card_media, card_mediana, card_moda, card_std):
            columna_tarjetas.addWidget(card)
        cuerpo.addLayout(columna_tarjetas)

        plot = pg.PlotWidget()
        plot.setMinimumHeight(220)
        plot.setMinimumWidth(280)
        cuerpo.addWidget(plot, stretch=1)

        frame_layout.addLayout(cuerpo)

        bloque = {
            "columna": columna,
            "frame": frame,
            "lbl_titulo": lbl_titulo,
            "cards": {"Media": card_media, "Mediana": card_mediana, "Moda": card_moda, "Desviación Std": card_std},
            "lbls": {"Media": lbl_media, "Mediana": lbl_mediana, "Moda": lbl_moda, "Desviación Std": lbl_std},
            "plot": plot,
        }
        btn_quitar.clicked.connect(lambda: self._quitar_bloque_metrica(bloque))

        # Se inserta antes del stretch final, para que los bloques nuevos
        # se apilen de arriba hacia abajo y el espacio vacío quede al fondo.
        self.metricas_layout.insertWidget(self.metricas_layout.count() - 1, frame)
        self.metricas_bloques.append(bloque)
        self._recalcular_bloque_metrica(bloque)
        self._actualizar_alarmas()

    def _quitar_bloque_metrica(self, bloque):
        bloque["frame"].setParent(None)
        bloque["frame"].deleteLater()
        if bloque in self.metricas_bloques:
            self.metricas_bloques.remove(bloque)
        # Las alarmas de tipo Métrica sobre esta columna NO se borran ni se
        # desactivan -- siguen evaluándose y pueden seguir mandando correo,
        # simplemente ya no tienen dónde pintarse en esta pestaña.

    def _recalcular_bloque_metrica(self, bloque):
        columna = bloque["columna"]
        if self.filtered_df is None or self.filtered_df.empty or columna not in self.filtered_df.columns:
            return
        data = pd.to_numeric(self.filtered_df[columna], errors="coerce").dropna()
        if data.empty:
            return

        bloque["lbls"]["Media"].setText(f"{data.mean():,.2f}")
        bloque["lbls"]["Mediana"].setText(f"{data.median():,.2f}")
        moda = data.mode()
        bloque["lbls"]["Moda"].setText(f"{moda.iloc[0]:,.2f}" if not moda.empty else "N/A")
        bloque["lbls"]["Desviación Std"].setText(f"{data.std():,.2f}")

        plot = bloque["plot"]
        plot.setBackground(self.colors["card"])
        plot.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        plot.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        render_boxplot(plot, data, self.colors)

    def _actualizar_metricas(self):
        if not hasattr(self, "metricas_bloques"):
            return
        for bloque in self.metricas_bloques:
            self._recalcular_bloque_metrica(bloque)
        self._actualizar_alarmas()

    # ------------------------------------------------------------------
    # TAB INDICADORES
    # ------------------------------------------------------------------
    def _build_tab_indicadores(self, tab):
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        btn_nuevo = QPushButton("+ Nuevo Indicador")
        btn_nuevo.setObjectName("accentButton")
        btn_nuevo.clicked.connect(self._abrir_dialogo_nuevo_indicador)
        toolbar.addWidget(btn_nuevo)

        btn_sugerir = QPushButton("Sugerir Automáticamente")
        btn_sugerir.clicked.connect(self._sugerir_indicadores_auto)
        toolbar.addWidget(btn_sugerir)

        toolbar.addStretch()

        btn_guardar = QPushButton("Guardar Indicadores")
        btn_guardar.clicked.connect(self._guardar_indicadores)
        toolbar.addWidget(btn_guardar)

        btn_cargar = QPushButton("Cargar Indicadores")
        btn_cargar.clicked.connect(self._cargar_indicadores)
        toolbar.addWidget(btn_cargar)
        layout.addLayout(toolbar)

        aviso = QLabel(
            "Crea tus propios indicadores: una operación básica sobre una columna (suma, promedio, "
            "conteo, etc.) o una fórmula simple combinando columnas. Se recalculan solos cada vez que "
            "cambian los datos o los filtros."
        )
        aviso.setObjectName("muted")
        aviso.setWordWrap(True)
        layout.addWidget(aviso)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contenedor = QWidget()
        self.indicadores_grid = QGridLayout(contenedor)
        scroll.setWidget(contenedor)
        layout.addWidget(scroll, stretch=1)

        self.indicadores = []
        self.indicador_widgets = []
        self._actualizar_indicadores()

    def _abrir_dialogo_nuevo_indicador(self):
        if self.df is None:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        dialogo = DialogoIndicador(list(self.df.columns), parent=self)
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            self.indicadores.append(dialogo.get_indicador())
            self._actualizar_indicadores()

    def _editar_indicador(self, indicador):
        if self.df is None:
            return
        dialogo = DialogoIndicador(list(self.df.columns), indicador=indicador, parent=self)
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            idx = self.indicadores.index(indicador)
            self.indicadores[idx] = dialogo.get_indicador()
            self._actualizar_indicadores()

    def _eliminar_indicador(self, indicador):
        if indicador in self.indicadores:
            self.indicadores.remove(indicador)
            self._actualizar_indicadores()

    def _sugerir_indicadores_auto(self):
        if self.df is None:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        self.indicadores.extend(sugerir_indicadores_por_defecto(self.df))
        self._actualizar_indicadores()

    def _actualizar_indicadores(self):
        if not hasattr(self, "indicadores_grid"):
            return
        for w in self.indicador_widgets:
            w.setParent(None)
            w.deleteLater()
        self.indicador_widgets.clear()

        # self._ultimas_anomalias la llena la pestaña Narrativa al generar el
        # informe -- si todavía no se generó ninguna narrativa en esta
        # sesión, queda vacía y ningún indicador se marca como afectado (no
        # es un cálculo aparte, reutiliza lo que Narrativa ya detectó).
        columnas_con_anomalias = {
            c for a in getattr(self, "_ultimas_anomalias", []) for c in a.get("columnas", [])
        }

        df = self.filtered_df if self.filtered_df is not None else self.df
        for i, indicador in enumerate(self.indicadores):
            valor = indicador.calcular(df) if df is not None else None
            regla = next(
                (r for r in self.reglas_alarma if r["tipo"] == "Indicador" and r["objetivo"] == indicador.nombre),
                None,
            )
            en_alarma = evaluar_regla(regla, valor) if regla else False
            card = IndicadorCard(
                indicador, valor, self._editar_indicador, self._eliminar_indicador,
                en_alarma=en_alarma, mensaje_alarma=regla.get("mensaje") if regla else None,
                columnas_con_anomalias=columnas_con_anomalias,
            )
            self.indicadores_grid.addWidget(card, i // 4, i % 4)
            self.indicador_widgets.append(card)

        self._actualizar_alarmas()
        self._notificar_panel()

    def _guardar_indicadores(self):
        if not self.indicadores:
            QMessageBox.information(self, "Sin indicadores", "No hay indicadores para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar indicadores", "indicadores.json", "JSON (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([ind.to_dict() for ind in self.indicadores], f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
            return
        QMessageBox.information(self, "Listo", f"Indicadores guardados en:\n{path}")

    def _cargar_indicadores(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cargar indicadores", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nuevos = [Indicador.from_dict(item) for item in data]
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar", str(e))
            return
        self.indicadores.extend(nuevos)
        self._actualizar_indicadores()

    # ------------------------------------------------------------------
    # TAB FRECUENCIAS
    # ------------------------------------------------------------------
    def _build_tab_frecuencias(self, tab):
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        top.addWidget(QLabel("Columna(s) a analizar:"))
        self.freq_col_list = QListWidget()
        self.freq_col_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.freq_col_list.setMaximumHeight(80)
        self.freq_col_list.itemSelectionChanged.connect(
            lambda: (self._update_frecuencias(), self._actualizar_alarmas())
        )
        top.addWidget(self.freq_col_list, stretch=1)
        layout.addLayout(top)

        cards_row = QHBoxLayout()
        self.card_freq_moda, self.lbl_freq_moda = create_stat_card("Más Frecuente (Moda)")
        self.card_freq_menos, self.lbl_freq_menos = create_stat_card("Menos Frecuente")
        self.card_freq_unicos, self.lbl_freq_unicos = create_stat_card("Categorías Únicas")
        for card in (self.card_freq_moda, self.card_freq_menos, self.card_freq_unicos):
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        # Resumen de anomalías (según el último informe generado en
        # Narrativa) -- independiente de qué columna(s) esté mirando el
        # usuario más arriba, así que no depende de freq_col_list.
        cards_row_anomalias = QHBoxLayout()
        self.card_freq_anomalia_top, self.lbl_freq_anomalia_top = create_stat_card(
            "Anomalía Más Frecuente"
        )
        self.card_freq_anomalia_col, self.lbl_freq_anomalia_col = create_stat_card(
            "Columna Más Afectada"
        )
        for card in (self.card_freq_anomalia_top, self.card_freq_anomalia_col):
            cards_row_anomalias.addWidget(card)
        layout.addLayout(cards_row_anomalias)
        self.lbl_freq_anomalia_top.setText("Genera Narrativa primero")
        self.lbl_freq_anomalia_col.setText("Genera Narrativa primero")

        body_widget = QWidget()
        body = QHBoxLayout(body_widget)

        self.freq_table_model = PandasTableModel()
        self.freq_table_view = QTableView()
        self.freq_table_view.setModel(self.freq_table_model)
        self.freq_table_view.setAlternatingRowColors(True)
        self.freq_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.freq_table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.freq_table_view.clicked.connect(self._mostrar_celda_en_status)
        body.addWidget(self.freq_table_view, stretch=1)

        self.freq_plot = pg.PlotWidget()
        self.freq_plot.setMinimumWidth(320)
        body.addWidget(self.freq_plot, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body_widget)
        layout.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------
    # TAB NARRATIVA
    # ------------------------------------------------------------------
    def _build_tab_narrativa(self, tab):
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.btn_generar_narrativa = QPushButton("Generar Narrativa")
        self.btn_generar_narrativa.setObjectName("accentButton")
        self.btn_generar_narrativa.clicked.connect(self._generar_narrativa)
        controls.addWidget(self.btn_generar_narrativa)

        self.btn_config_relaciones_temporales = QPushButton("Configurar relaciones de fechas")
        self.btn_config_relaciones_temporales.clicked.connect(self._configurar_relaciones_temporales)
        controls.addWidget(self.btn_config_relaciones_temporales)

        self.btn_narrativa_exportar_pdf = QPushButton("Exportar a PDF")
        self.btn_narrativa_exportar_pdf.clicked.connect(self._narrativa_exportar_pdf)
        controls.addWidget(self.btn_narrativa_exportar_pdf)

        self.btn_ml_multivariado = QPushButton("Detección por combinación (ML): Desactivado")
        self.btn_ml_multivariado.setCheckable(True)
        self.btn_ml_multivariado.toggled.connect(self._alternar_ml_multivariado)
        controls.addWidget(self.btn_ml_multivariado)

        self.lbl_narrativa_estado = QLabel(
            "Genera un informe con los hallazgos automáticos de tus datos actuales."
        )
        self.lbl_narrativa_estado.setObjectName("muted")
        controls.addWidget(self.lbl_narrativa_estado)
        controls.addStretch()
        layout.addLayout(controls)

        # Informe y Linaje son sub-pestañas DENTRO de Narrativa (no
        # pestañas del nivel superior): Linaje solo tiene sentido después
        # de generar un informe (necesita self._anomalias_por_tabla y las
        # relaciones ya calculadas), así que vivir junto al informe dentro
        # del mismo botón "Generar Narrativa" deja esa dependencia clara,
        # en vez de una pestaña suelta que se ve rota si se abre primero.
        self.subtabs_narrativa = QTabWidget()
        # Ocultas hasta generar Narrativa por primera vez: Linaje no tiene sentido
        # antes (necesita las relaciones y anomalías de un informe), y dejar el
        # espacio libre mientras tanto evita dos sub-pestañas que se ven rotas o
        # vacías apenas se abre esta pestaña. Una vez generada la primera vez,
        # quedan visibles aunque el informe se marque desactualizado después
        # (el contenido anterior sigue siendo útil; ver _marcar_narrativa_desactualizada).
        self.subtabs_narrativa.setVisible(False)
        layout.addWidget(self.subtabs_narrativa, stretch=1)

        self.lbl_narrativa_vacia = QLabel(
            "Presiona \"Generar Narrativa\" para ver aquí el Informe y el Linaje."
        )
        self.lbl_narrativa_vacia.setObjectName("muted")
        self.lbl_narrativa_vacia.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_narrativa_vacia, stretch=1)

        tab_informe = QWidget()
        layout_informe = QVBoxLayout(tab_informe)
        self.narrativa_browser = QTextBrowser()
        self.narrativa_browser.setOpenExternalLinks(False)
        # Los enlaces "Marcar como resuelta" no son páginas de verdad, así que
        # se desactiva la navegación automática y se maneja el clic a mano.
        self.narrativa_browser.setOpenLinks(False)
        self.narrativa_browser.anchorClicked.connect(self._on_narrativa_anchor_clicked)
        self.narrativa_browser.setMinimumHeight(400)
        layout_informe.addWidget(self.narrativa_browser)
        self.subtabs_narrativa.addTab(tab_informe, "Informe")

        tab_linaje = QWidget()
        self._build_subtab_linaje(tab_linaje)
        self.subtabs_narrativa.addTab(tab_linaje, "Linaje")

        self._narrativa_actualizada = False
        self._narrativa_generada_alguna_vez = False
        self._ultimas_anomalias = []
        self._anomalias_por_tabla = {}
        self._linaje_disponible = False
        self._ultimo_df_narrativa = None
        self._ultimo_nombre_dataset_narrativa = "tu dataset"
        self._actualizar_estado_linaje()

    def _build_subtab_linaje(self, tab):
        """Sub-pestaña 'Linaje' de Narrativa: parte de UN registro puntual
        (por ID, o eligiéndolo directamente de la lista de anomalías) y
        muestra en una lista anidada (con sangría, sin líneas ni flechas)
        todo lo que está conectado con él a través de las relaciones ya
        confirmadas -- hacia tablas padre Y tablas hijas. A propósito es
        una lista, no un grafo dibujado: la relación real entre registros
        casi siempre es un árbol, y una lista anidada se lee sin ambigüedad
        de layout ni líneas cruzándose."""
        layout = QVBoxLayout(tab)

        self.lbl_linaje_estado = QLabel(
            "Genera Narrativa primero -- Linaje necesita las relaciones y anomalías de tu último informe."
        )
        self.lbl_linaje_estado.setObjectName("muted")
        layout.addWidget(self.lbl_linaje_estado)

        buscar_row = QHBoxLayout()
        buscar_row.addWidget(QLabel("Tabla:"))
        self.combo_linaje_tabla = QComboBox()
        self.combo_linaje_tabla.currentTextChanged.connect(self._on_linaje_tabla_change)
        buscar_row.addWidget(self.combo_linaje_tabla)

        buscar_row.addWidget(QLabel("Columna:"))
        self.combo_linaje_columna = QComboBox()
        buscar_row.addWidget(self.combo_linaje_columna)

        buscar_row.addWidget(QLabel("Valor (ID):"))
        self.txt_linaje_valor = QLineEdit()
        self.txt_linaje_valor.setPlaceholderText("ej. 102")
        self.txt_linaje_valor.returnPressed.connect(self._buscar_linaje)
        buscar_row.addWidget(self.txt_linaje_valor)

        self.btn_linaje_buscar = QPushButton("Ver linaje")
        self.btn_linaje_buscar.clicked.connect(self._buscar_linaje)
        buscar_row.addWidget(self.btn_linaje_buscar)

        buscar_row.addWidget(QLabel("Profundidad:"))
        self.spin_linaje_profundidad = QSpinBox()
        self.spin_linaje_profundidad.setRange(1, 4)
        self.spin_linaje_profundidad.setValue(2)
        self.spin_linaje_profundidad.setToolTip(
            "Cuántos pasos de relación explorar desde el registro elegido "
            "(hacia tablas padre e hijas). Un número más alto puede tardar "
            "más en esquemas grandes."
        )
        buscar_row.addWidget(self.spin_linaje_profundidad)

        buscar_row.addStretch()
        layout.addLayout(buscar_row)

        cuerpo = QHBoxLayout()

        columna_izq = QVBoxLayout()
        columna_izq.addWidget(QLabel("O elige una anomalía detectada:"))
        self.lista_linaje_anomalias = QListWidget()
        self.lista_linaje_anomalias.setMaximumWidth(320)
        self.lista_linaje_anomalias.itemDoubleClicked.connect(self._on_linaje_anomalia_elegida)
        columna_izq.addWidget(self.lista_linaje_anomalias)
        cuerpo.addLayout(columna_izq)

        # El diagrama de cajas conectadas (linaje_grafo.py / linaje_ui.py) reemplaza
        # al árbol con sangría: misma información (registros conectados siguiendo
        # las relaciones del esquema), al estilo Palantir. El clic derecho en una
        # caja ("Explorar desde aquí") re-ancla la búsqueda ahí -- mismo
        # comportamiento que antes tenía el árbol, ahora desde una caja.
        self.panel_linaje = PanelLinaje(self.colors, al_explorar_desde=self._explorar_linaje_desde)
        cuerpo.addWidget(self.panel_linaje, stretch=1)

        layout.addLayout(cuerpo, stretch=1)

        self._controles_linaje = [
            self.combo_linaje_tabla, self.combo_linaje_columna, self.txt_linaje_valor,
            self.btn_linaje_buscar, self.lista_linaje_anomalias, self.spin_linaje_profundidad,
            self.panel_linaje,
        ]

        # Conectado al final, después de crear todos los widgets del
        # bloque: setValue(2) más arriba ya dispara valueChanged, y en ese
        # momento self.panel_linaje todavía no existiría si esto se
        # conectara antes.
        self.spin_linaje_profundidad.valueChanged.connect(self._on_linaje_profundidad_change)

    def _actualizar_estado_linaje(self):
        """Habilita o deshabilita toda la sub-pestaña Linaje según si hay
        un informe de Narrativa generado sobre los datos actuales
        (self._linaje_disponible). Se llama tanto al generar Narrativa como
        en cada punto donde esos datos quedan obsoletos (ver
        _resetear_filtro_anomalias)."""
        if not hasattr(self, "lbl_linaje_estado"):
            return  # la sub-pestaña todavía no se construyó
        disponible = bool(self._linaje_disponible)
        for w in self._controles_linaje:
            w.setEnabled(disponible)
        if disponible:
            self.lbl_linaje_estado.setText(
                "Elige una tabla y un ID, o doble clic en una anomalía de la izquierda."
            )
        else:
            self.panel_linaje.mostrar(None, self.colors)
            self.lista_linaje_anomalias.clear()
            self.lbl_linaje_estado.setText(
                "Genera Narrativa primero -- Linaje necesita las relaciones y anomalías de tu último informe."
            )

    def _poblar_linaje_tras_narrativa(self):
        """Se llama al final de _generar_narrativa(): repuebla los combos
        de tabla/columna y la lista de anomalías clicables de Linaje con lo
        recién calculado."""
        self.combo_linaje_tabla.blockSignals(True)
        self.combo_linaje_tabla.clear()
        self.combo_linaje_tabla.addItems(sorted(self.tablas.keys()))
        self.combo_linaje_tabla.setCurrentText(self.nombre_tabla_activa)
        self.combo_linaje_tabla.blockSignals(False)
        self._on_linaje_tabla_change(self.nombre_tabla_activa)

        self.lista_linaje_anomalias.clear()
        for a in self._ultimas_anomalias:
            idxs = a.get("indices_atipicos")
            idx_unico = idxs[0] if idxs else a.get("fila_indice")
            if idx_unico is None:
                continue
            etiqueta = ETIQUETAS_TIPO_ANOMALIA.get(a.get("tipo"), a.get("tipo") or "Anomalía")
            item = QListWidgetItem(f"{etiqueta} — {self.nombre_tabla_activa} (fila {idx_unico})")
            item.setData(Qt.ItemDataRole.UserRole, (self.nombre_tabla_activa, idx_unico))
            self.lista_linaje_anomalias.addItem(item)

    def _on_linaje_tabla_change(self, nombre_tabla):
        self.combo_linaje_columna.clear()
        df_t = self.tablas.get(nombre_tabla)
        if df_t is None:
            return
        self.combo_linaje_columna.addItems([str(c) for c in df_t.columns])
        relaciones = (
            self.relaciones_ontologia if self.relaciones_ontologia is not None
            else inferir_relaciones(self.tablas)
        )
        columna_sugerida = columna_clave_de_tabla(nombre_tabla, relaciones)
        if columna_sugerida and columna_sugerida in df_t.columns:
            self.combo_linaje_columna.setCurrentText(columna_sugerida)

    def _on_linaje_anomalia_elegida(self, item):
        nombre_tabla, indice_fila = item.data(Qt.ItemDataRole.UserRole)
        df_t = self.tablas.get(nombre_tabla)
        if df_t is None or indice_fila not in df_t.index:
            return
        self.combo_linaje_tabla.setCurrentText(nombre_tabla)  # repuebla combo_linaje_columna con la sugerida
        columna = self.combo_linaje_columna.currentText() or df_t.columns[0]
        if columna not in df_t.columns:
            return
        valor = df_t.at[indice_fila, columna]
        self.txt_linaje_valor.setText(str(valor))
        self._buscar_linaje()

    def _on_linaje_profundidad_change(self, _valor):
        # Solo re-busca si ya hay algo cargado -- si el usuario todavía no
        # elige un ID, tocar el spinbox no debe lanzar una búsqueda vacía.
        if self.txt_linaje_valor.text().strip():
            self._buscar_linaje()

    def _buscar_linaje(self):
        tabla = self.combo_linaje_tabla.currentText()
        columna = self.combo_linaje_columna.currentText()
        texto_valor = self.txt_linaje_valor.text().strip()
        df_t = self.tablas.get(tabla)
        if not tabla or not columna or not texto_valor or df_t is None:
            return

        valor = cast_valor_a_dtype(texto_valor, df_t[columna].dtype)
        relaciones = (
            self.relaciones_ontologia if self.relaciones_ontologia is not None
            else inferir_relaciones(self.tablas)
        )
        raiz = explorar_linaje(
            tabla, valor, columna, self.tablas, relaciones,
            filas_anomalas=self._anomalias_por_tabla,
            max_saltos=self.spin_linaje_profundidad.value(),
        )
        if raiz is None:
            self.panel_linaje.mostrar(None, self.colors)
            self.lbl_linaje_estado.setText(f'La columna "{columna}" no existe en {tabla}.')
            return
        if raiz.indice_fila is None:
            self.panel_linaje.mostrar(None, self.colors, mensaje_vacio=(
                f'No se encontró {columna} = "{texto_valor}" en {tabla}.'
            ))
            self.lbl_linaje_estado.setText(f'No se encontró {columna} = "{texto_valor}" en {tabla}.')
            return

        self.lbl_linaje_estado.setText(
            "Elige una tabla y un ID, o doble clic en una anomalía de la izquierda."
        )
        self.panel_linaje.mostrar(construir_grafo_linaje(raiz, tablas=self.tablas), self.colors)

    def _explorar_linaje_desde(self, tabla, columna, valor):
        """Re-ancla la búsqueda de Linaje en un registro que apareció como
        nodo conectado (clic derecho en una caja de linaje_ui.py), reutilizando los
        mismos combos/campo de texto que ya usa una búsqueda manual -- así
        el usuario ve con claridad desde dónde está explorando ahora."""
        if tabla not in self.tablas:
            return
        self.combo_linaje_tabla.blockSignals(True)
        self.combo_linaje_tabla.setCurrentText(tabla)
        self.combo_linaje_tabla.blockSignals(False)
        self._on_linaje_tabla_change(tabla)
        columnas_disponibles = [self.combo_linaje_columna.itemText(i) for i in range(self.combo_linaje_columna.count())]
        if columna in columnas_disponibles:
            self.combo_linaje_columna.setCurrentText(columna)
        self.txt_linaje_valor.setText(str(valor))
        self._buscar_linaje()

    def _marcar_narrativa_desactualizada(self):
        """Los datos/filtros cambiaron desde la última narrativa generada:
        no se recalcula sola (puede ser costoso con BDs grandes), pero se
        avisa que conviene volver a generarla."""
        self._narrativa_actualizada = False
        if hasattr(self, "lbl_narrativa_estado"):
            self.lbl_narrativa_estado.setText(
                "Los datos cambiaron desde la última narrativa generada. "
                "Presiona \"Generar Narrativa\" para actualizarla."
            )

    def _alternar_ml_multivariado(self, activar):
        """Prende/apaga la detección por combinación de columnas (Isolation
        Forest). Al activarla, corre el auto-test de confiabilidad SOBRE
        ESTE dataset y le muestra al usuario un número medido de verdad, no
        una tabla genérica -- ver deteccion_multivariada.evaluar_confiabilidad."""
        df = self.filtered_df if self.filtered_df is not None else self.df
        if df is None or df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            self.btn_ml_multivariado.setChecked(False)
            return

        if activar:
            resultado = evaluar_confiabilidad(df)
            if resultado is None:
                QMessageBox.information(
                    self, "Dataset muy chico",
                    "Este dataset no tiene suficientes filas o columnas numéricas para que "
                    "la detección por combinación (ML) aporte algo confiable."
                )
                self.btn_ml_multivariado.setChecked(False)
                return

            tasa = resultado["tasa_deteccion"] * 100
            mensaje = (
                f"Con tus datos actuales, este chequeo detectó el {tasa:.0f}% de las "
                f"anomalías de prueba fabricadas para medirlo ({resultado['n_pruebas']} pruebas "
                f"sobre {resultado['n_filas_dataset']:,} filas).\n\n"
            )
            mensaje += (
                "Es un resultado confiable para activarlo en este dataset."
                if resultado["confiable"] else
                "Es un resultado bajo -- puede deberse a pocas filas o poca variación entre "
                "columnas. Activarlo igual es tu decisión, pero puede marcar cosas poco útiles."
            )
            respuesta = QMessageBox.question(
                self, "Detección por combinación de columnas (ML)",
                mensaje + "\n\n¿Activar este chequeo para la narrativa?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if respuesta != QMessageBox.Yes:
                self.btn_ml_multivariado.setChecked(False)
                return

        self.ml_multivariado_activado = activar
        self.btn_ml_multivariado.setText(
            "Detección por combinación (ML): Activado" if activar
            else "Detección por combinación (ML): Desactivado"
        )
        self._marcar_narrativa_desactualizada()

    def _configurar_relaciones_temporales(self):
        df = self.filtered_df if self.filtered_df is not None else self.df
        if df is None or df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        columnas_fecha = detectar_columnas_fecha(df)
        pares_iniciales = []
        if self.fingerprint_actual:
            pares_iniciales = self.memoria.obtener_pares_temporales(self.fingerprint_actual)

        dialogo = ConfigurarRelacionesTemporalesDialog(columnas_fecha, pares_iniciales, parent=self)
        if dialogo.exec() == QDialog.Accepted:
            pares = dialogo.obtener_pares()
            if self.fingerprint_actual:
                self.memoria.guardar_pares_temporales(self.fingerprint_actual, pares)
            self._marcar_narrativa_desactualizada()

    def _generar_narrativa(self):
        df = self.filtered_df if self.filtered_df is not None else self.df
        if df is None or df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        nombre_dataset = self.lbl_archivo.text().split("\n")[0] if hasattr(self, "lbl_archivo") else "tu dataset"
        if not nombre_dataset or nombre_dataset == "Ningún archivo cargado.":
            nombre_dataset = "tu dataset"

        pares_temporales_personalizados = []
        if self.fingerprint_actual:
            pares_temporales_personalizados = self.memoria.obtener_pares_temporales(self.fingerprint_actual)

        detector = SemanticAnomalyDetector(
            df, pares_temporales_personalizados=pares_temporales_personalizados,
            linea_base_ml=self.linea_base_ml if self.ml_activado else None,
            nombre_tabla=self.nombre_tabla_activa,
            ml_multivariado_activado=self.ml_multivariado_activado,
        )
        anomalias = detector.detect_all()

        # Le suma a cada anomalía su historial (cuántas veces se ha visto,
        # hace cuántos días, si sigue sin resolverse) -- esto es lo que hace
        # que el informe "recuerde" en vez de partir de cero cada vez.
        if self.fingerprint_actual:
            anomalias = self.memoria.enriquecer_con_memoria(self.fingerprint_actual, anomalias)

        html = self._generar_html_narrativa(df, anomalias, nombre_dataset)

        self.narrativa_browser.setHtml(html)
        self._ultimas_anomalias = anomalias
        self._ultimo_df_narrativa = df
        self._ultimo_nombre_dataset_narrativa = nombre_dataset
        self._notificar_panel()
        self._narrativa_actualizada = True
        if not self._narrativa_generada_alguna_vez:
            self._narrativa_generada_alguna_vez = True
            self.lbl_narrativa_vacia.setVisible(False)
            self.subtabs_narrativa.setVisible(True)
        self._linaje_disponible = True
        self._actualizar_estado_linaje()
        self._poblar_linaje_tras_narrativa()

        # Habilita/repuebla el filtro "Anomalía" de Datos y el resumen de
        # anomalías de Frecuencias con lo recién calculado -- ambos leen
        # self._ultimas_anomalias, no recalculan nada por su cuenta.
        self._repoblar_filtro_anomalias()
        self._actualizar_resumen_anomalias_frecuencias()

        # Reflejar las anomalías detectadas como notas automáticas en la
        # pestaña Datos (círculo rojo + fondo rosado + tooltip con el motivo).
        # Esto es siempre sobre la tabla activa -- las otras tablas de la
        # base relacional (si las hay) no se marcan en Datos, solo entran
        # en el capítulo "Cómo se cruzan tus tablas" del informe.
        self.table_model.limpiar_notas_narrativa()
        notas_celda = anomalias_a_notas_celda(anomalias, df)
        for row_label, col_name, texto, exclusiva_ml in notas_celda:
            self.table_model.agregar_nota(
                row_label, col_name, texto, es_anomalia=True, origen_narrativa=True,
                exclusiva_ml=exclusiva_ml, emitir=False,
            )
        if notas_celda:
            self.table_model.layoutChanged.emit()

        celdas_txt = f" · {len(notas_celda):,} celda(s) marcada(s) en Datos" if notas_celda else ""
        extra_relacional = f" · {len(self.tablas)} tablas relacionadas" if len(self.tablas) > 1 else ""
        self.lbl_narrativa_estado.setText(
            f"Narrativa generada sobre {len(df):,} fila(s) · {len(anomalias)} anomalía(s) detectada(s)"
            f"{celdas_txt}{extra_relacional}."
        )

    def _generar_html_narrativa(self, df, anomalias, nombre_dataset, modo_impresion=False):
        """Arma el HTML del informe. Con 1 sola tabla, es EXACTAMENTE lo de
        siempre. Con 2+ tablas cargadas, además detecta anomalías en las
        demás tablas (sobre sus datos completos, sin los filtros que estén
        aplicados a la tabla activa) y arma el cruce entre ellas usando el
        esquema confirmado en "Ver esquema" -- o, si el usuario todavía no
        lo abrió esta vez, el esquema automático de siempre (sin guardarlo).
        modo_impresion=True arma una versión sin los elementos que solo
        tienen sentido dentro de la app (ej. "Marcar como resuelta"),
        pensada para exportar a PDF o imprimir."""
        self.procedencia.podar(self.tablas)
        eventos_proc = self.procedencia.eventos_de_tabla(self.nombre_tabla_activa)
        if len(self.tablas) <= 1:
            self._anomalias_por_tabla = {
                self.nombre_tabla_activa: _todos_los_indices_anomalos(anomalias)
            }
            return NarrativeGenerator(
                df, anomalias, indicadores=self.indicadores, modo_impresion=modo_impresion,
                eventos_procedencia=eventos_proc,
            ).generar_html(nombre_dataset)

        relaciones = (
            self.relaciones_ontologia if self.relaciones_ontologia is not None
            else inferir_relaciones(self.tablas)
        )

        tablas_relacionadas = {}
        self._anomalias_por_tabla = {
            self.nombre_tabla_activa: _todos_los_indices_anomalos(anomalias)
        }
        for nombre_t, df_t in self.tablas.items():
            if nombre_t == self.nombre_tabla_activa:
                continue
            # El interruptor de ML (self.ml_multivariado_activado) se probó
            # y confirmó solo sobre la tabla activa (ver _alternar_ml_multivariado).
            # Antes se aplicaba igual a TODAS las demás tablas sin medir nada
            # ahí: podía activarse "a ciegas" en una tabla donde el mismo
            # chequeo de confiabilidad habría dado un resultado bajo. Ahora
            # se vuelve a medir para esta tabla en particular (barato: usa
            # muestra y pocos árboles, ver evaluar_confiabilidad) y solo se
            # activa si el resultado es confiable para ELLA.
            #
            # Además (v1 simple, ver ontologia.enriquecer_tabla_con_relaciones):
            # si esta tabla es la "principal" de alguna relación 1-a-muchos
            # con dirección clara, el ML no la mira sola -- la mira con
            # columnas extra resumiendo sus tablas hijas (cantidad, suma,
            # promedio), igual que hace Foundry al aplanar su Ontología
            # antes de correr el modelo. El resto de los chequeos (reglas,
            # nulos, duplicados) siguen viendo la tabla tal cual, sin tocar.
            df_t_para_ml = enriquecer_tabla_con_relaciones(nombre_t, self.tablas, relaciones)
            ml_confiable_para_esta_tabla = False
            if self.ml_multivariado_activado:
                resultado_ml_tabla = evaluar_confiabilidad(df_t_para_ml)
                ml_confiable_para_esta_tabla = bool(
                    resultado_ml_tabla and resultado_ml_tabla["confiable"]
                )
            detector_t = SemanticAnomalyDetector(
                df_t,
                linea_base_ml=self.linea_base_ml if self.ml_activado else None,
                nombre_tabla=nombre_t,
                ml_multivariado_activado=ml_confiable_para_esta_tabla,
                df_multivariado=df_t_para_ml,
            )
            anomalias_t = detector_t.detect_all()
            tablas_relacionadas[nombre_t] = {"df": df_t, "anomalias": anomalias_t}
            self._anomalias_por_tabla[nombre_t] = _todos_los_indices_anomalos(anomalias_t)

        return NarrativeGenerator(
            df, anomalias,
            nombre_tabla=self.nombre_tabla_activa,
            tablas_relacionadas=tablas_relacionadas,
            relaciones=relaciones,
            indicadores=self.indicadores,
            modo_impresion=modo_impresion,
            eventos_procedencia=eventos_proc,
        ).generar_html(nombre_dataset)

    def _on_narrativa_anchor_clicked(self, url):
        """Captura el clic en 'Marcar como resuelta' dentro del informe.
        No es una navegación real -- por eso narrativa_browser tiene
        setOpenLinks(False) -- así que esto es lo único que hace algo con
        ese clic."""
        href = url.toString()
        if not href.startswith("resolver:"):
            return

        identidad = _decodificar_identidad_anomalia(href[len("resolver:"):])
        if not identidad or not self.fingerprint_actual:
            return

        self.memoria.marcar_resuelta(
            self.fingerprint_actual, identidad.get("tipo"), identidad.get("columnas") or []
        )
        self.lbl_narrativa_estado.setText(
            "Anomalía marcada como resuelta. Si vuelve a aparecer, Hadar te lo va a hacer notar."
        )
        # Se regenera para que el informe deje de mostrar la frase de
        # recurrencia de esa anomalía (ya no está "sin resolver").
        self._generar_narrativa()

    def _narrativa_exportar_pdf(self):
        if not self.narrativa_browser.toPlainText().strip():
            QMessageBox.information(
                self, "Sin narrativa", "Presiona \"Generar Narrativa\" antes de exportar."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Narrativa a PDF", "narrativa.pdf", "PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        # ScreenResolution (no HighResolution): QTextDocument interpreta los
        # tamaños en px del CSS del informe asumiendo la resolución lógica
        # de pantalla (96 DPI). Con HighResolution, QPrinter imprime a una
        # resolución física mucho mayor y Qt no reescala la fuente al pasar
        # de una referencia a otra -- el texto queda microscópico en el PDF
        # aunque se vea normal en pantalla. Usando la misma referencia de
        # DPI en ambos lados, el tamaño del PDF queda igual al que se ve en
        # la vista previa.
        printer = QPrinter(QPrinter.PrinterMode.ScreenResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))

        # El documento que se ve en pantalla tiene elementos pensados solo
        # para la app (el enlace "Marcar como resuelta") que no significan
        # nada en un PDF estático -- por eso se arma una versión aparte,
        # limpia, en vez de imprimir tal cual lo que está en pantalla.
        html_impresion = self._generar_html_narrativa(
            self._ultimo_df_narrativa, self._ultimas_anomalias,
            self._ultimo_nombre_dataset_narrativa, modo_impresion=True,
        )
        documento_impresion = QTextDocument()
        documento_impresion.setHtml(html_impresion)

        # QTextDocument.print_() pagina automáticamente el HTML del informe
        # (tantas páginas como haga falta) -- no requiere recortar a mano
        # como sí hace falta con el lienzo de la pestaña Reporte.
        documento_impresion.print_(printer)

        QMessageBox.information(self, "Listo", f"Narrativa exportada a:\n{path}")

    # ------------------------------------------------------------------
    # TAB REPORTE (lienzo tipo Excel+Word)
    # ------------------------------------------------------------------
    def _build_tab_reporte(self, tab):
        layout = QVBoxLayout(tab)

        barra = QHBoxLayout()
        # Antes eran 10 botones "+X" en fila, y con "Línea de Tiempo" sumado
        # ya no entraban en el ancho disponible (se veían cortados). En vez
        # de acomodarlos apretados, se juntan en un solo botón desplegable
        # -- así el espacio no vuelve a reventar el día que se agregue un
        # tipo de bloque más.
        btn_agregar = QToolButton()
        btn_agregar.setText("+ Agregar bloque")
        btn_agregar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn_agregar.setObjectName("accentButton")
        menu_agregar = QMenu(btn_agregar)
        menu_agregar.addAction("Texto", self._reporte_agregar_texto)
        menu_agregar.addAction("Gráfico", self._reporte_agregar_grafico)
        menu_agregar.addAction("Métrica", self._reporte_agregar_metrica)
        menu_agregar.addAction("Indicador", self._reporte_agregar_indicador)
        menu_agregar.addAction("Frecuencias", self._reporte_agregar_frecuencias)
        menu_agregar.addAction("Narrativa", self._reporte_agregar_narrativa)
        menu_agregar.addAction("Esquema", self._reporte_agregar_esquema)
        menu_agregar.addAction("Imagen", self._reporte_agregar_imagen)
        menu_agregar.addAction("Cálculo", self._reporte_agregar_calculo)
        menu_agregar.addAction("Línea de Tiempo", self._reporte_agregar_linea_tiempo)
        btn_agregar.setMenu(menu_agregar)
        barra.addWidget(btn_agregar)

        btn_exportar = QPushButton("Exportar a PDF")
        btn_exportar.setObjectName("accentButton")
        barra.addStretch()
        barra.addWidget(btn_exportar)
        layout.addLayout(barra)

        # Barra de hojas: cada hoja es una página independiente del reporte,
        # con su propio lienzo y su propia numeración de recuadros.
        barra_hojas = QHBoxLayout()
        self.reporte_tabbar = QTabBar()
        self.reporte_tabbar.setTabsClosable(True)
        self.reporte_tabbar.setExpanding(False)
        self.reporte_tabbar.setDrawBase(False)
        self.reporte_tabbar.currentChanged.connect(self._reporte_cambiar_hoja)
        self.reporte_tabbar.tabCloseRequested.connect(self._reporte_eliminar_hoja)
        self.reporte_tabbar.tabBarDoubleClicked.connect(self._reporte_renombrar_hoja)
        barra_hojas.addWidget(self.reporte_tabbar, stretch=1)
        btn_nueva_hoja = QPushButton("+ Hoja")
        btn_nueva_hoja.clicked.connect(lambda: self._reporte_nueva_hoja())
        barra_hojas.addWidget(btn_nueva_hoja)
        layout.addLayout(barra_hojas)

        self._reporte_texto_activo = None
        self._crear_barra_formato_texto(layout)

        self.reporte_hojas = []
        self.reporte_scene = None
        self.reporte_view = QGraphicsView()
        self.reporte_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.reporte_view.setBackgroundBrush(QBrush(QColor(self.colors["bg"])))
        layout.addWidget(self.reporte_view, stretch=1)

        btn_exportar.clicked.connect(self._reporte_exportar_pdf)

        self._reporte_nueva_hoja("Hoja 1")

    # -- barra fija de formato de texto (fuente/tamaño/negrita/rojo) --
    def _crear_barra_formato_texto(self, layout_reporte):
        """Fila de formato de texto SIEMPRE visible dentro de la pestaña
        Reporte (no flotante encima del lienzo). Antes vivía como un widget
        flotante hijo del viewport del QGraphicsView y se posicionaba a mano;
        eso generaba una carrera de foco con los QComboBox (su lista
        desplegable es una ventana aparte) que a veces impedía elegir fuente
        o tamaño. Como fila fija de la pestaña, ese problema desaparece."""
        barra = QFrame()
        barra.setObjectName("card")
        fila = QHBoxLayout(barra)
        fila.setContentsMargins(8, 6, 8, 6)
        fila.setSpacing(6)

        self.lbl_barra_formato = QLabel("Formato de texto: elige un recuadro de texto (doble clic) para activarlo.")
        self.lbl_barra_formato.setObjectName("muted")
        # OJO: un QLabel de una sola línea (sin wrap) le pide a su layout,
        # como mínimo, el ancho completo del texto -- a diferencia de un
        # QTabBar (las pestañas de arriba, que sí achican/truncan solas),
        # un QLabel normal NO cede nada por su cuenta. Como esta fila vive
        # en el mismo QVBoxLayout que "Exportar a PDF" y "+ Hoja" (filas
        # separadas, pero mismo ancho de página), ese mínimo tan largo
        # terminaba empujando TODO lo demás de la pestaña Reporte fuera
        # del ancho disponible. Ignored le dice al layout "no me uses de
        # referencia para el mínimo": el texto se puede recortar sin que
        # se lleve por delante los botones de las otras filas.
        self.lbl_barra_formato.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        fila.addWidget(self.lbl_barra_formato, stretch=1)

        fila.addWidget(QLabel("Fuente:"))
        self.combo_fuente_texto = QComboBox()
        self.combo_fuente_texto.addItems(REPORTE_FUENTES)
        fila.addWidget(self.combo_fuente_texto)

        fila.addWidget(QLabel("Tamaño:"))
        self.combo_tamano_texto = QComboBox()
        self.combo_tamano_texto.addItems([str(t) for t in REPORTE_TAMANOS])
        fila.addWidget(self.combo_tamano_texto)

        self.btn_negrita_texto = QPushButton("N")
        self.btn_negrita_texto.setCheckable(True)
        self.btn_negrita_texto.setFixedWidth(28)
        self.btn_negrita_texto.setToolTip("Negrita")
        font_negrita = self.btn_negrita_texto.font()
        font_negrita.setBold(True)
        self.btn_negrita_texto.setFont(font_negrita)
        fila.addWidget(self.btn_negrita_texto)

        self.btn_rojo_texto = QPushButton("A")
        self.btn_rojo_texto.setCheckable(True)
        self.btn_rojo_texto.setFixedWidth(28)
        self.btn_rojo_texto.setToolTip("Resaltar en rojo")
        self.btn_rojo_texto.setStyleSheet(
            f"QPushButton {{ color: {REPORTE_COLOR_ROJO}; font-weight: bold; }}"
        )
        fila.addWidget(self.btn_rojo_texto)

        self.barra_texto_reporte = barra
        self.barra_texto_reporte.setEnabled(False)
        layout_reporte.addWidget(barra)

        self.combo_fuente_texto.currentTextChanged.connect(
            lambda familia: self._reporte_aplicar_formato_texto(
                lambda fmt: self._set_fuente_familia(fmt, familia)
            )
        )
        self.combo_tamano_texto.currentTextChanged.connect(
            lambda tam: self._reporte_aplicar_formato_texto(
                lambda fmt: fmt.setFontPointSize(float(tam))
            )
        )
        self.btn_negrita_texto.toggled.connect(
            lambda activo: self._reporte_aplicar_formato_texto(
                lambda fmt: fmt.setFontWeight(QFont.Weight.Bold if activo else QFont.Weight.Normal)
            )
        )
        self.btn_rojo_texto.toggled.connect(
            lambda activo: self._reporte_aplicar_formato_texto(
                lambda fmt: fmt.setForeground(
                    QColor(REPORTE_COLOR_ROJO) if activo else QColor(REPORTE_PAPEL_TINTA)
                )
            )
        )

    @staticmethod
    def _set_fuente_familia(fmt, familia):
        if hasattr(fmt, "setFontFamilies"):
            fmt.setFontFamilies([familia])
        else:
            fmt.setFontFamily(familia)

    def _reporte_mostrar_barra_texto(self, box_item):
        """Se llama cuando el usuario entra a editar un recuadro de texto
        (doble clic): activa la barra de formato y sincroniza sus controles
        con el formato del texto en la posición actual del cursor."""
        self._reporte_texto_activo = box_item
        self.barra_texto_reporte.setEnabled(True)
        self.lbl_barra_formato.setText("Formato de texto:")

        cursor = box_item.text_item.textCursor()
        fmt = cursor.charFormat()
        fuente = fmt.font()

        familia = fuente.family() or "Arial"
        if familia not in REPORTE_FUENTES:
            familia = "Arial"
        self.combo_fuente_texto.blockSignals(True)
        self.combo_fuente_texto.setCurrentText(familia)
        self.combo_fuente_texto.blockSignals(False)

        tam = int(fuente.pointSize()) if fuente.pointSize() > 0 else 12
        if tam not in REPORTE_TAMANOS:
            tam = min(REPORTE_TAMANOS, key=lambda t: abs(t - tam))
        self.combo_tamano_texto.blockSignals(True)
        self.combo_tamano_texto.setCurrentText(str(tam))
        self.combo_tamano_texto.blockSignals(False)

        self.btn_negrita_texto.blockSignals(True)
        self.btn_negrita_texto.setChecked(fuente.bold())
        self.btn_negrita_texto.blockSignals(False)

        self.btn_rojo_texto.blockSignals(True)
        self.btn_rojo_texto.setChecked(fmt.foreground().color().name().lower() == REPORTE_COLOR_ROJO.lower())
        self.btn_rojo_texto.blockSignals(False)

    def _reporte_ocultar_barra_texto(self, box_item):
        # La barra ahora es una fila fija (no flotante), así que no hace
        # falta ocultarla: se deja habilitada, apuntando al último recuadro
        # de texto editado, para poder seguir ajustándole el formato aunque
        # el cursor ya no esté parpadeando adentro.
        pass

    def _reporte_aplicar_formato_texto(self, modificar_formato):
        """modificar_formato: función que recibe un QTextCharFormat y lo
        modifica in-place. Se aplica a la selección actual del recuadro de
        texto activo (o, si no hay selección, a lo próximo que se escriba)."""
        box_item = self._reporte_texto_activo
        if box_item is None or box_item.text_item is None:
            return
        text_item = box_item.text_item
        try:
            cursor = text_item.textCursor()
            fmt = QTextCharFormat()
            modificar_formato(fmt)
            cursor.mergeCharFormat(fmt)
        except RuntimeError:
            return  # el recuadro fue eliminado mientras tanto

        # Devolver el foco al recuadro se deja para el próximo ciclo de
        # eventos (QTimer con delay 0): hacerlo de inmediato, todavía dentro
        # del manejador de señal del combo, interfería con el cierre de su
        # lista desplegable y a veces impedía que el cambio de fuente/tamaño
        # se registrara — esa era la causa del bug de "no me deja escoger".
        QTimer.singleShot(0, lambda: self._reporte_reenfocar_texto(text_item, cursor))

    def _reporte_reenfocar_texto(self, text_item, cursor):
        try:
            text_item.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
            # Para que el recuadro reciba de verdad las teclas (y no solo se
            # vea "enfocado"), la vista que lo contiene también necesita el
            # foco real del teclado — no basta con enfocar el ítem dentro de
            # la escena. Sin esta línea, el cursor parpadeaba pero las teclas
            # se iban a ningún lado (o al combo), bloqueando la escritura.
            self.reporte_view.setFocus(Qt.FocusReason.OtherFocusReason)
            text_item.setTextCursor(cursor)
            text_item.setFocus(Qt.FocusReason.OtherFocusReason)
        except RuntimeError:
            pass  # el recuadro fue eliminado mientras tanto

    def _reporte_nueva_hoja(self, nombre=None):
        if nombre is None:
            nombre = f"Hoja {len(self.reporte_hojas) + 1}"
        scene = QGraphicsScene(0, 0, 850, 1100)
        scene._offset = 0
        pagina = QGraphicsRectItem(0, 0, 850, 1100)
        pagina.setBrush(QBrush(QColor(REPORTE_PAPEL_BG)))
        pagina.setPen(QPen(QColor(REPORTE_PAPEL_BORDE), 1))
        pagina.setZValue(-1000)
        scene.addItem(pagina)
        self.reporte_hojas.append({"nombre": nombre, "scene": scene, "pagina": pagina})

        self.reporte_tabbar.blockSignals(True)
        idx = self.reporte_tabbar.addTab(nombre)
        self.reporte_tabbar.blockSignals(False)
        self.reporte_tabbar.setCurrentIndex(idx)
        self._reporte_cambiar_hoja(idx)

    def _reporte_cambiar_hoja(self, index):
        if index < 0 or index >= len(self.reporte_hojas):
            return
        self.reporte_scene = self.reporte_hojas[index]["scene"]
        self.reporte_view.setScene(self.reporte_scene)

    def _reporte_renombrar_hoja(self, index):
        if index < 0 or index >= len(self.reporte_hojas):
            return
        actual = self.reporte_hojas[index]["nombre"]
        nuevo, ok = QInputDialog.getText(
            self, "Renombrar hoja", "Nombre de la hoja:", QLineEdit.Normal, actual
        )
        if not ok or not nuevo.strip():
            return
        nuevo = nuevo.strip()
        self.reporte_hojas[index]["nombre"] = nuevo
        self.reporte_tabbar.setTabText(index, nuevo)

    def _reporte_eliminar_hoja(self, index):
        if index < 0 or index >= len(self.reporte_hojas):
            return
        if len(self.reporte_hojas) <= 1:
            QMessageBox.information(
                self, "No se puede eliminar", "Debe quedar al menos una hoja en el reporte."
            )
            return
        resp = QMessageBox.question(
            self, "Eliminar hoja",
            f"¿Eliminar \"{self.reporte_hojas[index]['nombre']}\" y todo su contenido?"
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        del self.reporte_hojas[index]
        self.reporte_tabbar.blockSignals(True)
        self.reporte_tabbar.removeTab(index)
        self.reporte_tabbar.blockSignals(False)
        nuevo_index = max(0, self.reporte_tabbar.currentIndex())
        self._reporte_cambiar_hoja(nuevo_index)

    def _reporte_siguiente_posicion(self, ancho, alto):
        """Posiciona cada recuadro nuevo en cascada dentro de la hoja activa,
        para que no queden todos exactamente encima del anterior."""
        escena = self.reporte_scene
        offset = getattr(escena, "_offset", 0)
        x = 40 + (offset % 6) * 25
        y = 40 + (offset % 6) * 25
        escena._offset = offset + 1
        return QRectF(x, y, ancho, alto)

    def _reporte_agregar_texto(self):
        rect = self._reporte_siguiente_posicion(220, 100)
        item = ReportBoxItem("texto", rect, self.colors, ventana_principal=self)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_anomalia_desde_linea_tiempo(self, anomalia):
        rect = self._reporte_siguiente_posicion(280, 140)
        item = ReportBoxItem("lista", rect, self.colors, ventana_principal=self)
        cuerpo = anomalia.get("descripcion", "")
        frase_memoria = anomalia.get("frase_memoria")
        if frase_memoria:
            cuerpo += f"<br><br>{frase_memoria}"
        item.set_lista_text("Hallazgo — Línea de Tiempo", cuerpo)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_metricas_tiempo(self, titulo, html):
        # Se llama desde el botón "Enviar a Reporte" del diálogo "Ver
        # métricas de tiempo" de la Línea de Tiempo -- son solo números y
        # texto, así que va como bloque 'lista', igual que Frecuencias y
        # que los hallazgos de anomalías, no como imagen.
        rect = self._reporte_siguiente_posicion(300, 160)
        item = ReportBoxItem("lista", rect, self.colors, ventana_principal=self)
        item.set_lista_text(titulo, html)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_linea_tiempo(self):
        if not hasattr(self, "panel_linea_tiempo") or self.df is None or self.df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        panel = self.panel_linea_tiempo

        def regenerar(panel=panel):
            return panel.snapshot()

        rect = self._reporte_siguiente_posicion(360, 300)
        item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self, titulo=panel.nombre())
        item.regenerar = regenerar
        item.set_pixmap(regenerar())
        self.reporte_scene.addItem(item)

    def _reporte_agregar_grafico(self):
        if not self.chart_panels:
            QMessageBox.information(
                self, "Sin gráficos",
                "Primero genera al menos un gráfico en la pestaña Gráficos."
            )
            return
        opciones = [f"{i + 1}. {p.etiqueta()}" for i, p in enumerate(self.chart_panels)]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir gráfico", "¿Cuál gráfico quieres llevar al reporte?", opciones, 0, False
        )
        if not ok or not elegido:
            return
        idx = opciones.index(elegido)
        panel = self.chart_panels[idx]
        titulo = panel.nombre() or None

        def regenerar(panel=panel):
            return panel.snapshot()

        rect = self._reporte_siguiente_posicion(340, 240)
        item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self, titulo=titulo)
        item.regenerar = regenerar
        item.set_pixmap(regenerar())
        self.reporte_scene.addItem(item)

    def _reporte_agregar_indicador(self):
        if not self.indicadores:
            QMessageBox.information(
                self, "Sin indicadores",
                "Todavía no has creado ningún indicador. Ve a la pestaña Indicadores para crear uno."
            )
            return
        nombres = [ind.nombre for ind in self.indicadores]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir indicador", "¿Qué indicador quieres llevar al reporte?", nombres, 0, False
        )
        if not ok or not elegido:
            return
        indicador = next((ind for ind in self.indicadores if ind.nombre == elegido), None)
        if indicador is None:
            return

        def calcular(indicador=indicador):
            df_actual = self.filtered_df if self.filtered_df is not None else self.df
            valor = indicador.calcular(df_actual)
            valor_crudo = float(valor) if isinstance(valor, (int, float)) and not pd.isna(valor) else None
            return indicador.nombre, indicador.formatear(valor), valor_crudo

        rect = self._reporte_siguiente_posicion(200, 110)
        item = ReportBoxItem("calculo", rect, self.colors, ventana_principal=self)
        item.regenerar = calcular
        titulo, valor_str, valor_crudo = calcular()
        item.set_calculo_text(titulo, valor_str, valor_crudo)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_narrativa(self):
        if not hasattr(self, "narrativa_browser"):
            return
        if not getattr(self, "_narrativa_actualizada", False):
            df = self.filtered_df if self.filtered_df is not None else self.df
            if df is None or df.empty:
                QMessageBox.information(self, "Sin datos", "Carga datos primero.")
                return
            self._generar_narrativa()

        opciones = [
            "Informe completo (foto)",
            "Preguntas sugeridas (lista)",
        ]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir tipo", "¿Qué quieres traer de Narrativa?", opciones, 0, False
        )
        if not ok or not elegido:
            return

        if elegido == "Informe completo (foto)":
            def regenerar():
                if not getattr(self, "_narrativa_actualizada", False):
                    self._generar_narrativa()
                self.narrativa_browser.repaint()
                QApplication.processEvents()
                return self.narrativa_browser.grab()

            rect = self._reporte_siguiente_posicion(360, 260)
            item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self)
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)
        else:
            def calcular():
                df_actual = self.filtered_df if self.filtered_df is not None else self.df
                titulo = "Preguntas sugeridas"
                if df_actual is None or df_actual.empty:
                    return titulo, "-"
                preguntas = QuestionAssistant(df_actual, self._ultimas_anomalias).suggest()
                if not preguntas:
                    return titulo, "Sin preguntas sugeridas por ahora."
                cuerpo = "<br>".join(f"{i + 1}. {p}" for i, p in enumerate(preguntas))
                return titulo, cuerpo

            rect = self._reporte_siguiente_posicion(280, 220)
            item = ReportBoxItem("lista", rect, self.colors, ventana_principal=self)
            item.regenerar = calcular
            titulo, cuerpo = calcular()
            item.set_lista_text(titulo, cuerpo)
            self.reporte_scene.addItem(item)

    def _reporte_agregar_esquema(self):
        """Trae el esquema sugerido/confirmado entre tus tablas como una
        foto -- útil para dejar constancia visual de cómo está armada la
        base relacional dentro del informe final."""
        if len(self.tablas) < 2:
            QMessageBox.information(
                self, "Sin esquema para mostrar",
                "Carga 2 o más tablas relacionadas (ver botón \"Ver esquema\" en "
                "Datos) para poder agregar el esquema al reporte."
            )
            return

        def regenerar():
            dialogo_temp = DialogoEsquemaOntologia(self.tablas, self.colors,
                                                    relaciones_iniciales=self.relaciones_ontologia)
            return dialogo_temp.render_a_pixmap()

        rect = self._reporte_siguiente_posicion(420, 260)
        item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self, titulo="Esquema de tus tablas")
        item.regenerar = regenerar
        item.set_pixmap(regenerar())
        self.reporte_scene.addItem(item)

    def _reporte_agregar_imagen(self):
        """Sube una imagen desde el disco (ej. una captura de pantalla que
        tomaste tú mismo del mapa) y la deja como recuadro fijo — sin botón
        de actualizar, porque no está ligada a ningún dato de la app."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Elegir imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "No se pudo abrir", "Ese archivo no se pudo cargar como imagen.")
            return

        rect = self._reporte_siguiente_posicion(320, 220)
        item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self, con_refresh=False)
        item.set_pixmap(pixmap)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_metrica(self):
        """Trae una de las 4 estadísticas -- o el gráfico de caja y bigote --
        de uno de los bloques de la pestaña Métricas."""
        if not self.metricas_bloques:
            QMessageBox.information(
                self, "Sin métricas", "Agrega al menos un bloque de métricas primero."
            )
            return

        columnas_disponibles = [b["columna"] for b in self.metricas_bloques]
        columna, ok = QInputDialog.getItem(
            self, "Elegir columna", "¿De qué columna quieres traer una métrica?",
            columnas_disponibles, 0, False
        )
        if not ok or not columna:
            return

        opciones = ["Media", "Mediana", "Moda", "Desviación Std", "Gráfico de caja y bigote (foto)"]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir métrica", "¿Qué quieres traer al reporte?", opciones, 0, False
        )
        if not ok or not elegido:
            return

        if elegido == "Gráfico de caja y bigote (foto)":
            bloque = next((b for b in self.metricas_bloques if b["columna"] == columna), None)
            if bloque is None:
                return

            def regenerar(plot=bloque["plot"]):
                try:
                    import pyqtgraph.exporters as pg_exporters
                    exportador = pg_exporters.ImageExporter(plot.getPlotItem())
                    imagen = exportador.export(toBytes=True)
                    pixmap = QPixmap.fromImage(imagen)
                    if not pixmap.isNull():
                        return pixmap
                except Exception:
                    pass
                return plot.grab()

            rect = self._reporte_siguiente_posicion(340, 240)
            item = ReportBoxItem(
                "imagen", rect, self.colors, ventana_principal=self, titulo=f"Caja y bigote — {columna}"
            )
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)
            return

        def calcular(elegido=elegido, columna=columna):
            df_actual = self.filtered_df
            if df_actual is None or df_actual.empty or columna not in df_actual.columns:
                return f"{elegido} de {columna}", "-", None
            serie = pd.to_numeric(df_actual[columna], errors="coerce").dropna()
            if serie.empty:
                return f"{elegido} de {columna}", "-", None
            if elegido == "Media":
                valor = serie.mean()
            elif elegido == "Mediana":
                valor = serie.median()
            elif elegido == "Moda":
                moda = serie.mode()
                valor = moda.iloc[0] if not moda.empty else None
            else:
                valor = serie.std()
            if valor is None or pd.isna(valor):
                return f"{elegido} de {columna}", "-", None
            return f"{elegido} de {columna}", f"{valor:,.2f}", float(valor)

        rect = self._reporte_siguiente_posicion(200, 110)
        item = ReportBoxItem("calculo", rect, self.colors, ventana_principal=self)
        item.regenerar = calcular
        titulo, valor_str, valor_crudo = calcular()
        item.set_calculo_text(titulo, valor_str, valor_crudo)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_frecuencias(self):
        opciones = [
            "Tabla de frecuencias (foto)",
            "Gráfico de frecuencias (foto)",
            "Top 10 más repetidos (lista)",
        ]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir tipo", "¿Qué quieres traer de Frecuencias?", opciones, 0, False
        )
        if not ok or not elegido:
            return

        if elegido == "Tabla de frecuencias (foto)":
            def regenerar():
                self.freq_table_view.repaint()
                return self.freq_table_view.grab()

            rect = self._reporte_siguiente_posicion(340, 240)
            item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self)
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)

        elif elegido == "Gráfico de frecuencias (foto)":
            def regenerar():
                try:
                    import pyqtgraph.exporters as pg_exporters
                    exportador = pg_exporters.ImageExporter(self.freq_plot.getPlotItem())
                    imagen = exportador.export(toBytes=True)
                    pixmap = QPixmap.fromImage(imagen)
                    if not pixmap.isNull():
                        return pixmap
                except Exception:
                    pass
                return self.freq_plot.grab()

            rect = self._reporte_siguiente_posicion(340, 240)
            item = ReportBoxItem("imagen", rect, self.colors, ventana_principal=self)
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)

        else:  # Top 10 más repetidos
            if self.filtered_df is None or self.filtered_df.empty:
                QMessageBox.information(self, "Sin datos", "Carga datos primero.")
                return
            columnas = list(self.filtered_df.columns)
            columna, ok = QInputDialog.getItem(
                self, "Elegir columna", "¿De qué columna quieres el top 10?", columnas, 0, False
            )
            if not ok or not columna:
                return

            def calcular(columna=columna):
                df_actual = self.filtered_df
                titulo = f"Top 10 más repetidos — {columna}"
                if df_actual is None or columna not in df_actual.columns:
                    return titulo, "-"
                serie = df_actual[columna].dropna().astype(str)
                if serie.empty:
                    return titulo, "-"
                top = serie.value_counts().head(10)
                cuerpo = "<br>".join(f"{i + 1}. {val} ({freq})" for i, (val, freq) in enumerate(top.items()))
                return titulo, cuerpo

            rect = self._reporte_siguiente_posicion(240, 220)
            item = ReportBoxItem("lista", rect, self.colors, ventana_principal=self)
            item.regenerar = calcular
            titulo, cuerpo = calcular()
            item.set_lista_text(titulo, cuerpo)
            self.reporte_scene.addItem(item)

    def _reporte_agregar_calculo(self):
        if self.filtered_df is None or self.filtered_df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        origen, ok = QInputDialog.getItem(
            self, "Tipo de cálculo", "¿Qué quieres calcular?",
            [
                "Una operación nueva sobre una columna",
                "Combinar recuadros ya creados (Métrica/Indicador/Cálculo)",
            ], 0, False
        )
        if not ok or not origen:
            return

        if origen.startswith("Combinar"):
            self._reporte_agregar_calculo_combinado()
        else:
            self._reporte_agregar_calculo_columna()

    def _reporte_agregar_calculo_combinado(self):
        """Combina 2 valores con una operación básica — por ejemplo,
        Ganancia = Ingresos − Costos. Las fuentes disponibles son: cualquier
        Indicador ya creado (pestaña Indicadores), cualquier Métrica de la
        pestaña Métricas, o un recuadro de Cálculo que ya hayas puesto en
        esta hoja del Reporte. No hace falta ponerlos primero en el lienzo."""
        fuentes = []  # cada fuente: (etiqueta para elegir, nombre corto, función sin args -> valor o None)

        for ind in self.indicadores:
            fuentes.append((
                f"Indicador: {ind.nombre}", ind.nombre,
                lambda ind=ind: ind.calcular(self.filtered_df if self.filtered_df is not None else self.df)
            ))

        for bloque in getattr(self, "metricas_bloques", []):
            col_metrica = bloque["columna"]
            for nombre_metrica in ["Media", "Mediana", "Moda", "Desviación Std"]:
                fuentes.append((
                    f"Métrica: {nombre_metrica} de {col_metrica}",
                    f"{nombre_metrica} de {col_metrica}",
                    lambda nm=nombre_metrica, col=col_metrica: self._calcular_valor_metrica(nm, col)
                ))

        candidatos_lienzo = [
            it for it in self.reporte_scene.items()
            if isinstance(it, ReportBoxItem) and it.kind == "calculo"
        ]
        for it in candidatos_lienzo:
            fuentes.append((
                f"En el reporte: {it.titulo_calculo or 'Sin título'}",
                it.titulo_calculo or "Sin título",
                (lambda it=it: it.valor_crudo)
            ))

        if len(fuentes) < 2:
            QMessageBox.information(
                self, "Faltan datos para combinar",
                "Necesitas al menos 2 valores disponibles para combinar: crea un indicador en la "
                "pestaña Indicadores, elige una columna en la pestaña Métricas, o agrega un recuadro "
                "de Cálculo en esta hoja primero."
            )
            return

        etiquetas = [e for e, _, _ in fuentes]

        etiqueta_a, ok = QInputDialog.getItem(self, "Primer valor", "Elige el primer valor:", etiquetas, 0, False)
        if not ok or not etiqueta_a:
            return
        etiqueta_b, ok = QInputDialog.getItem(self, "Segundo valor", "Elige el segundo valor:", etiquetas, 0, False)
        if not ok or not etiqueta_b:
            return

        _, nombre_a, funcion_a = fuentes[etiquetas.index(etiqueta_a)]
        _, nombre_b, funcion_b = fuentes[etiquetas.index(etiqueta_b)]

        operacion, ok = QInputDialog.getItem(
            self, "Elegir operación", "¿Cómo quieres combinarlos?",
            ["Suma", "Resta (primero − segundo)", "Multiplicación", "División (primero ÷ segundo)"], 0, False
        )
        if not ok or not operacion:
            return

        formato, ok = QInputDialog.getText(
            self, "Formato de despliegue",
            "Formato del número (opcional, ej: {value:,.2f} o ${value:,.0f}):",
            QLineEdit.Normal, "{value:,.2f}"
        )
        formato = (formato.strip() if ok and formato and formato.strip() else "{value:,.2f}")

        def calcular(funcion_a=funcion_a, funcion_b=funcion_b, nombre_a=nombre_a, nombre_b=nombre_b,
                     operacion=operacion, formato=formato):
            try:
                valor_a = funcion_a()
                valor_b = funcion_b()
            except RuntimeError:
                return "Combinado", "-", None
            if (
                valor_a is None or valor_b is None
                or (isinstance(valor_a, float) and pd.isna(valor_a))
                or (isinstance(valor_b, float) and pd.isna(valor_b))
            ):
                return f"{nombre_a} / {nombre_b}", "-", None
            if operacion == "Suma":
                valor, simbolo = valor_a + valor_b, "+"
            elif operacion.startswith("Resta"):
                valor, simbolo = valor_a - valor_b, "−"
            elif operacion == "Multiplicación":
                valor, simbolo = valor_a * valor_b, "×"
            else:
                valor = (valor_a / valor_b) if valor_b != 0 else float("nan")
                simbolo = "÷"
            titulo = f"{nombre_a} {simbolo} {nombre_b}"
            if valor is None or (isinstance(valor, float) and pd.isna(valor)):
                return titulo, "-", None
            try:
                valor_str = formato.format(value=valor)
            except Exception:
                valor_str = f"{valor:,.2f}"
            return titulo, valor_str, float(valor)

        rect = self._reporte_siguiente_posicion(220, 110)
        item = ReportBoxItem("calculo", rect, self.colors, ventana_principal=self)
        item.regenerar = calcular
        titulo, valor_str, valor_crudo = calcular()
        item.set_calculo_text(titulo, valor_str, valor_crudo)
        self.reporte_scene.addItem(item)

    def _calcular_valor_metrica(self, nombre_metrica, columna):
        """Misma lógica que usa '+ Métrica' del Reporte, factorizada aparte
        para poder reutilizarla también al combinar valores en '+ Cálculo'."""
        df_actual = self.filtered_df
        if df_actual is None or df_actual.empty or not columna or columna not in df_actual.columns:
            return None
        serie = pd.to_numeric(df_actual[columna], errors="coerce").dropna()
        if serie.empty:
            return None
        if nombre_metrica == "Media":
            return serie.mean()
        if nombre_metrica == "Mediana":
            return serie.median()
        if nombre_metrica == "Moda":
            moda = serie.mode()
            return moda.iloc[0] if not moda.empty else None
        if nombre_metrica == "Desviación Std":
            return serie.std()
        return None

    def _reporte_agregar_calculo_columna(self):
        if self.filtered_df is None or self.filtered_df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        columnas = list(self.filtered_df.columns)
        columna_a, ok = QInputDialog.getItem(
            self, "Elegir columna", "¿Sobre qué columna quieres calcular?", columnas, 0, False
        )
        if not ok or not columna_a:
            return

        operaciones = [
            "Suma", "Promedio", "Mínimo", "Máximo", "Conteo (no nulos)",
            "Resta (esta columna − otra)",
            "Multiplicación (esta columna × otra)",
            "División (esta columna ÷ otra)",
        ]
        operacion, ok = QInputDialog.getItem(
            self, "Elegir operación", "¿Qué quieres calcular?", operaciones, 0, False
        )
        if not ok or not operacion:
            return

        columna_b = None
        if operacion.startswith(("Resta", "Multiplicación", "División")):
            columna_b, ok = QInputDialog.getItem(
                self, "Elegir segunda columna", "¿Contra qué columna la comparas?", columnas, 0, False
            )
            if not ok or not columna_b:
                return

        def calcular(columna_a=columna_a, columna_b=columna_b, operacion=operacion):
            df_actual = self.filtered_df
            if df_actual is None or columna_a not in df_actual.columns:
                return operacion, "-", None

            if operacion == "Conteo (no nulos)":
                conteo = int(df_actual[columna_a].notna().sum())
                return f"{operacion} de {columna_a}", f"{conteo:,}", float(conteo)

            if columna_b is not None:
                if columna_b not in df_actual.columns:
                    return operacion, "-", None
                suma_a = pd.to_numeric(df_actual[columna_a], errors="coerce").sum()
                suma_b = pd.to_numeric(df_actual[columna_b], errors="coerce").sum()
                if operacion.startswith("Resta"):
                    valor, titulo = suma_a - suma_b, f"{columna_a} − {columna_b}"
                elif operacion.startswith("Multiplicación"):
                    valor, titulo = suma_a * suma_b, f"{columna_a} × {columna_b}"
                else:
                    valor = (suma_a / suma_b) if suma_b != 0 else float("nan")
                    titulo = f"{columna_a} ÷ {columna_b}"
                if pd.isna(valor):
                    return titulo, "-", None
                valor_str = f"{valor:,.0f}" if float(valor).is_integer() else f"{valor:,.2f}"
                return titulo, valor_str, float(valor)

            numerica = pd.to_numeric(df_actual[columna_a], errors="coerce").dropna()
            if numerica.empty:
                return f"{operacion} de {columna_a}", "-", None
            if operacion == "Suma":
                valor = numerica.sum()
            elif operacion == "Promedio":
                valor = numerica.mean()
            elif operacion == "Mínimo":
                valor = numerica.min()
            else:
                valor = numerica.max()
            valor_str = f"{valor:,.0f}" if float(valor).is_integer() else f"{valor:,.2f}"
            return f"{operacion} de {columna_a}", valor_str, float(valor)

        rect = self._reporte_siguiente_posicion(200, 110)
        item = ReportBoxItem("calculo", rect, self.colors, ventana_principal=self)
        item.regenerar = calcular
        titulo, valor_str, valor_crudo = calcular()
        item.set_calculo_text(titulo, valor_str, valor_crudo)
        self.reporte_scene.addItem(item)

    def _reporte_exportar_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Reporte a PDF", "reporte.pdf", "PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))

        painter = QPainter(printer)
        for i, hoja in enumerate(self.reporte_hojas):
            if i > 0:
                printer.newPage()
            hoja["scene"].render(painter)
        painter.end()

        n_hojas = len(self.reporte_hojas)
        QMessageBox.information(
            self, "Listo",
            f"Reporte exportado a:\n{path}\n({n_hojas} hoja{'s' if n_hojas != 1 else ''})"
        )

    def _actualizar_resumen_anomalias_frecuencias(self):
        """Cards 'Anomalía Más Frecuente' y 'Columna Más Afectada' de la
        pestaña Frecuencias. No depende de qué columna(s) esté mirando el
        usuario en freq_col_list ni del filtro de Datos -- lee directo
        self._ultimas_anomalias, calculado por Narrativa la última vez que
        se generó el informe (igual que hace Indicadores con
        columnas_con_anomalias)."""
        if not hasattr(self, "lbl_freq_anomalia_top"):
            return

        anomalias = getattr(self, "_ultimas_anomalias", [])
        if not anomalias:
            self.lbl_freq_anomalia_top.setText("Genera Narrativa primero")
            self.lbl_freq_anomalia_col.setText("Genera Narrativa primero")
            return

        # Se suma 'filas_afectadas' (no la cantidad de anomalías) para que
        # un solo tipo con miles de filas pese más que diez anomalías de
        # una sola fila cada una -- "más frecuente" se refiere a cuántas
        # filas toca, no a cuántas entradas separadas hay en el informe.
        filas_por_tipo = {}
        filas_por_columna = {}
        for a in anomalias:
            n = a.get("filas_afectadas") or 0
            tipo = a.get("tipo")
            if tipo:
                filas_por_tipo[tipo] = filas_por_tipo.get(tipo, 0) + n
            for col in a.get("columnas", []):
                filas_por_columna[col] = filas_por_columna.get(col, 0) + n

        if filas_por_tipo:
            tipo_top, n_tipo = max(filas_por_tipo.items(), key=lambda kv: kv[1])
            etiqueta = ETIQUETAS_TIPO_ANOMALIA.get(tipo_top, tipo_top)
            self.lbl_freq_anomalia_top.setText(f"{etiqueta} ({n_tipo:,})")
        else:
            self.lbl_freq_anomalia_top.setText("-")

        if filas_por_columna:
            col_top, n_col = max(filas_por_columna.items(), key=lambda kv: kv[1])
            self.lbl_freq_anomalia_col.setText(f"{col_top} ({n_col:,})")
        else:
            self.lbl_freq_anomalia_col.setText("-")

    def _update_frecuencias(self):
        if not hasattr(self, "freq_table_model"):
            return

        self._actualizar_resumen_anomalias_frecuencias()

        if self.filtered_df is None or self.filtered_df.empty:
            self.freq_table_model.set_dataframe(pd.DataFrame())
            self.lbl_freq_moda.setText("-")
            self.lbl_freq_menos.setText("-")
            self.lbl_freq_unicos.setText("-")
            self.freq_plot.clear()
            return

        selected_cols = [item.text() for item in self.freq_col_list.selectedItems()]
        if not selected_cols:
            self.freq_table_model.set_dataframe(pd.DataFrame())
            self.lbl_freq_moda.setText("-")
            self.lbl_freq_menos.setText("-")
            self.lbl_freq_unicos.setText("-")
            self.freq_plot.clear()
            return

        rows = []
        first_col_counts = None
        for col in selected_cols:
            if col not in self.filtered_df.columns:
                continue
            serie_cruda = self.filtered_df[col]
            total = len(serie_cruda)
            if total == 0:
                continue
            n_vacios = int(serie_cruda.isna().sum())
            counts = serie_cruda.dropna().astype(str).value_counts()
            if n_vacios:
                counts[VALOR_VACIO_FILTRO] = n_vacios
            if first_col_counts is None:
                first_col_counts = counts
            for valor, freq in counts.items():
                pct = (freq / total) * 100
                rows.append({
                    "Columna": col,
                    "Valor": valor,
                    "Frecuencia": int(freq),
                    "% del Total": f"{pct:.2f}%",
                })

        freq_df = pd.DataFrame(rows)
        if not freq_df.empty:
            freq_df = freq_df.sort_values(["Columna", "Frecuencia"], ascending=[True, False]).reset_index(drop=True)
        self.freq_table_model.set_dataframe(freq_df)

        if not freq_df.empty:
            idx_max = freq_df["Frecuencia"].idxmax()
            idx_min = freq_df["Frecuencia"].idxmin()
            moda_row = freq_df.loc[idx_max]
            menos_row = freq_df.loc[idx_min]
            self.lbl_freq_moda.setText(f"{moda_row['Valor']} ({moda_row['Frecuencia']})")
            self.lbl_freq_menos.setText(f"{menos_row['Valor']} ({menos_row['Frecuencia']})")
            self.lbl_freq_unicos.setText(str(freq_df["Valor"].nunique()))
        else:
            self.lbl_freq_moda.setText("-")
            self.lbl_freq_menos.setText("-")
            self.lbl_freq_unicos.setText("-")

        self.freq_plot.clear()
        self.freq_plot.setBackground(self.colors["card"])
        self.freq_plot.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        self.freq_plot.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        self.freq_plot.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        self.freq_plot.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        if first_col_counts is not None and not first_col_counts.empty:
            top_counts = first_col_counts.sort_values(ascending=False).head(20)
            labels = top_counts.index.astype(str).tolist()
            values = top_counts.values.tolist()
            xvals = list(range(len(labels)))
            bar = pg.BarGraphItem(x=xvals, height=values, width=0.6, brush=COLOR_ACCENT)
            self.freq_plot.addItem(bar)
            step = max(1, len(labels) // 15)
            ticks = [(i, str(lbl)[:14]) for i, lbl in enumerate(labels) if i % step == 0]
            self.freq_plot.getAxis("bottom").setTicks([ticks])

    # ------------------------------------------------------------------
    # TAB ALARMA
    # ------------------------------------------------------------------
    def _build_tab_alarma(self, tab):
        layout = QVBoxLayout(tab)

        toolbar = QHBoxLayout()
        btn_nueva = QPushButton("+ Nueva Alarma")
        btn_nueva.setObjectName("accentButton")
        btn_nueva.clicked.connect(self._abrir_dialogo_nueva_alarma)
        toolbar.addWidget(btn_nueva)
        btn_config_correo = QPushButton("⚙ Configurar avisos por correo")
        btn_config_correo.clicked.connect(self._abrir_configuracion_correo)
        toolbar.addWidget(btn_config_correo)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        aviso = QLabel(
            "Define un umbral sobre un Indicador, una Métrica o una Frecuencia. "
            "Si el valor actual lo supera, se marca en rojo acá y en su pestaña de "
            "origen (Indicadores, o Métricas/Frecuencias si esa es la columna que "
            "tenés seleccionada ahí), y aparece la nota de aviso."
        )
        aviso.setObjectName("muted")
        aviso.setWordWrap(True)
        layout.addWidget(aviso)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contenedor = QWidget()
        self.alarma_grid = QGridLayout(contenedor)
        scroll.setWidget(contenedor)
        layout.addWidget(scroll, stretch=1)

        self.alarma_widgets = []
        self._actualizar_alarmas()

    def _columnas_para_alarma(self):
        if self.df is None:
            return [], []
        numericas = [
            str(c) for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])
        ]
        todas = [str(c) for c in self.df.columns]
        return numericas, todas

    def _abrir_dialogo_nueva_alarma(self):
        if self.df is None:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        numericas, todas = self._columnas_para_alarma()
        dialogo = DialogoAlarma(
            [i.nombre for i in self.indicadores], numericas, todas, parent=self
        )
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            nuevo_id = self.alarma_memoria.agregar_regla(**dialogo.get_datos())
            self.reglas_alarma = self.alarma_memoria.listar_reglas()
            # La tratamos como "recién configurada": si ya nace en rojo,
            # que mande el correo ahora, no que espere a un cambio futuro.
            self.estado_previo_alarmas.pop(nuevo_id, None)
            self._actualizar_alarmas()

    def _editar_alarma(self, regla):
        numericas, todas = self._columnas_para_alarma()
        dialogo = DialogoAlarma(
            [i.nombre for i in self.indicadores], numericas, todas, regla=regla, parent=self
        )
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            self.alarma_memoria.actualizar_regla(regla["id"], **dialogo.get_datos())
            self.reglas_alarma = self.alarma_memoria.listar_reglas()
            # Si la edición fue justo para activarle el aviso por correo
            # mientras ya estaba en rojo, que lo mande ahora en vez de
            # esperar a que baje y vuelva a subir.
            self.estado_previo_alarmas.pop(regla["id"], None)
            self._actualizar_alarmas()
            self._actualizar_indicadores()

    def _eliminar_alarma(self, regla):
        self.alarma_memoria.eliminar_regla(regla["id"])
        self.reglas_alarma = self.alarma_memoria.listar_reglas()
        self._actualizar_alarmas()
        self._actualizar_indicadores()

    def _abrir_configuracion_correo(self):
        dialogo = DialogoConfiguracionCorreo(host=self, parent=self)
        dialogo.exec()

    def _actualizar_alarmas(self):
        """Refresca la pestaña Alarma completa, marca en rojo las tarjetas
        de Métricas/Frecuencias que correspondan (las de Indicadores se
        marcan solas en _actualizar_indicadores), y manda un correo por
        cada regla que ACABA de pasar de verde a rojo (no reenvía mientras
        se mantenga en rojo, solo en el momento en que se dispara)."""
        if not hasattr(self, "alarma_grid"):
            return
        for w in self.alarma_widgets:
            w.setParent(None)
            w.deleteLater()
        self.alarma_widgets.clear()

        df = self.filtered_df if self.filtered_df is not None else self.df

        for i, regla in enumerate(self.reglas_alarma):
            valor = calcular_valor_actual(regla, df, self.indicadores)
            en_alarma = evaluar_regla(regla, valor)
            card = AlarmaCard(regla, valor, en_alarma, self._editar_alarma, self._eliminar_alarma)
            self.alarma_grid.addWidget(card, i // 3, i % 3)
            self.alarma_widgets.append(card)

            estaba_en_alarma = self.estado_previo_alarmas.get(regla["id"], False)
            if en_alarma and not estaba_en_alarma and regla.get("notificar_correo"):
                config = cargar_configuracion()
                destino = regla.get("correo_destino") or config.get("correo_destino_predeterminado")
                if not destino:
                    self.statusBar().showMessage(
                        f"⚠ Alarma «{regla['nombre']}»: no se mandó correo, falta un correo de destino.", 8000
                    )
                else:
                    def _al_terminar_envio(ok, error, nombre_regla=regla["nombre"]):
                        if ok:
                            self.statusBar().showMessage(f"📧 Correo de alarma «{nombre_regla}» enviado.", 5000)
                        else:
                            self.statusBar().showMessage(
                                f"⚠ No se pudo mandar el correo de «{nombre_regla}»: {error}", 10000
                            )
                    disparar_envio_correo(self, destino, regla, valor, on_resultado=_al_terminar_envio)
            self.estado_previo_alarmas[regla["id"]] = en_alarma

        self._marcar_alarmas_en_metricas(df)
        self._marcar_alarmas_en_frecuencias(df)

    def _marcar_alarmas_en_metricas(self, df):
        if not hasattr(self, "metricas_bloques"):
            return
        for bloque in self.metricas_bloques:
            columna = bloque["columna"]
            estado = {nombre: False for nombre in bloque["cards"]}
            for regla in self.reglas_alarma:
                if regla["tipo"] != "Métrica" or regla["objetivo"] != columna:
                    continue
                valor = calcular_valor_actual(regla, df, self.indicadores)
                if evaluar_regla(regla, valor) and regla["estadistico"] in estado:
                    estado[regla["estadistico"]] = True
            for nombre, card in bloque["cards"].items():
                set_card_alarma(card, estado[nombre])

    def _marcar_alarmas_en_frecuencias(self, df):
        if not hasattr(self, "card_freq_moda"):
            return
        seleccionadas = (
            [item.text() for item in self.freq_col_list.selectedItems()]
            if hasattr(self, "freq_col_list") else []
        )
        en_alarma = False
        if len(seleccionadas) == 1:
            columna = seleccionadas[0]
            for regla in self.reglas_alarma:
                if regla["tipo"] != "Frecuencia" or regla["objetivo"] != columna:
                    continue
                valor = calcular_valor_actual(regla, df, self.indicadores)
                if evaluar_regla(regla, valor):
                    en_alarma = True
                    break
        set_card_alarma(self.card_freq_moda, en_alarma)