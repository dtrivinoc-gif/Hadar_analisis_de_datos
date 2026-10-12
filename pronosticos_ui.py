"""
pronosticos_ui.py — Pantalla de las «palabras clave» de la pestaña Predicción.

La lógica de cálculo vive en pronosticos.py / series_tiempo.py / eventos.py (sin Qt).
Aquí solo se dibuja: el usuario elige una pregunta («¿cuándo se acaba el stock?»),
Hadar muestra el formulario justo con los campos que esa pregunta necesita,
y el resultado aparece como: titular con rango, tres cifras, explicación,
gráfico, tabla y avisos.

Misma interfaz que PanelPrediccion y PanelControl, para que main_window la trate igual:
    aplicar_tema(colors) · notificar_cambio() · config_a_dict() · cargar_config(datos)

`host` es la ventana principal: de ahí se leen tablas, nombre_tabla_activa y colors.
La configuración se guarda dentro del .hadarproy (ver main_window._prediccion_a_dict).
El resultado NO se guarda: se recalcula con un clic.
"""
import html
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView, QHeaderView,
)

from .panel import estilo_tarjeta
from .prediccion import tipo_columna
from .ajustes import aplicar_ajustes
from .historial import limpiar as limpiar_historial, registrar_y_comparar
from .pronostico_base import fecha_txt, num_txt
from .pronosticos import (
    ConfigPronostico, ORDEN_PALABRAS, PALABRAS_CLAVE, ejecutar, sugerir_configuracion,
)
from .pronosticos_pdf import ContextoPregunta, exportar_pdf_pregunta

try:
    import pyqtgraph as pg
except Exception:  # sin pyqtgraph se omite solo el gráfico
    pg = None

# Mismo estilo "Tufte" de PanelPrediccion: poca tinta, un solo color de énfasis.
ACENTO = "#6366f1"
VERDE = "#34d399"
AMBAR = "#fbbf24"
ROJO = "#f87171"
GRIS = "#71717a"
FUENTE_SERIF = '"Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif'
REGLA = "rgba(127,127,127,0.30)"
COLOR_NIVEL = {"info": ACENTO, "atencion": AMBAR, "alerta": ROJO}
ROTULO_NIVEL = {"info": "", "atencion": "Atención · ", "alerta": "Alerta · "}

_QSS_TABLA = (
    "QTableWidget { background: transparent; border: none; gridline-color: transparent; "
    f"font-family: {FUENTE_SERIF}; }}"
    "QTableWidget::item { padding: 3px 6px; border: none; }"
    "QHeaderView::section { background: transparent; border: none; "
    "border-bottom: 1px solid rgba(127,127,127,0.55); padding: 4px 6px; color: #a1a1aa; font-weight: 600; }"
)

_NADA = "(ninguna)"
_TODOS = "(todos juntos)"
_MAX_VALORES = 500      # elementos que se ofrecen en la lista «solo este elemento»
_MAX_FILAS_TABLA = 200


if pg is not None:
    class _GraficoEstatico(pg.PlotWidget):
        """Sin zoom ni arrastre: la rueda del mouse sigue desplazando la pantalla."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setMouseEnabled(False, False)

        def wheelEvent(self, ev):
            ev.ignore()


class _Hilo(QThread):
    """Calcula en segundo plano para que la ventana no se congele."""
    listo = Signal(object)
    fallo = Signal(str)

    def __init__(self, cfg, tablas, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._tablas = tablas

    def run(self):
        try:
            self.listo.emit(ejecutar(self._cfg, self._tablas))
        except ValueError as e:                      # mensajes ya escritos en español para el usuario
            self.fallo.emit(str(e))
        except Exception as e:                       # cualquier otra cosa: se avisa, no se cae la app
            self.fallo.emit(f"No pude calcularlo ({type(e).__name__}): {e}")


def _vaciar(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        elif item.layout() is not None:
            _vaciar(item.layout())


def _a_segundos(x):
    """Fechas -> segundos desde 1970 (lo que entiende el eje de fechas de pyqtgraph)."""
    return pd.to_datetime(np.asarray(x)).values.astype("datetime64[ns]").astype("int64") / 1e9


class PanelPalabras(QWidget):
    sugiere_modelo = Signal()      # el usuario pidió pasar a «Entrenar un modelo»

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.colors = host.colors
        self._cfg = ConfigPronostico()
        self._widgets = {}            # campo -> (widget, tipo)
        self._errores_form = []
        self._cargando = False
        self._firma_tablas = None
        self._hilo = None
        self._historial = []          # resumen de cada ejecución (se guarda en el proyecto; ver historial.py)
        self._resultado = None
        self._contexto_pdf = None        # foto de lo calculado (para el PDF)
        self._contexto_pendiente = None
        self._construir()
        self.aplicar_tema(self.colors)

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------
    def _titulo_chico(self, texto):
        lbl = QLabel(texto)
        lbl.setStyleSheet("color: #a1a1aa; font-weight: 600; font-size: 13px; letter-spacing: 1px;")
        return lbl

    def _seccion(self, numero, titulo):
        marco = QFrame()
        marco.setObjectName("seccionTufte")
        marco.setStyleSheet(f"#seccionTufte{{background: transparent; border: none; border-top: 1px solid {REGLA};}}")
        lay = QVBoxLayout(marco)
        lay.setContentsMargins(0, 16, 0, 10)
        lay.setSpacing(8)
        lay.addWidget(self._titulo_chico(f"{numero}   {titulo}"))
        return marco, lay

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

        titulo = QLabel("Preguntas sobre el futuro")
        titulo.setStyleSheet(f"font-size: 24px; font-family: {FUENTE_SERIF};")
        lay.addWidget(titulo)
        sub = QLabel("Elige una pregunta y Hadar responde siempre con un rango (nunca una cifra exacta), "
                     "diciendo qué asumió y qué tan confiable es.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        # Aviso discreto: solo aparece si la tabla no tiene fechas (las preguntas necesitan tiempo).
        self.lbl_sugerencia = QLabel('Esta tabla no tiene fechas. Para adivinar el valor de una columna, '
                                     'prueba <a href="modelo" style="color:#6366f1; text-decoration:none;">'
                                     'Entrenar un modelo</a>.')
        self.lbl_sugerencia.setObjectName("muted")
        self.lbl_sugerencia.setTextFormat(Qt.RichText)
        self.lbl_sugerencia.setWordWrap(True)
        self.lbl_sugerencia.setVisible(False)
        self.lbl_sugerencia.linkActivated.connect(lambda _l: self.sugiere_modelo.emit())
        lay.addWidget(self.lbl_sugerencia)

        self.lbl_vacio = QLabel("Carga una tabla para empezar.")
        self.lbl_vacio.setObjectName("muted")
        lay.addWidget(self.lbl_vacio)

        # 1. La pregunta
        self.card_pregunta, l1 = self._seccion("1", "¿Qué quieres saber?")
        self.combo_palabra = QComboBox()
        for clave in ORDEN_PALABRAS:
            self.combo_palabra.addItem(PALABRAS_CLAVE[clave]["nombre"], clave)
        self.combo_palabra.currentIndexChanged.connect(self._cambio_palabra)
        l1.addWidget(self.combo_palabra)
        self.lbl_pregunta = QLabel("")
        self.lbl_pregunta.setWordWrap(True)
        self.lbl_pregunta.setStyleSheet(f"font-size: 16px; font-family: {FUENTE_SERIF};")
        l1.addWidget(self.lbl_pregunta)
        self.lbl_ejemplos = QLabel("")
        self.lbl_ejemplos.setObjectName("muted")
        self.lbl_ejemplos.setWordWrap(True)
        l1.addWidget(self.lbl_ejemplos)
        lay.addWidget(self.card_pregunta)

        # 2. Los datos
        self.card_datos, l2 = self._seccion("2", "Con qué datos")
        fila_tabla = QHBoxLayout()
        fila_tabla.addWidget(QLabel("Tabla"))
        self.combo_tabla = QComboBox()
        self.combo_tabla.setMaximumWidth(360)
        self.combo_tabla.currentIndexChanged.connect(self._cambio_tabla)
        fila_tabla.addWidget(self.combo_tabla)
        self.btn_sugerir = QPushButton("Sugerir columnas")
        self.btn_sugerir.clicked.connect(self._sugerir_todo)
        fila_tabla.addWidget(self.btn_sugerir)
        fila_tabla.addStretch(1)
        l2.addLayout(fila_tabla)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(8)
        self.grid.setColumnStretch(2, 1)
        l2.addLayout(self.grid)
        nota = QLabel("* obligatorio")
        nota.setObjectName("muted")
        l2.addWidget(nota)
        lay.addWidget(self.card_datos)

        # 3. Calcular
        fila_calc = QHBoxLayout()
        self.btn_calcular = QPushButton("Calcular")
        self.btn_calcular.clicked.connect(self._calcular)
        fila_calc.addWidget(self.btn_calcular)
        self.lbl_estado = QLabel("")
        self.lbl_estado.setObjectName("muted")
        self.lbl_estado.setWordWrap(True)
        fila_calc.addWidget(self.lbl_estado, 1)
        lay.addLayout(fila_calc)

        # Resultado
        self.card_resultado, self.res_lay = self._seccion("3", "Respuesta")
        self.card_resultado.setVisible(False)
        lay.addWidget(self.card_resultado)
        lay.addStretch(1)
        self._actualizar_pregunta()

    # ------------------------------------------------------------------
    # Ciclo de vida (misma interfaz que PanelPrediccion)
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        self.colors = colors
        self.setStyleSheet(estilo_tarjeta(colors) + f" QLabel {{ font-family: {FUENTE_SERIF}; }}")
        self.btn_calcular.setStyleSheet(
            f"QPushButton{{background: transparent; color: {ACENTO}; font-weight: 600; padding: 7px 22px; "
            f"border: 1px solid {ACENTO}; border-radius: 3px;}} "
            f"QPushButton:hover{{background: rgba(99,102,241,0.16);}} "
            f"QPushButton:disabled{{color: rgba(127,127,127,0.6); border-color: rgba(127,127,127,0.35);}}")

    def notificar_cambio(self):
        """Solo se rehace la pantalla si cambiaron las TABLAS (no en cada filtro o indicador)."""
        if self._firma() != self._firma_tablas and self.isVisible():
            self._refrescar_todo()

    def showEvent(self, event):
        super().showEvent(event)
        if self._firma() != self._firma_tablas:
            self._refrescar_todo()

    def _tablas(self):
        return getattr(self.host, "tablas", None) or {}

    def _firma(self):
        return tuple((n, df.shape, tuple(map(str, df.columns))) for n, df in self._tablas().items())

    def config_a_dict(self) -> dict:
        # Si la pestaña nunca se mostró, los widgets están vacíos: leerlos borraría
        # la configuración que el proyecto trae guardada.
        if self._firma_tablas is not None:
            self._leer_formulario()
        d = self._cfg.to_dict()
        d["historial"] = list(self._historial)
        return d

    def cargar_config(self, datos):
        self._cfg = ConfigPronostico.from_dict(datos)
        self._historial = limpiar_historial((datos or {}).get("historial") if isinstance(datos, dict) else None)
        if self._cfg.palabra not in PALABRAS_CLAVE:
            self._cfg.palabra = "proyeccion"
        self._resultado = None
        self._firma_tablas = None
        self._widgets.clear()
        self.card_resultado.setVisible(False)
        if self.isVisible():
            self._refrescar_todo()

    # ------------------------------------------------------------------
    # Rellenar la pantalla
    # ------------------------------------------------------------------
    def _df(self):
        return self._tablas().get(self._cfg.tabla)

    def _columnas(self):
        df = self._df()
        return [str(c) for c in df.columns] if df is not None else []

    def _refrescar_todo(self):
        self._firma_tablas = self._firma()
        self._resultado = None
        self.card_resultado.setVisible(False)
        tablas = self._tablas()
        hay = bool(tablas)
        self.lbl_vacio.setVisible(not hay)
        for w in (self.card_pregunta, self.card_datos, self.btn_calcular):
            w.setVisible(hay)
        if not hay:
            return
        self._cargando = True
        nombres = list(tablas.keys())
        self.combo_palabra.setCurrentIndex(max(self.combo_palabra.findData(self._cfg.palabra), 0))
        self._cfg.palabra = self.combo_palabra.currentData() or "proyeccion"
        self.combo_tabla.clear()
        self.combo_tabla.addItems(nombres)
        tabla = self._cfg.tabla if self._cfg.tabla in tablas else (
            getattr(self.host, "nombre_tabla_activa", None) or nombres[0])
        if tabla not in tablas:
            tabla = nombres[0]
        self.combo_tabla.setCurrentText(tabla)
        self._cfg.tabla = tabla
        self._cargando = False
        self._limpiar_columnas_invalidas()
        self._sugerir(solo_vacios=True, automatico=True)
        self._actualizar_pregunta()
        self._actualizar_sugerencia()
        self._reconstruir_formulario()

    def _actualizar_sugerencia(self):
        """Muestra el aviso solo si la tabla elegida no tiene ninguna columna de fecha."""
        df = self._df()
        sin_fechas = False
        if df is not None:
            try:
                sin_fechas = not any(tipo_columna(df[c]) in ("fecha", "fecha_texto") for c in df.columns)
            except Exception:
                sin_fechas = False
        self.lbl_sugerencia.setVisible(sin_fechas)

    def _actualizar_pregunta(self):
        spec = PALABRAS_CLAVE.get(self._cfg.palabra) or PALABRAS_CLAVE["proyeccion"]
        self.lbl_pregunta.setText(spec["pregunta"])
        self.lbl_ejemplos.setText("Por ejemplo: " + spec["ejemplos"])

    def _limpiar_columnas_invalidas(self):
        """Si la tabla cambió, se olvidan las columnas que ya no existen en ella."""
        cols = set(self._columnas())
        for nombre in ConfigPronostico.__dataclass_fields__:
            v = getattr(self._cfg, nombre)
            if nombre.startswith("columna_") and nombre != "columnas_explicativas":
                if v is not None and v not in cols:
                    setattr(self._cfg, nombre, None)
        self._cfg.columnas_explicativas = [c for c in (self._cfg.columnas_explicativas or []) if c in cols]
        if self._cfg.columna_entidad is None:
            self._cfg.entidad = None

    def _sugerir(self, solo_vacios=True, automatico=False):
        """Propone columnas. Automático: solo lo básico (fecha y valor, o lo obligatorio).
        Con el botón «Sugerir columnas»: todo lo que la pregunta pueda usar."""
        df = self._df()
        if df is None:
            return
        spec = PALABRAS_CLAVE[self._cfg.palabra]
        campos = {c["campo"]: c for c in spec["campos"]}
        sug = sugerir_configuracion(df, self._cfg.palabra)
        basicos = set()
        if spec["mecanismo"] == "serie":
            basicos |= {"columna_fecha", "columna_valor"}
        if self._cfg.palabra == "duracion":
            basicos |= {"columna_inicio", "columna_fin"}
        cols = set(self._columnas())
        for campo, valor in sug.items():
            if campo not in campos or valor not in cols:
                continue
            if automatico and not (campo in basicos or campos[campo]["obligatorio"]):
                continue
            actual = getattr(self._cfg, campo, None)
            if solo_vacios and actual is not None and actual in cols:
                continue
            setattr(self._cfg, campo, valor)

    def _sugerir_todo(self):
        self._leer_formulario()
        self._sugerir(solo_vacios=False, automatico=False)
        self._reconstruir_formulario()

    # ------------------------------------------------------------------
    # Formulario dinámico (se dibuja a partir del catálogo PALABRAS_CLAVE)
    # ------------------------------------------------------------------
    def _crear_widget(self, c, columnas):
        tipo, campo = c["tipo"], c["campo"]
        valor = getattr(self._cfg, campo, None)
        if tipo in ("fecha", "numerica", "columna"):
            w = QComboBox()
            w.addItem(_NADA)
            w.addItems(columnas)
            if valor in columnas:
                w.setCurrentText(valor)
            w.setMaximumWidth(360)
        elif tipo == "valor_de_columna":
            w = QComboBox()
            w.setEditable(True)
            w.setMaximumWidth(360)
            w.addItem(_TODOS)
            w.setProperty("valor_inicial", valor)
        elif tipo == "opcion":
            w = QComboBox()
            w.addItems([str(o) for o in c["opciones"]])
            w.setCurrentText(str(valor if valor is not None else c["defecto"]))
            w.setMaximumWidth(220)
        elif tipo == "entero":
            w = QSpinBox()
            w.setRange(0, 1_000_000)
            base = valor if valor is not None else c["defecto"]
            w.setValue(int(base or 0))
            w.setMaximumWidth(140)
        elif tipo == "numero":
            w = QLineEdit()
            base = valor if valor is not None else c["defecto"]
            w.setText("" if base is None else f"{base:g}")
            w.setPlaceholderText("número")
            w.setMaximumWidth(160)
        elif tipo == "texto":
            w = QLineEdit()
            w.setText("" if valor is None else str(valor))
            w.setMaximumWidth(260)
        elif tipo == "multi_columna":
            w = QListWidget()
            w.setMaximumHeight(120)
            w.setMaximumWidth(360)
            marcadas = set(valor or [])
            for col in columnas:
                it = QListWidgetItem(col)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if col in marcadas else Qt.Unchecked)
                w.addItem(it)
        else:                                   # tipo desconocido: texto simple
            w = QLineEdit("" if valor is None else str(valor))
        if c.get("ayuda"):
            w.setToolTip(c["ayuda"])
        return w

    def _reconstruir_formulario(self):
        _vaciar(self.grid)
        self._widgets.clear()
        spec = PALABRAS_CLAVE.get(self._cfg.palabra)
        if spec is None or self._df() is None:
            return
        cols = self._columnas()
        fila = 0
        for c in spec["campos"]:
            if c["campo"] in self._widgets:
                continue
            w = self._crear_widget(c, cols)
            self._widgets[c["campo"]] = (w, c["tipo"])
            etiqueta = QLabel(c["etiqueta"] + (" *" if c["obligatorio"] else ""))
            self.grid.addWidget(etiqueta, fila, 0, Qt.AlignTop)
            self.grid.addWidget(w, fila, 1, Qt.AlignTop)
            if c.get("ayuda"):
                ayuda = QLabel(c["ayuda"])
                ayuda.setObjectName("muted")
                ayuda.setWordWrap(True)
                self.grid.addWidget(ayuda, fila, 2, Qt.AlignTop)
            fila += 1
        # «Solo este elemento» depende de la columna de elemento elegida.
        if "entidad" in self._widgets and "columna_entidad" in self._widgets:
            self._widgets["columna_entidad"][0].currentIndexChanged.connect(self._llenar_valores_entidad)
            self._llenar_valores_entidad()

    def _llenar_valores_entidad(self, *_):
        par_e, par_c = self._widgets.get("entidad"), self._widgets.get("columna_entidad")
        df = self._df()
        if not par_e or not par_c or df is None:
            return
        combo, combo_col = par_e[0], par_c[0]
        previo = combo.property("valor_inicial") or (combo.currentText() if combo.currentText() != _TODOS else None)
        combo.setProperty("valor_inicial", None)
        combo.clear()
        combo.addItem(_TODOS)
        col = combo_col.currentText()
        if col != _NADA and col in df.columns:
            try:
                vals = sorted(df[col].dropna().astype(str).unique().tolist())[:_MAX_VALORES]
                combo.addItems(vals)
            except Exception:
                pass
        if previo and combo.findText(str(previo)) >= 0:
            combo.setCurrentText(str(previo))

    def _leer_formulario(self):
        """Pasa lo que hay en pantalla a self._cfg. Anota los errores de escritura (números mal puestos)."""
        self._errores_form = []
        if self.combo_tabla.count():
            self._cfg.tabla = self.combo_tabla.currentText() or self._cfg.tabla
        etiquetas = {c["campo"]: c["etiqueta"] for c in (PALABRAS_CLAVE.get(self._cfg.palabra) or {}).get("campos", [])}
        for campo, (w, tipo) in list(self._widgets.items()):
            try:
                if tipo in ("fecha", "numerica", "columna"):
                    t = w.currentText()
                    valor = None if t == _NADA else t
                elif tipo == "valor_de_columna":
                    t = w.currentText().strip()
                    valor = None if (not t or t == _TODOS) else t
                elif tipo == "opcion":
                    valor = w.currentText()
                elif tipo == "entero":
                    valor = int(w.value())
                elif tipo == "numero":
                    t = w.text().strip().replace(",", ".")
                    if not t:
                        valor = None
                    else:
                        try:
                            valor = float(t)
                        except ValueError:
                            valor = None
                            self._errores_form.append(f"«{etiquetas.get(campo, campo)}» debe ser un número.")
                elif tipo == "multi_columna":
                    valor = [w.item(i).text() for i in range(w.count()) if w.item(i).checkState() == Qt.Checked]
                else:
                    t = w.text().strip()
                    valor = t or None
            except RuntimeError:        # el widget ya se destruyó (cambio de pregunta): no hay nada que leer
                continue
            setattr(self._cfg, campo, valor)

    # ------------------------------------------------------------------
    # Cambios del usuario
    # ------------------------------------------------------------------
    def _cambio_palabra(self, _i=None):
        if self._cargando:
            return
        self._leer_formulario()
        nueva = self.combo_palabra.currentData()
        if not nueva or nueva == self._cfg.palabra:
            return
        vieja = PALABRAS_CLAVE.get(self._cfg.palabra, {}).get("campos", [])
        defectos_viejos = {c["campo"]: c["defecto"] for c in vieja}
        self._cfg.palabra = nueva
        # Un número que aún valía el valor por defecto de la pregunta anterior (por ejemplo, 30 días)
        # pasa al valor por defecto de la nueva, en vez de arrastrarse.
        for c in PALABRAS_CLAVE[nueva]["campos"]:
            if c["tipo"] in ("numero", "entero") and c["defecto"] is not None:
                actual = getattr(self._cfg, c["campo"], None)
                if actual is None or actual == defectos_viejos.get(c["campo"]):
                    setattr(self._cfg, c["campo"], c["defecto"])
        self._sugerir(solo_vacios=True, automatico=True)
        self._resultado = None
        self.card_resultado.setVisible(False)
        self.lbl_estado.setText("")
        self._actualizar_pregunta()
        self._reconstruir_formulario()

    def _cambio_tabla(self, _i=None):
        if self._cargando:
            return
        self._leer_formulario()          # lee lo de la tabla anterior antes de cambiarla
        self._cfg.tabla = self.combo_tabla.currentText()
        self._limpiar_columnas_invalidas()
        self._sugerir(solo_vacios=True, automatico=True)
        self._resultado = None
        self.card_resultado.setVisible(False)
        self._actualizar_sugerencia()
        self._reconstruir_formulario()

    # ------------------------------------------------------------------
    # Calcular
    # ------------------------------------------------------------------
    def _calcular(self):
        if self._hilo is not None:
            return
        self._leer_formulario()
        if self._errores_form:
            self.lbl_estado.setText(" ".join(self._errores_form))
            return
        cfg = ConfigPronostico.from_dict(self._cfg.to_dict())      # copia: el formulario puede cambiar mientras calcula
        panel_aj = getattr(self.host, "panel_ajustes", None)       # decisiones del proyecto (subpestaña «Ajustes»)
        if panel_aj is not None:
            aplicar_ajustes(cfg, panel_aj.ajustes)
        df = self._tablas().get(cfg.tabla)
        if df is None:
            self.lbl_estado.setText("Elige la tabla con los datos.")
            return
        self._contexto_pendiente = ContextoPregunta(
            config=cfg.to_dict(), n_filas=len(df), n_columnas=df.shape[1],
            calculado_en=datetime.now().strftime("%d/%m/%Y %H:%M"))
        self.btn_calcular.setEnabled(False)
        self.lbl_estado.setText("Calculando…")
        self.card_resultado.setVisible(False)
        self._hilo = _Hilo(cfg, {cfg.tabla: df}, self)
        self._hilo.listo.connect(self._fin_calculo)
        self._hilo.fallo.connect(self._fallo_calculo)
        self._hilo.finished.connect(self._hilo_terminado)
        self._hilo.start()

    def _hilo_terminado(self):
        if self._hilo is not None:
            self._hilo.deleteLater()
        self._hilo = None
        self.btn_calcular.setEnabled(True)

    def _fallo_calculo(self, mensaje):
        self.lbl_estado.setText(mensaje)

    def _fin_calculo(self, r):
        ctx = self._contexto_pendiente
        try:    # compara con la última vez que se hizo este mismo análisis y guarda el resumen en el proyecto
            avisos = registrar_y_comparar(self._historial, ctx.config, r, getattr(ctx, "n_filas", None))
            r.advertencias = list(avisos) + list(r.advertencias)
        except Exception:
            pass                # el historial nunca debe impedir mostrar el resultado
        self._resultado = r
        self._contexto_pdf = ctx
        self.lbl_estado.setText("")
        self._mostrar_resultado(r)

    # ------------------------------------------------------------------
    # Mostrar el resultado
    # ------------------------------------------------------------------
    def _parrafo(self, texto, muted=False, tam=None):
        lbl = QLabel(texto)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if muted:
            lbl.setObjectName("muted")
        if tam:
            lbl.setStyleSheet(f"font-size: {tam}px; font-family: {FUENTE_SERIF};")
        return lbl

    def _mostrar_resultado(self, r):
        # Se conserva el título de la sección (primer elemento) y se borra lo demás.
        while self.res_lay.count() > 1:
            item = self.res_lay.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                _vaciar(item.layout())

        color = COLOR_NIVEL.get(r.nivel, ACENTO)
        rotulo = ROTULO_NIVEL.get(r.nivel, "")
        titular = QLabel(f'<span style="color:{color}">●</span> '
                         f'<span style="color:{color}">{_esc(rotulo)}</span>{_esc(r.titular)}')
        titular.setTextFormat(Qt.RichText)
        titular.setWordWrap(True)
        titular.setTextInteractionFlags(Qt.TextSelectableByMouse)
        titular.setStyleSheet(f"font-size: 19px; font-family: {FUENTE_SERIF};")
        self.res_lay.addWidget(titular)

        if r.por_que_confianza or r.confianza:
            self.res_lay.addWidget(self._parrafo(
                f"Confianza {r.confianza}" + (f": {r.por_que_confianza}" if r.por_que_confianza else ""), muted=True))

        if r.tarjetas:
            fila = QHBoxLayout()
            fila.setSpacing(28)
            for valor, titulo, explicacion in r.tarjetas[:3]:
                col = QVBoxLayout()
                col.setSpacing(2)
                v = QLabel(str(valor))
                v.setStyleSheet(f"font-size: 26px; font-family: {FUENTE_SERIF};")
                col.addWidget(v)
                t = QLabel(str(titulo))
                t.setStyleSheet("font-weight: 600;")
                t.setWordWrap(True)
                col.addWidget(t)
                col.addWidget(self._parrafo(str(explicacion), muted=True))
                col.addStretch(1)
                fila.addLayout(col, 1)
            self.res_lay.addLayout(fila)

        for frase in r.detalle:
            self.res_lay.addWidget(self._parrafo(frase))

        g = self._grafico(r.grafico)
        if g is not None:
            self.res_lay.addWidget(g)

        if r.tabla is not None and len(r.tabla):
            if r.tabla_titulo:
                self.res_lay.addWidget(self._titulo_chico(r.tabla_titulo))
            self.res_lay.addWidget(self._tabla(r.tabla))

        if r.supuestos:
            self.res_lay.addWidget(self._titulo_chico("Qué asume Hadar"))
            self.res_lay.addWidget(self._parrafo(" ".join(r.supuestos), muted=True))
        for aviso in r.advertencias:
            self.res_lay.addWidget(self._parrafo(f'<span style="color:{AMBAR}">Ojo:</span> {_esc(aviso)}'))
            self.res_lay.itemAt(self.res_lay.count() - 1).widget().setTextFormat(Qt.RichText)

        fila_btn = QHBoxLayout()
        btn = QPushButton("Copiar resumen")
        btn.setMaximumWidth(160)
        btn.clicked.connect(lambda: QApplication.clipboard().setText(r.texto()))
        fila_btn.addWidget(btn)
        self.btn_pdf = QPushButton("Exportar PDF ▾")
        self.btn_pdf.setMaximumWidth(160)
        menu_pdf = QMenu(self.btn_pdf)
        accion_todo = menu_pdf.addAction("Todo (configuración y respuesta)")
        accion_res = menu_pdf.addAction("Solo los resultados")
        accion_todo.triggered.connect(lambda: self._exportar_pdf("todo"))
        accion_res.triggered.connect(lambda: self._exportar_pdf("resultados"))
        self.btn_pdf.setMenu(menu_pdf)
        # la flecha ya va en el texto: se oculta la que dibuja Qt para que no salgan dos
        self.btn_pdf.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0px; }")
        self.btn_pdf.setToolTip("Guarda la respuesta en PDF: todo (con la configuración) o solo los resultados.")
        fila_btn.addWidget(self.btn_pdf)
        fila_btn.addStretch(1)
        self.res_lay.addLayout(fila_btn)
        self.card_resultado.setVisible(True)

    def _exportar_pdf(self, modo):
        """modo = "todo" (configuración + respuesta) o "resultados" (solo la respuesta)."""
        if self._resultado is None or self._contexto_pdf is None:
            return
        carpeta = ""
        try:
            from .proyecto import ruta_carpeta_proyectos
            carpeta = ruta_carpeta_proyectos()
        except Exception:
            pass
        nombre = re.sub(r"[^A-Za-z0-9_-]+", "_", str(self._resultado.palabra)).strip("_") or "pregunta"
        sufijo = "completo" if modo == "todo" else "resultados"
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Exportar a PDF", os.path.join(carpeta, f"pregunta_{nombre}_{sufijo}.pdf"), "PDF (*.pdf)")
        if not ruta:
            return
        if not ruta.lower().endswith(".pdf"):
            ruta += ".pdf"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            exportar_pdf_pregunta(ruta, self._resultado, self._contexto_pdf, modo)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "No se pudo crear el PDF", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.lbl_estado.setText(f"PDF guardado: {ruta}")
        QMessageBox.information(self, "PDF guardado", f"Guardado en:\n{ruta}")

    def _tabla(self, df):
        extra = len(df) > _MAX_FILAS_TABLA
        d = df.head(_MAX_FILAS_TABLA)
        tabla = QTableWidget(len(d), len(d.columns))
        tabla.setStyleSheet(_QSS_TABLA)
        tabla.setShowGrid(False)
        tabla.setHorizontalHeaderLabels([str(c) for c in d.columns])
        tabla.verticalHeader().setVisible(False)
        tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tabla.setSelectionMode(QAbstractItemView.NoSelection)
        tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        tabla.horizontalHeader().setStretchLastSection(True)
        for i in range(len(d)):
            for j, col in enumerate(d.columns):
                v = d.iloc[i, j]
                numero = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                if isinstance(v, (pd.Timestamp, np.datetime64)):
                    t = pd.Timestamp(v)
                    txt = fecha_txt(t, con_hora=bool(t.hour or t.minute))
                elif numero:
                    txt = num_txt(v)
                elif v is None or (not isinstance(v, str) and pd.isna(v)):
                    txt = "—"
                else:
                    txt = str(v)
                it = QTableWidgetItem(txt)
                if numero:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tabla.setItem(i, j, it)
        tabla.setFixedHeight(min(len(d), 12) * 26 + 36)
        if extra:
            cont = QWidget()
            lay = QVBoxLayout(cont)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(tabla)
            lay.addWidget(self._parrafo(f"Se muestran las primeras {_MAX_FILAS_TABLA} filas de {len(df)}.", muted=True))
            return cont
        return tabla

    def _grafico(self, g):
        if pg is None or not g:
            return None
        try:
            fondo = self.colors.get("bg", "#000000") if isinstance(self.colors, dict) else "#000000"
            tipo = g.get("tipo")
            if tipo == "serie":
                plot = _GraficoEstatico(background=fondo, axisItems={"bottom": pg.DateAxisItem(orientation="bottom")})
            else:
                plot = _GraficoEstatico(background=fondo)
            plot.setMinimumHeight(260)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.showGrid(x=False, y=False)
            if g.get("etiqueta_y"):
                plot.setLabel("left", str(g["etiqueta_y"]))
            for eje in ("bottom", "left"):
                ax = plot.getAxis(eje)
                ax.setPen(pg.mkPen(GRIS, width=1))
                ax.setTextPen(pg.mkPen("#a1a1aa"))

            if tipo == "serie":
                xh, yh = _a_segundos(g["x_hist"]), np.asarray(g["y_hist"], dtype=float)
                plot.addItem(pg.PlotCurveItem(xh, yh, pen=pg.mkPen("#a1a1aa", width=1.5)))
                if g.get("linea") is not None:
                    lx, ly = g["linea"]
                    plot.addItem(pg.PlotCurveItem(_a_segundos(lx), np.asarray(ly, dtype=float),
                                                  pen=pg.mkPen(ACENTO, width=1.5, style=Qt.DashLine)))
                if g.get("x_fut") is not None:
                    xf = _a_segundos(g["x_fut"])
                    bajo = pg.PlotCurveItem(xf, np.asarray(g["y_bajo"], dtype=float), pen=pg.mkPen(None))
                    alto = pg.PlotCurveItem(xf, np.asarray(g["y_alto"], dtype=float), pen=pg.mkPen(None))
                    plot.addItem(pg.FillBetweenItem(bajo, alto, brush=pg.mkBrush(99, 102, 241, 70)))
                    plot.addItem(pg.PlotCurveItem(xf, np.asarray(g["y_fut"], dtype=float),
                                                  pen=pg.mkPen(ACENTO, width=2)))
                if g.get("umbral") is not None:
                    linea = pg.InfiniteLine(pos=float(g["umbral"]), angle=0,
                                            pen=pg.mkPen(AMBAR, width=1, style=Qt.DashLine),
                                            label=f"límite {num_txt(g['umbral'])}",
                                            labelOpts={"color": AMBAR, "position": 0.08})
                    plot.addItem(linea)
            elif tipo == "barras":
                y = np.asarray(g["y"], dtype=float)
                xs = np.arange(len(y))
                plot.addItem(pg.BarGraphItem(x=xs, height=y, width=0.6, brush=pg.mkBrush(99, 102, 241, 200),
                                             pen=pg.mkPen(None)))
                plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(GRIS, width=1)))
                plot.getAxis("bottom").setTicks([[(int(i), str(lab)) for i, lab in zip(xs, g["x"])]])
            else:
                return None
            return plot
        except Exception:
            return None      # el gráfico es un extra: si falla, el resto del resultado se muestra igual


def _esc(texto):
    return html.escape(str(texto))