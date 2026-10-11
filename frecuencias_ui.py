"""
Pestaña Frecuencias de Hadar (antes vivía dentro de main_window.py).

Muestra, para las columnas que el usuario elija, cuántas veces se repite
cada valor: una tabla ordenada de más a menos frecuente y un gráfico de
barras con los valores más repetidos de la primera columna elegida.

Estilo "Tufte": sin cajas ni cuadrículas, solo líneas finas; cifras
grandes en tipografía con serifa; texto en gris para lo secundario; y UN
solo color de énfasis (el acento de Hadar) para lo que importa. El rojo
queda reservado para las alarmas.

Para no recargar la pantalla, las columnas NO se listan todas a la vez:
hay un solo botón ("Columnas: Producto ▾") que despliega la lista con un
buscador. En el gráfico las barras no llevan nombre debajo: al apretar una
barra aparece su detalle (valor completo, veces y porcentaje).

Cómo se conecta con la ventana principal (igual que Panel o Predicción):
    self.panel_frecuencias = PanelFrecuencias(host=self, ...)
y la ventana principal solo llama a:
    poblar_columnas(cols)        al cargar/cambiar datos
    actualizar()                 para recalcular tabla, cifras y gráfico
    actualizar_resumen_anomalias()
    marcar_alarma(bool)          para pintar la alarma de la cifra principal
    aplicar_tema(colors)         al cambiar de tema claro/oscuro
    columnas_seleccionadas() / seleccionar_columnas([...])
Para el Reporte se exponen .tabla_view y .plot.
"""
import html
import time
from functools import reduce

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QPoint, QTimer, QEvent, QIdentityProxyModel
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QAbstractItemView, QTableView, QHeaderView, QFrame, QSizePolicy,
    QPushButton, QLineEdit, QApplication,
)

from .config import COLOR_ACCENT, COLOR_DANGER
from .table_model import PandasTableModel

# Cuántos valores (los más repetidos) se dibujan en el gráfico.
MAX_BARRAS = 20
# Cuántas filas de la tabla se muestran POR COLUMNA (las más frecuentes). Una
# columna tipo ID o fecha-hora puede tener cientos de miles de valores
# distintos; mostrarlos todos congelaba la app y nadie los lee. Los números
# de las cifras (moda, categorías únicas) siempre se calculan con TODOS.
MAX_FILAS_TABLA = 500
# Cuántos nombres de columna caben en el botón antes de resumir ("5 elegidas").
MAX_NOMBRES_EN_BOTON = 3
MAX_CARACTERES_BOTON = 60

FUENTE_SERIFA = '"Georgia", "Cambria", "Times New Roman", serif'


def _rgba(color_hex, alfa):
    """'#6366f1' + 0.12 -> 'rgba(99, 102, 241, 0.12)' para usar en QSS."""
    c = QColor(color_hex)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alfa})"


class _ProxyAlineacion(QIdentityProxyModel):
    """Alinea a la derecha las columnas numéricas (Frecuencia y % del
    Total) tanto en las celdas como en su encabezado, sin tocar
    PandasTableModel. Las dos primeras columnas (Columna, Valor) quedan
    a la izquierda."""

    PRIMERA_COLUMNA_NUMERICA = 2

    def _alineacion(self, columna):
        if columna >= self.PRIMERA_COLUMNA_NUMERICA:
            return Qt.AlignRight | Qt.AlignVCenter
        return Qt.AlignLeft | Qt.AlignVCenter

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.TextAlignmentRole and index.isValid():
            return self._alineacion(index.column())
        return super().data(index, role)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.TextAlignmentRole and orientation == Qt.Horizontal:
            return self._alineacion(section)
        return super().headerData(section, orientation, role)


class PanelFrecuencias(QWidget):
    def __init__(self, host, valor_vacio="(vacíos)", etiquetas_anomalia=None, parent=None):
        super().__init__(parent)
        self.host = host
        self.setAttribute(Qt.WA_StyledBackground, True)
        # Texto que representa "celda vacía" (el mismo que usa el filtro de
        # Datos) y nombres legibles de cada tipo de anomalía; los define
        # main_window porque también los usa la pestaña Datos.
        self.valor_vacio = valor_vacio
        self.etiquetas_anomalia = etiquetas_anomalia or {}
        # Estado del gráfico (lo que se está dibujando y la barra apretada).
        self._ultimos_conteos = None
        self._ultima_columna = ""
        self._ultimo_total = 0
        self._n_columnas_elegidas = 1
        self._barras_info = None      # {"etiquetas": [...], "valores": [...]}
        self._barra_sel = None        # índice de la barra apretada, o None
        self._construir()
        self.aplicar_tema(host.colors)

    # ------------------------------------------------------------------
    # Construcción de la pantalla
    # ------------------------------------------------------------------
    def _construir(self):
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(28, 22, 28, 16)
        raiz.setSpacing(14)

        # Título + una línea que explica qué se está viendo.
        self.lbl_titulo = QLabel("Frecuencias")
        self.lbl_titulo.setObjectName("freqTitulo")
        raiz.addWidget(self.lbl_titulo)
        self.lbl_subtitulo = QLabel(
            "Cuántas veces aparece cada valor en las columnas que elijas."
        )
        self.lbl_subtitulo.setObjectName("freqSubtitulo")
        raiz.addWidget(self.lbl_subtitulo)

        # Selector de columnas: UN botón que resume lo elegido y despliega
        # la lista completa (con buscador) solo cuando se aprieta.
        fila_columnas = QHBoxLayout()
        self.btn_columnas = QPushButton("Columnas: ninguna  ▾")
        self.btn_columnas.setObjectName("freqBotonColumnas")
        self.btn_columnas.setCursor(Qt.PointingHandCursor)
        self.btn_columnas.setToolTip("Elige una o más columnas para contar sus valores.")
        self.btn_columnas.setMinimumWidth(240)
        self.btn_columnas.clicked.connect(self._abrir_selector)
        fila_columnas.addWidget(self.btn_columnas)
        fila_columnas.addStretch(1)
        raiz.addLayout(fila_columnas)

        self._construir_popup_columnas()

        # Cifras principales (sin cajas): el dato grande, su rótulo chico.
        fila_datos = QHBoxLayout()
        fila_datos.setSpacing(36)
        self.fig_moda, self.lbl_moda, self.lbl_moda_alarma = self._crear_cifra(
            "MÁS FRECUENTE (MODA)", con_alarma=True
        )
        self.fig_menos, self.lbl_menos, _ = self._crear_cifra("MENOS FRECUENTE")
        self.fig_unicos, self.lbl_unicos, _ = self._crear_cifra("CATEGORÍAS ÚNICAS")
        for fig in (self.fig_moda, self.fig_menos, self.fig_unicos):
            fila_datos.addWidget(fig)
        fila_datos.addStretch(1)
        raiz.addLayout(fila_datos)

        # Resumen de anomalías (según el último informe de Narrativa):
        # no depende de las columnas elegidas arriba.
        self.lbl_anomalias = QLabel("ANOMALÍAS (ÚLTIMO INFORME DE NARRATIVA)")
        self.lbl_anomalias.setObjectName("freqEtiqueta")
        raiz.addWidget(self.lbl_anomalias)

        fila_anomalias = QHBoxLayout()
        fila_anomalias.setSpacing(36)
        self.fig_anom_top, self.lbl_anom_top, _ = self._crear_cifra("ANOMALÍA MÁS FRECUENTE")
        self.fig_anom_col, self.lbl_anom_col, _ = self._crear_cifra("COLUMNA MÁS AFECTADA")
        for fig in (self.fig_anom_top, self.fig_anom_col):
            fila_anomalias.addWidget(fig)
        fila_anomalias.addStretch(1)
        raiz.addLayout(fila_anomalias)
        self.lbl_anom_top.setText("Genera Narrativa primero")
        self.lbl_anom_col.setText("Genera Narrativa primero")

        # Aviso cuando no hay nada que mostrar (sin datos o sin columna).
        self.lbl_vacio = QLabel("")
        self.lbl_vacio.setObjectName("freqVacio")
        self.lbl_vacio.setWordWrap(True)
        raiz.addWidget(self.lbl_vacio)

        # Cuerpo: tabla ordenada por frecuencia a la izquierda, gráfico a
        # la derecha (misma arquitectura de siempre).
        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(28)

        self.tabla_model = PandasTableModel()
        self._proxy = _ProxyAlineacion(self)
        self._proxy.setSourceModel(self.tabla_model)
        self.tabla_view = QTableView()
        self.tabla_view.setObjectName("freqTabla")
        self.tabla_view.setModel(self._proxy)
        self.tabla_view.setAlternatingRowColors(False)
        self.tabla_view.setShowGrid(False)
        self.tabla_view.setFrameShape(QFrame.NoFrame)
        self.tabla_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tabla_view.verticalHeader().setVisible(False)
        self.tabla_view.verticalHeader().setDefaultSectionSize(28)
        self.tabla_view.horizontalHeader().setHighlightSections(False)
        self.tabla_view.horizontalHeader().setStretchLastSection(False)
        self.tabla_view.clicked.connect(self.host._mostrar_celda_en_status)
        columna_tabla = QVBoxLayout()
        columna_tabla.setSpacing(6)
        columna_tabla.addWidget(self.tabla_view, stretch=1)
        self.lbl_truncado = QLabel("")
        self.lbl_truncado.setObjectName("freqVacio")
        self.lbl_truncado.setWordWrap(True)
        self.lbl_truncado.setVisible(False)
        columna_tabla.addWidget(self.lbl_truncado)
        cuerpo.addLayout(columna_tabla, stretch=3)

        columna_grafico = QVBoxLayout()
        columna_grafico.setSpacing(6)
        self.lbl_grafico = QLabel("")
        self.lbl_grafico.setObjectName("freqEtiqueta")
        columna_grafico.addWidget(self.lbl_grafico)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumWidth(320)
        self.plot.setFrameShape(QFrame.NoFrame)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Gráfico fijo (sin zoom ni arrastre): lo que se interactúa es la
        # barra, apretándola para ver su detalle.
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setCursor(Qt.PointingHandCursor)
        self.plot.scene().sigMouseClicked.connect(self._clic_grafico)
        columna_grafico.addWidget(self.plot, stretch=1)
        cuerpo.addLayout(columna_grafico, stretch=2)

        raiz.addLayout(cuerpo, stretch=1)
        self._mostrar_aviso_vacio("Carga datos y elige una o más columnas.")

    def _construir_popup_columnas(self):
        """Ventanita que se despliega bajo el botón de columnas: buscador +
        lista de columnas (clic = una sola; Ctrl + clic = varias)."""
        self.popup = QFrame(self, Qt.Popup)
        self.popup.setObjectName("freqPopup")
        self.popup.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self.popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.buscador = QLineEdit()
        self.buscador.setObjectName("freqBuscador")
        self.buscador.setPlaceholderText("Buscar columna…")
        self.buscador.setClearButtonEnabled(True)
        self.buscador.textChanged.connect(self._filtrar_columnas)
        lay.addWidget(self.buscador)

        self.lista = QListWidget()
        self.lista.setObjectName("freqColumnas")
        self.lista.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lista.setFixedHeight(230)
        self.lista.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lista.itemSelectionChanged.connect(self._al_cambiar_seleccion)
        lay.addWidget(self.lista)

    def _abrir_selector(self):
        self.popup.setFixedWidth(max(self.btn_columnas.width(), 300))
        abajo = self.btn_columnas.mapToGlobal(QPoint(0, self.btn_columnas.height() + 2))
        self.popup.move(abajo)
        self.buscador.clear()
        self.popup.show()
        self.buscador.setFocus()

    def _filtrar_columnas(self, texto):
        t = texto.strip().lower()
        for i in range(self.lista.count()):
            item = self.lista.item(i)
            item.setHidden(bool(t) and t not in item.text().lower())

    def _actualizar_texto_boton(self):
        elegidas = self.columnas_seleccionadas()
        unidas = ", ".join(elegidas)
        if not elegidas:
            resumen = "ninguna"
        elif len(elegidas) <= MAX_NOMBRES_EN_BOTON and len(unidas) <= MAX_CARACTERES_BOTON:
            resumen = unidas
        elif len(elegidas) == 1:
            resumen = unidas[:MAX_CARACTERES_BOTON - 1] + "…"
        else:
            resumen = f"{len(elegidas)} elegidas"
        self.btn_columnas.setText(f"Columnas: {resumen}  ▾")

    def _crear_cifra(self, etiqueta, con_alarma=False):
        """Una cifra principal: rótulo chico arriba, valor grande abajo.
        Devuelve (contenedor, label_valor, label_alarma|None)."""
        marco = QFrame()
        marco.setObjectName("freqCifra")
        marco.setProperty("alarma", False)
        marco.setMaximumWidth(360)
        lay = QVBoxLayout(marco)
        lay.setContentsMargins(10, 0, 0, 0)
        lay.setSpacing(2)

        lbl_etiqueta = QLabel(etiqueta)
        lbl_etiqueta.setObjectName("freqEtiqueta")
        lay.addWidget(lbl_etiqueta)

        lbl_valor = QLabel("-")
        lbl_valor.setObjectName("freqValor")
        lbl_valor.setWordWrap(True)
        lbl_valor.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(lbl_valor)

        lbl_alarma = None
        if con_alarma:
            lbl_alarma = QLabel("Alarma activa")
            lbl_alarma.setObjectName("freqAlarma")
            lbl_alarma.setVisible(False)
            lay.addWidget(lbl_alarma)
        return marco, lbl_valor, lbl_alarma

    # ------------------------------------------------------------------
    # Tema (claro / oscuro)
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        self._colors = colors
        bg = colors.get("bg", "#ffffff")
        texto = colors.get("text", "#111111")
        apagado = colors.get("muted", "#777777")
        borde = colors.get("border", "#dddddd")
        acento = COLOR_ACCENT
        peligro = COLOR_DANGER
        self.setStyleSheet(f"""
            PanelFrecuencias {{ background: {bg}; }}
            QLabel {{ background: transparent; }}
            QLabel#freqTitulo {{
                font-family: {FUENTE_SERIFA}; font-size: 24px; color: {texto};
            }}
            QLabel#freqSubtitulo {{
                font-family: {FUENTE_SERIFA}; font-size: 13px; font-style: italic;
                color: {apagado};
            }}
            QLabel#freqEtiqueta {{
                font-size: 10px; letter-spacing: 1px; color: {apagado};
            }}
            QLabel#freqValor {{
                font-family: {FUENTE_SERIFA}; font-size: 20px; color: {texto};
            }}
            QLabel#freqAlarma {{
                font-size: 11px; font-weight: 600; color: {peligro};
            }}
            QLabel#freqVacio {{
                font-family: {FUENTE_SERIFA}; font-size: 13px; font-style: italic;
                color: {apagado};
            }}
            QFrame#freqCifra {{
                background: transparent; border: none;
                border-left: 3px solid transparent;
            }}
            QFrame#freqCifra[alarma="true"] {{ border-left: 3px solid {peligro}; }}

            QPushButton#freqBotonColumnas {{
                background: transparent; border: none;
                border-bottom: 1px solid {borde};
                font-family: {FUENTE_SERIFA}; font-size: 14px;
                color: {acento}; text-align: left; padding: 5px 4px;
            }}
            QPushButton#freqBotonColumnas:hover {{ border-bottom: 1px solid {acento}; }}

            QFrame#freqPopup {{
                background: {bg}; border: 1px solid {borde};
            }}
            QLineEdit#freqBuscador {{
                background: transparent; border: none;
                border-bottom: 1px solid {borde}; padding: 4px 2px;
                font-family: {FUENTE_SERIFA}; font-size: 13px; color: {texto};
            }}
            QLineEdit#freqBuscador:focus {{ border-bottom: 1px solid {acento}; }}
            QListWidget#freqColumnas {{
                background: transparent; border: none; outline: 0;
                font-family: {FUENTE_SERIFA}; font-size: 13px; color: {apagado};
            }}
            QListWidget#freqColumnas::item {{
                background: transparent; color: {apagado}; padding: 4px 8px;
            }}
            QListWidget#freqColumnas::item:hover {{ color: {texto}; }}
            QListWidget#freqColumnas::item:selected {{
                background: {_rgba(acento, 0.10)}; color: {acento}; font-weight: 700;
            }}

            QTableView#freqTabla {{
                background: {bg}; border: none; outline: 0;
                gridline-color: transparent; color: {texto};
                font-family: {FUENTE_SERIFA}; font-size: 13px;
                selection-background-color: {_rgba(acento, 0.12)};
                selection-color: {texto};
            }}
            QTableView#freqTabla::item {{
                border: none; border-bottom: 1px solid {borde}; padding: 2px 6px;
            }}
            QTableView#freqTabla QHeaderView::section {{
                background: {bg}; color: {apagado}; border: none;
                border-bottom: 1px solid {texto}; padding: 4px 6px;
                font-size: 11px; font-weight: normal;
            }}
            QTableView#freqTabla QTableCornerButton::section {{
                background: {bg}; border: none;
            }}
        """)
        # Para que la propiedad [alarma] vuelva a evaluarse con los colores nuevos.
        self._repulir(self.fig_moda)
        # El gráfico se dibuja a mano, así que se repinta con el tema nuevo.
        self._dibujar_grafico()

    @staticmethod
    def _repulir(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    # ------------------------------------------------------------------
    # API para la ventana principal
    # ------------------------------------------------------------------
    def poblar_columnas(self, columnas):
        """Rellena la lista con las columnas actuales conservando lo que ya
        estaba elegido; si no había nada elegido, marca la primera.
        No recalcula (quien llama decide cuándo)."""
        self.lista.blockSignals(True)
        elegidas = [item.text() for item in self.lista.selectedItems()]
        self.lista.clear()
        for c in columnas:
            self.lista.addItem(QListWidgetItem(c))
        if elegidas:
            for i in range(self.lista.count()):
                item = self.lista.item(i)
                if item.text() in elegidas:
                    item.setSelected(True)
        elif self.lista.count():
            self.lista.item(0).setSelected(True)
        self.lista.blockSignals(False)
        self._actualizar_texto_boton()

    def columnas_seleccionadas(self):
        # Se ordenan como aparecen en la lista (el orden en que se hizo clic
        # no importa); así la "primera columna" del gráfico es predecible.
        return [
            self.lista.item(i).text()
            for i in range(self.lista.count())
            if self.lista.item(i).isSelected()
        ]

    def seleccionar_columnas(self, nombres):
        """Deja elegidas exactamente estas columnas (ej. tras renombrar una)
        y recalcula una sola vez."""
        self.lista.blockSignals(True)
        for i in range(self.lista.count()):
            item = self.lista.item(i)
            item.setSelected(item.text() in nombres)
        self.lista.blockSignals(False)
        self._al_cambiar_seleccion()

    def marcar_alarma(self, en_alarma):
        """Marca (o desmarca) la cifra principal cuando hay una regla de
        Alarma tipo Frecuencia activa sobre la columna elegida."""
        self.fig_moda.setProperty("alarma", bool(en_alarma))
        self.lbl_moda_alarma.setVisible(bool(en_alarma))
        self._repulir(self.fig_moda)

    # ------------------------------------------------------------------
    # Cálculo y dibujo
    # ------------------------------------------------------------------
    def _al_cambiar_seleccion(self):
        self._actualizar_texto_boton()
        self.actualizar()
        # Cambiar de columna no cambia ninguna alarma: solo hay que volver a
        # marcar (o desmarcar) la de esta pestaña. Antes se reconstruía la
        # pestaña Alarma completa en cada clic.
        df = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
        if df is not None:
            self.host._marcar_alarmas_en_frecuencias(df)

    def _mostrar_aviso_vacio(self, texto):
        self.lbl_vacio.setText(texto)
        self.lbl_vacio.setVisible(bool(texto))

    def _limpiar(self, aviso):
        self.tabla_model.set_dataframe(pd.DataFrame())
        for lbl in (self.lbl_moda, self.lbl_menos, self.lbl_unicos):
            lbl.setText("-")
        self._ultimos_conteos = None
        self._ultima_columna = ""
        self._barra_sel = None
        self.lbl_truncado.setVisible(False)
        self._dibujar_grafico()
        self._mostrar_aviso_vacio(aviso)

    def _contar_valores(self, serie):
        """Cuántas veces aparece cada valor de la columna, de mayor a menor,
        con los vacíos como una fila más. Los valores se pasan a texto solo
        una vez por valor DISTINTO (no por cada fila de la columna), y solo
        si todavía no son texto."""
        n_vacios = int(serie.isna().sum())
        try:
            conteos = serie.value_counts(dropna=True)  # ya viene de mayor a menor
        except TypeError:
            # Celdas con listas u otros objetos que no se pueden agrupar:
            # se pasan a texto primero.
            conteos = serie.dropna().astype(str).value_counts()
        if isinstance(serie.dtype, pd.CategoricalDtype):
            conteos = conteos[conteos > 0]  # sin categorías que nadie usa
        if conteos.index.inferred_type != "string":
            conteos.index = conteos.index.astype(str)
            if not conteos.index.is_unique:
                # Ej. el número 1 y el texto "1" quedan como el mismo valor.
                conteos = conteos.groupby(level=0).sum().sort_values(ascending=False, kind="stable")
        if n_vacios:
            # Se inserta en su lugar (la lista ya está ordenada) en vez de
            # volver a ordenar todo.
            pos = int((conteos.to_numpy() >= n_vacios).sum())
            conteos = pd.concat([
                conteos.iloc[:pos], pd.Series({self.valor_vacio: n_vacios}), conteos.iloc[pos:]
            ])
        return conteos

    def actualizar(self):
        t0 = time.perf_counter()
        self.actualizar_resumen_anomalias()

        df = self.host.filtered_df
        if df is None or df.empty:
            self._limpiar("Carga datos y elige una o más columnas.")
            return

        columnas = self.columnas_seleccionadas()
        if not columnas:
            self._limpiar("Elige al menos una columna para ver sus frecuencias.")
            return

        # 1) Contar los valores de cada columna elegida (lo más pesado).
        por_columna = []  # (columna, conteos de mayor a menor, total de filas)
        for col in columnas:
            if col not in df.columns:
                continue
            serie = df[col]
            if len(serie) == 0:
                continue
            conteos = self._contar_valores(serie)
            if len(conteos):
                por_columna.append((col, conteos, len(serie)))
        t_conteo = time.perf_counter()

        # 2) Tabla: solo los MAX_FILAS_TABLA valores más frecuentes de cada
        #    columna, con las columnas en orden alfabético.
        trozos = []
        recortadas = []
        for col, conteos, total in sorted(por_columna, key=lambda t: str(t[0])):
            top = conteos.head(MAX_FILAS_TABLA)
            if len(conteos) > len(top):
                recortadas.append((col, len(conteos)))
            trozos.append(pd.DataFrame({
                "Columna": col,
                "Valor": list(top.index),
                "Frecuencia": top.to_numpy().astype("int64"),
                "% del Total": [f"{v / total * 100:.2f}%" for v in top.to_numpy()],
            }))
        freq_df = pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame()
        self.tabla_model.set_dataframe(freq_df)
        self._ajustar_columnas_tabla()

        if recortadas:
            detalle = ", ".join(f"«{c}» ({n:,} distintos)" for c, n in recortadas[:3])
            if len(recortadas) > 3:
                detalle += "…"
            self.lbl_truncado.setText(
                f"La tabla muestra los {MAX_FILAS_TABLA} valores más frecuentes de cada "
                f"columna. Tienen más: {detalle}."
            )
            self.lbl_truncado.setVisible(True)
        else:
            self.lbl_truncado.setVisible(False)

        # 3) Cifras principales: se calculan con TODOS los valores, no solo
        #    con los de la tabla.
        if por_columna:
            _, c_moda, _ = max(por_columna, key=lambda t: t[1].iloc[0])
            _, c_menos, _ = min(por_columna, key=lambda t: t[1].iloc[-1])
            indices = [c.index for _, c, _ in por_columna]
            unicos = len(indices[0]) if len(indices) == 1 else len(reduce(lambda a, b: a.union(b), indices))
            self.lbl_moda.setText(f"{c_moda.index[0]} ({int(c_moda.iloc[0])})")
            self.lbl_menos.setText(f"{c_menos.index[-1]} ({int(c_menos.iloc[-1])})")
            self.lbl_unicos.setText(f"{unicos:,}")
            self._mostrar_aviso_vacio("")
        else:
            for lbl in (self.lbl_moda, self.lbl_menos, self.lbl_unicos):
                lbl.setText("-")
            self._mostrar_aviso_vacio("Las columnas elegidas no tienen valores para contar.")
        t_tabla = time.perf_counter()

        # 4) Gráfico: la primera columna elegida; si hay más de una se avisa
        #    en el rótulo para que no parezca que falta información.
        if por_columna:
            self._ultima_columna, self._ultimos_conteos, self._ultimo_total = por_columna[0]
        else:
            self._ultima_columna, self._ultimos_conteos, self._ultimo_total = "", None, 0
        self._n_columnas_elegidas = len(columnas)
        self._barra_sel = None
        self._dibujar_grafico()

        total_s = time.perf_counter() - t0
        if total_s > 0.25:
            print(
                f"[Hadar · tiempo] Frecuencias recalcular: {total_s:.2f} s "
                f"(contar valores {t_conteo - t0:.2f} s, tabla y cifras {t_tabla - t_conteo:.2f} s, "
                f"gráfico {time.perf_counter() - t_tabla:.2f} s; "
                f"{len(df):,} filas, {len(columnas)} columna(s))"
            )

    def _ajustar_columnas_tabla(self):
        cabecera = self.tabla_view.horizontalHeader()
        n = self._proxy.columnCount()
        if n < 4:
            return
        # Columna y números se ajustan a su contenido; "Valor" (que puede
        # ser largo) toma el resto del ancho.
        for i in range(n):
            cabecera.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        cabecera.setSectionResizeMode(1, QHeaderView.Stretch)

    def _dibujar_grafico(self):
        """Barras verticales sin nombres debajo. La barra apretada (o, si
        no hay ninguna apretada, la más alta) va en el color de énfasis y
        la apretada muestra su detalle encima."""
        c = self._colors
        p = self.plot
        p.clear()
        p.setBackground(c.get("bg", "#ffffff"))
        p.showGrid(x=False, y=False)
        p.hideAxis("bottom")
        p.hideAxis("left")
        p.invertY(False)

        conteos = self._ultimos_conteos
        if conteos is None or len(conteos) == 0:
            self._barras_info = None
            self.lbl_grafico.setText("")
            return

        top = conteos.sort_values(ascending=False).head(MAX_BARRAS)
        etiquetas = [str(v) for v in top.index]
        valores = np.asarray(top.values, dtype=float)
        n = len(valores)
        xs = np.arange(n, dtype=float)
        maximo = float(valores.max())
        self._barras_info = {"etiquetas": etiquetas, "valores": valores}
        if self._barra_sel is not None and self._barra_sel >= n:
            self._barra_sel = None

        # Un solo color de énfasis; el resto en gris.
        enfatizada = self._barra_sel if self._barra_sel is not None else 0
        gris = QColor(c.get("muted", "#777777"))
        gris.setAlpha(150)
        pinceles = [
            pg.mkBrush(COLOR_ACCENT) if i == enfatizada else pg.mkBrush(gris)
            for i in range(n)
        ]
        barras = pg.BarGraphItem(
            x=xs, height=valores, width=0.62,
            brushes=pinceles, pen=pg.mkPen(None),
        )
        p.addItem(barras)

        # Línea base finita para "apoyar" las barras.
        p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(c.get("muted", "#777777"), width=1)))

        # Número encima de cada barra (salvo la apretada: su detalle ya
        # incluye el número).
        fuente_num = QFont("Georgia", 9)
        for x, v, i in zip(xs, valores, range(n)):
            if i == self._barra_sel:
                continue
            txt = pg.TextItem(f"{int(v):,}", color=c.get("text", "#111111"), anchor=(0.5, 1))
            txt.setFont(fuente_num)
            txt.setPos(x, v)
            p.addItem(txt)

        if self._barra_sel is not None:
            self._dibujar_detalle(self._barra_sel, xs, valores, maximo, n, c)

        # Espacio arriba para que quepa el detalle sobre la barra más alta.
        p.setXRange(-0.6, n - 0.4, padding=0)
        p.setYRange(0, maximo * 1.45, padding=0)

        titulo = self._ultima_columna.upper()
        if self._n_columnas_elegidas > 1:
            titulo += " (PRIMERA ELEGIDA)"
        titulo += "  ·  TOCA UNA BARRA PARA VER SU DETALLE"
        self.lbl_grafico.setText(titulo)

    def _dibujar_detalle(self, i, xs, valores, maximo, n, c):
        etiqueta = self._barras_info["etiquetas"][i]
        freq = int(valores[i])
        pct = (freq / self._ultimo_total * 100) if self._ultimo_total else 0
        texto = (
            f"<div style='font-family: Georgia, serif; font-size: 11pt; "
            f"color: {c.get('text', '#111111')};'>"
            f"<b>{html.escape(etiqueta)}</b><br>"
            f"<span style='color: {c.get('muted', '#777777')};'>"
            f"{freq:,} veces · {pct:.2f}% del total</span></div>"
        )
        # Para que el recuadro no se salga por los costados: cerca del borde
        # izquierdo se ancla a su izquierda, cerca del derecho a su derecha.
        if n > 3 and i <= 1:
            ancla = (0, 1)
        elif n > 3 and i >= n - 2:
            ancla = (1, 1)
        else:
            ancla = (0.5, 1)
        detalle = pg.TextItem(
            html=texto, anchor=ancla,
            fill=pg.mkBrush(c.get("bg", "#ffffff")),
            border=pg.mkPen(c.get("border", "#dddddd")),
        )
        detalle.setPos(xs[i], valores[i] + maximo * 0.04)
        self.plot.addItem(detalle)

    def _clic_grafico(self, evento):
        """Apretar una barra muestra su detalle; apretarla otra vez, o
        apretar en cualquier espacio vacío del gráfico, lo oculta."""
        if evento.button() != Qt.LeftButton or not self._barras_info:
            return
        vb = self.plot.getPlotItem().vb
        pos = evento.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        punto = vb.mapSceneToView(pos)
        x, y = punto.x(), punto.y()
        valores = self._barras_info["valores"]
        n = len(valores)

        # Solo cuenta si el clic cae SOBRE la barra (con un poquito de
        # margen a los lados y un alto mínimo para que las barras muy
        # bajas también se puedan apretar).
        i = int(round(x))
        sobre_barra = False
        if 0 <= i < n and abs(x - i) <= 0.35:
            alto_clicable = max(float(valores[i]), float(valores.max()) * 0.04)
            sobre_barra = 0 <= y <= alto_clicable

        if sobre_barra:
            self._barra_sel = None if self._barra_sel == i else i
        else:
            self._barra_sel = None

        if self._barra_sel is not None:
            # Mientras haya un detalle abierto, un clic en cualquier otra
            # parte de la ventana también lo oculta (ver eventFilter).
            QApplication.instance().installEventFilter(self)
        # Se redibuja apenas termina de procesarse el clic (no dentro de él).
        QTimer.singleShot(0, self._dibujar_grafico)

    def eventFilter(self, obj, event):
        """Oculta el detalle de la barra cuando se hace clic fuera del
        gráfico. Solo está instalado mientras hay un detalle abierto."""
        if event.type() == QEvent.MouseButtonPress:
            if self._barra_sel is None:
                QApplication.instance().removeEventFilter(self)
            elif (
                isinstance(obj, QWidget)
                and obj is not self.plot
                and not self.plot.isAncestorOf(obj)
            ):
                self._barra_sel = None
                QApplication.instance().removeEventFilter(self)
                QTimer.singleShot(0, self._dibujar_grafico)
        return False

    # ------------------------------------------------------------------
    # Resumen de anomalías (lee lo último que calculó Narrativa)
    # ------------------------------------------------------------------
    def actualizar_resumen_anomalias(self):
        anomalias = getattr(self.host, "_ultimas_anomalias", [])
        if not anomalias:
            self.lbl_anom_top.setText("Genera Narrativa primero")
            self.lbl_anom_col.setText("Genera Narrativa primero")
            return

        # Se suma 'filas_afectadas' (no la cantidad de anomalías) para que
        # un tipo con miles de filas pese más que diez anomalías de una
        # sola fila: "más frecuente" = a cuántas filas toca.
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
            etiqueta = self.etiquetas_anomalia.get(tipo_top, tipo_top)
            self.lbl_anom_top.setText(f"{etiqueta} ({n_tipo:,})")
        else:
            self.lbl_anom_top.setText("-")

        if filas_por_columna:
            col_top, n_col = max(filas_por_columna.items(), key=lambda kv: kv[1])
            self.lbl_anom_col.setText(f"{col_top} ({n_col:,})")
        else:
            self.lbl_anom_col.setText("-")