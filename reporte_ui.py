"""
Pestaña Reporte de Hadar (antes vivía dentro de main_window.py): el lienzo
tipo "hoja" donde se arman informes con cajas de texto, gráficos, indicadores,
cálculos, etc., y se exportan a PDF.

Aquí vive todo lo de la pestaña: la barra de arriba (+ Agregar bloque,
Exportar a PDF), las hojas, la barra de formato de texto, el lienzo y cada
una de las formas de traer un bloque (_reporte_agregar_*). Cómo se dibuja cada
caja está en report_widgets.py (ReportBoxItem).

Cómo se conecta con la ventana principal (igual que Gráficos o Frecuencias):
    self.panel_reporte = PanelReporte(host=self)
`host` es la ventana principal: de ahí se leen los datos (df, filtered_df),
los gráficos, indicadores, métricas, etc. Esta pestaña nunca los modifica.

Estilo: el mismo de la pestaña Gráficos -- sin cajas ni rellenos, serifa
(Georgia), botones que son solo texto con una línea inferior en el color de
acento, y la hoja como lo único con presencia.
"""
import pandas as pd

from PySide6.QtCore import Qt, QRectF, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPageSize, QPainter, QPen, QPixmap, QTextCharFormat,
)
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGraphicsRectItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox,
    QPushButton, QSizePolicy, QTabBar, QToolButton, QVBoxLayout, QWidget,
)

from .config import COLOR_ACCENT
from .narrativa import QuestionAssistant
from .ontologia_ui import DialogoEsquemaOntologia
from .report_widgets import (
    ReportBoxItem, REPORTE_PAPEL_BORDE, REPORTE_PAPEL_BG, REPORTE_PAPEL_TINTA,
    REPORTE_COLOR_ROJO, REPORTE_FUENTES, REPORTE_TAMANOS,
)

FUENTE_SERIFA = '"Georgia", "Cambria", "Times New Roman", serif'


class PanelReporte(QWidget):
    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        layout = QVBoxLayout(self)

        # Barra de acciones. Antes eran 10 botones "+X" en fila; se juntan en
        # un solo botón desplegable para que el espacio no reviente el día
        # que se agregue un tipo de bloque más.
        barra = QHBoxLayout()
        self.btn_agregar = QToolButton()
        self.btn_agregar.setText("+ Agregar bloque")
        self.btn_agregar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_agregar.setObjectName("repAccion")
        self.btn_agregar.setCursor(Qt.CursorShape.PointingHandCursor)
        menu_agregar = QMenu(self.btn_agregar)
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
        self.btn_agregar.setMenu(menu_agregar)
        barra.addWidget(self.btn_agregar)

        self.btn_exportar = QPushButton("Exportar a PDF")
        self.btn_exportar.setObjectName("repAccion")
        self.btn_exportar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_exportar.clicked.connect(self._reporte_exportar_pdf)
        barra.addStretch()
        barra.addWidget(self.btn_exportar)
        layout.addLayout(barra)

        # Barra de hojas: cada hoja es una página independiente del reporte,
        # con su propio lienzo y su propia numeración de recuadros.
        barra_hojas = QHBoxLayout()
        self.reporte_tabbar = QTabBar()
        self.reporte_tabbar.setObjectName("repHojas")
        self.reporte_tabbar.setTabsClosable(True)
        self.reporte_tabbar.setExpanding(False)
        self.reporte_tabbar.setDrawBase(False)
        self.reporte_tabbar.currentChanged.connect(self._reporte_cambiar_hoja)
        self.reporte_tabbar.tabCloseRequested.connect(self._reporte_eliminar_hoja)
        self.reporte_tabbar.tabBarDoubleClicked.connect(self._reporte_renombrar_hoja)
        barra_hojas.addWidget(self.reporte_tabbar, stretch=1)
        self.btn_nueva_hoja = QPushButton("+ Hoja")
        self.btn_nueva_hoja.setObjectName("repSecundario")
        self.btn_nueva_hoja.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_nueva_hoja.clicked.connect(lambda: self._reporte_nueva_hoja())
        barra_hojas.addWidget(self.btn_nueva_hoja)
        layout.addLayout(barra_hojas)

        self._reporte_texto_activo = None
        self._crear_barra_formato_texto(layout)

        self.reporte_hojas = []
        self.reporte_scene = None
        self.reporte_view = QGraphicsView()
        self.reporte_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.reporte_view.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self.reporte_view, stretch=1)

        self.aplicar_tema(host.colors)
        self._reporte_nueva_hoja("Hoja 1")

    # ------------------------------------------------------------------
    # Tema
    # ------------------------------------------------------------------
    def aplicar_tema(self, colors):
        """Estilo de la pestaña (solo afecta a los widgets con nombre propio
        de aquí, no a los diálogos) y fondo del lienzo."""
        c = colors
        self.setStyleSheet(f"""
            QToolButton#repAccion, QPushButton#repAccion {{
                background: transparent; border: none; border-radius: 0;
                border-bottom: 1px solid {COLOR_ACCENT}; padding: 5px 8px;
                font-family: {FUENTE_SERIFA}; font-size: 14px;
                font-weight: 700; color: {COLOR_ACCENT};
            }}
            QToolButton#repAccion:hover, QPushButton#repAccion:hover {{
                background: transparent; border-bottom: 2px solid {COLOR_ACCENT};
            }}
            QToolButton#repAccion::menu-indicator {{ image: none; width: 0px; }}

            QPushButton#repSecundario {{
                background: transparent; border: none; border-radius: 0;
                border-bottom: 1px solid {c['border']}; padding: 5px 8px;
                font-family: {FUENTE_SERIFA}; font-size: 13px; color: {c['muted']};
            }}
            QPushButton#repSecundario:hover {{
                background: transparent; color: {c['text']};
                border-bottom: 1px solid {c['text']};
            }}

            QTabBar#repHojas::tab {{
                background: transparent; border: none;
                border-bottom: 2px solid transparent; padding: 5px 12px;
                font-family: {FUENTE_SERIFA}; font-size: 13px; color: {c['muted']};
            }}
            QTabBar#repHojas::tab:hover {{ color: {c['text']}; }}
            QTabBar#repHojas::tab:selected {{
                color: {c['text']}; border-bottom: 2px solid {COLOR_ACCENT};
            }}

            QFrame#repFormato {{
                background: transparent; border: none;
                border-bottom: 1px solid {c['border']};
            }}
            QFrame#repFormato QLabel {{
                background: transparent; border: none; color: {c['muted']};
                font-family: {FUENTE_SERIFA}; font-size: 12px;
            }}
            QPushButton#repFmt {{
                background: transparent; border: none; border-radius: 0;
                border-bottom: 2px solid transparent; padding: 2px 4px;
                color: {c['text']};
            }}
            QPushButton#repFmt:hover {{ border-bottom: 2px solid {c['border']}; }}
            QPushButton#repFmt:checked {{ border-bottom: 2px solid {c['text']}; }}
            QPushButton#repFmt:disabled {{ color: {c['muted']}; }}
        """)
        self.reporte_view.setBackgroundBrush(QBrush(QColor(c["bg"])))

    # -- barra fija de formato de texto (fuente/tamaño/negrita/rojo) --
    def _crear_barra_formato_texto(self, layout_reporte):
        """Fila de formato de texto SIEMPRE visible dentro de la pestaña
        (no flotante encima del lienzo). Antes vivía como un widget flotante
        hijo del viewport del QGraphicsView y se posicionaba a mano; eso
        generaba una carrera de foco con los QComboBox (su lista desplegable
        es una ventana aparte) que a veces impedía elegir fuente o tamaño.
        Como fila fija de la pestaña, ese problema desaparece."""
        barra = QFrame()
        barra.setObjectName("repFormato")
        fila = QHBoxLayout(barra)
        fila.setContentsMargins(8, 6, 8, 6)
        fila.setSpacing(6)

        self.lbl_barra_formato = QLabel("Formato de texto: elige un recuadro de texto (doble clic) para activarlo.")
        # OJO: un QLabel de una sola línea (sin wrap) le pide a su layout,
        # como mínimo, el ancho completo del texto, y empujaba TODO lo demás
        # de la pestaña fuera del ancho disponible. Ignored le dice al layout
        # "no me uses de referencia para el mínimo": el texto se puede recortar.
        self.lbl_barra_formato.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
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
        self.btn_negrita_texto.setObjectName("repFmt")
        self.btn_negrita_texto.setCheckable(True)
        self.btn_negrita_texto.setFixedWidth(28)
        self.btn_negrita_texto.setToolTip("Negrita")
        font_negrita = self.btn_negrita_texto.font()
        font_negrita.setBold(True)
        self.btn_negrita_texto.setFont(font_negrita)
        fila.addWidget(self.btn_negrita_texto)

        self.btn_rojo_texto = QPushButton("A")
        self.btn_rojo_texto.setObjectName("repFmt")
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
        item = ReportBoxItem("texto", rect, self.host.colors, ventana_principal=self)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_anomalia_desde_linea_tiempo(self, anomalia):
        rect = self._reporte_siguiente_posicion(280, 140)
        item = ReportBoxItem("lista", rect, self.host.colors, ventana_principal=self)
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
        item = ReportBoxItem("lista", rect, self.host.colors, ventana_principal=self)
        item.set_lista_text(titulo, html)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_linea_tiempo(self):
        if not hasattr(self.host, "panel_linea_tiempo") or self.host.df is None or self.host.df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        panel = self.host.panel_linea_tiempo

        def regenerar(panel=panel):
            return panel.snapshot()

        rect = self._reporte_siguiente_posicion(360, 300)
        item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self, titulo=panel.nombre())
        item.regenerar = regenerar
        item.set_pixmap(regenerar())
        self.reporte_scene.addItem(item)

    def _reporte_agregar_grafico(self):
        if not self.host.chart_panels:
            QMessageBox.information(
                self, "Sin gráficos",
                "Primero genera al menos un gráfico en la pestaña Gráficos."
            )
            return
        opciones = [f"{i + 1}. {p.etiqueta()}" for i, p in enumerate(self.host.chart_panels)]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir gráfico", "¿Cuál gráfico quieres llevar al reporte?", opciones, 0, False
        )
        if not ok or not elegido:
            return
        idx = opciones.index(elegido)
        panel = self.host.chart_panels[idx]
        titulo = panel.nombre() or None

        def regenerar(panel=panel):
            return panel.snapshot()

        rect = self._reporte_siguiente_posicion(340, 240)
        item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self, titulo=titulo)
        item.regenerar = regenerar
        item.set_pixmap(regenerar())
        self.reporte_scene.addItem(item)

    def _reporte_agregar_indicador(self):
        if not self.host.indicadores:
            QMessageBox.information(
                self, "Sin indicadores",
                "Todavía no has creado ningún indicador. Ve a la pestaña Indicadores para crear uno."
            )
            return
        nombres = [ind.nombre for ind in self.host.indicadores]
        elegido, ok = QInputDialog.getItem(
            self, "Elegir indicador", "¿Qué indicador quieres llevar al reporte?", nombres, 0, False
        )
        if not ok or not elegido:
            return
        indicador = next((ind for ind in self.host.indicadores if ind.nombre == elegido), None)
        if indicador is None:
            return

        def calcular(indicador=indicador):
            df_actual = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
            valor = indicador.calcular(df_actual)
            valor_crudo = float(valor) if isinstance(valor, (int, float)) and not pd.isna(valor) else None
            return indicador.nombre, indicador.formatear(valor), valor_crudo

        rect = self._reporte_siguiente_posicion(200, 110)
        item = ReportBoxItem("calculo", rect, self.host.colors, ventana_principal=self)
        item.regenerar = calcular
        titulo, valor_str, valor_crudo = calcular()
        item.set_calculo_text(titulo, valor_str, valor_crudo)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_narrativa(self):
        if not hasattr(self.host, "narrativa_browser"):
            return
        if not getattr(self.host, "_narrativa_actualizada", False):
            df = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
            if df is None or df.empty:
                QMessageBox.information(self, "Sin datos", "Carga datos primero.")
                return
            self.host._generar_narrativa()

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
                if not getattr(self.host, "_narrativa_actualizada", False):
                    self.host._generar_narrativa()
                self.host.narrativa_browser.repaint()
                QApplication.processEvents()
                return self.host.narrativa_browser.grab()

            rect = self._reporte_siguiente_posicion(360, 260)
            item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self)
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)
        else:
            def calcular():
                df_actual = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
                titulo = "Preguntas sugeridas"
                if df_actual is None or df_actual.empty:
                    return titulo, "-"
                preguntas = QuestionAssistant(df_actual, self.host._ultimas_anomalias).suggest()
                if not preguntas:
                    return titulo, "Sin preguntas sugeridas por ahora."
                cuerpo = "<br>".join(f"{i + 1}. {p}" for i, p in enumerate(preguntas))
                return titulo, cuerpo

            rect = self._reporte_siguiente_posicion(280, 220)
            item = ReportBoxItem("lista", rect, self.host.colors, ventana_principal=self)
            item.regenerar = calcular
            titulo, cuerpo = calcular()
            item.set_lista_text(titulo, cuerpo)
            self.reporte_scene.addItem(item)

    def _reporte_agregar_esquema(self):
        """Trae el esquema sugerido/confirmado entre tus tablas como una
        foto -- útil para dejar constancia visual de cómo está armada la
        base relacional dentro del informe final."""
        if len(self.host.tablas) < 2:
            QMessageBox.information(
                self, "Sin esquema para mostrar",
                "Carga 2 o más tablas relacionadas (ver botón \"Ver esquema\" en "
                "Datos) para poder agregar el esquema al reporte."
            )
            return

        def regenerar():
            dialogo_temp = DialogoEsquemaOntologia(self.host.tablas, self.host.colors,
                                                    relaciones_iniciales=self.host.relaciones_ontologia)
            return dialogo_temp.render_a_pixmap()

        rect = self._reporte_siguiente_posicion(420, 260)
        item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self, titulo="Esquema de tus tablas")
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
        item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self, con_refresh=False)
        item.set_pixmap(pixmap)
        self.reporte_scene.addItem(item)

    def _reporte_agregar_metrica(self):
        """Trae una de las 4 estadísticas -- o el gráfico de caja y bigote --
        de uno de los bloques de la pestaña Métricas."""
        if not self.host.metricas_bloques:
            QMessageBox.information(
                self, "Sin métricas", "Agrega al menos un bloque de métricas primero."
            )
            return

        columnas_disponibles = [b["columna"] for b in self.host.metricas_bloques]
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
            bloque = next((b for b in self.host.metricas_bloques if b["columna"] == columna), None)
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
                "imagen", rect, self.host.colors, ventana_principal=self, titulo=f"Caja y bigote — {columna}"
            )
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)
            return

        def calcular(elegido=elegido, columna=columna):
            df_actual = self.host.filtered_df
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
        item = ReportBoxItem("calculo", rect, self.host.colors, ventana_principal=self)
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
                self.host.panel_frecuencias.tabla_view.repaint()
                return self.host.panel_frecuencias.tabla_view.grab()

            rect = self._reporte_siguiente_posicion(340, 240)
            item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self)
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)

        elif elegido == "Gráfico de frecuencias (foto)":
            def regenerar():
                try:
                    import pyqtgraph.exporters as pg_exporters
                    exportador = pg_exporters.ImageExporter(self.host.panel_frecuencias.plot.getPlotItem())
                    imagen = exportador.export(toBytes=True)
                    pixmap = QPixmap.fromImage(imagen)
                    if not pixmap.isNull():
                        return pixmap
                except Exception:
                    pass
                return self.host.panel_frecuencias.plot.grab()

            rect = self._reporte_siguiente_posicion(340, 240)
            item = ReportBoxItem("imagen", rect, self.host.colors, ventana_principal=self)
            item.regenerar = regenerar
            item.set_pixmap(regenerar())
            self.reporte_scene.addItem(item)

        else:  # Top 10 más repetidos
            if self.host.filtered_df is None or self.host.filtered_df.empty:
                QMessageBox.information(self, "Sin datos", "Carga datos primero.")
                return
            columnas = list(self.host.filtered_df.columns)
            columna, ok = QInputDialog.getItem(
                self, "Elegir columna", "¿De qué columna quieres el top 10?", columnas, 0, False
            )
            if not ok or not columna:
                return

            def calcular(columna=columna):
                df_actual = self.host.filtered_df
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
            item = ReportBoxItem("lista", rect, self.host.colors, ventana_principal=self)
            item.regenerar = calcular
            titulo, cuerpo = calcular()
            item.set_lista_text(titulo, cuerpo)
            self.reporte_scene.addItem(item)

    def _reporte_agregar_calculo(self):
        if self.host.filtered_df is None or self.host.filtered_df.empty:
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

        for ind in self.host.indicadores:
            fuentes.append((
                f"Indicador: {ind.nombre}", ind.nombre,
                lambda ind=ind: ind.calcular(self.host.filtered_df if self.host.filtered_df is not None else self.host.df)
            ))

        for bloque in getattr(self.host, "metricas_bloques", []):
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
        item = ReportBoxItem("calculo", rect, self.host.colors, ventana_principal=self)
        item.regenerar = calcular
        titulo, valor_str, valor_crudo = calcular()
        item.set_calculo_text(titulo, valor_str, valor_crudo)
        self.reporte_scene.addItem(item)

    def _calcular_valor_metrica(self, nombre_metrica, columna):
        """Misma lógica que usa '+ Métrica' del Reporte, factorizada aparte
        para poder reutilizarla también al combinar valores en '+ Cálculo'."""
        df_actual = self.host.filtered_df
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
        if self.host.filtered_df is None or self.host.filtered_df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return
        columnas = list(self.host.filtered_df.columns)
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
            df_actual = self.host.filtered_df
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
        item = ReportBoxItem("calculo", rect, self.host.colors, ventana_principal=self)
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

        # Para el PDF, la hoja va "limpia": sin cajas seleccionadas (su marco
        # no es parte del reporte) y sin el borde gris del papel.
        for hoja in self.reporte_hojas:
            hoja["scene"].clearSelection()
            hoja["pagina"].setPen(QPen(Qt.PenStyle.NoPen))
        painter = QPainter(printer)
        try:
            for i, hoja in enumerate(self.reporte_hojas):
                if i > 0:
                    printer.newPage()
                hoja["scene"].render(painter)
        finally:
            painter.end()
            for hoja in self.reporte_hojas:
                hoja["pagina"].setPen(QPen(QColor(REPORTE_PAPEL_BORDE), 1))

        n_hojas = len(self.reporte_hojas)
        QMessageBox.information(
            self, "Listo",
            f"Reporte exportado a:\n{path}\n({n_hojas} hoja{'s' if n_hojas != 1 else ''})"
        )
